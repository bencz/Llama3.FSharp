namespace Llama3.Backend.Cuda

open System
open ILGPU
open ILGPU.Runtime
open ILGPU.Runtime.Cuda
open Llama3.Core

/// Static description of a CUDA device.
type CudaDeviceInfo =
    { Index: int
      Name: string
      TotalMemory: int64
      Architecture: string
      Multiprocessors: int }

/// Kernel launchers, compiled on first use. Delegates are built straight from the kernel
/// functions (F# 11 direct delegate construction): the compiler checks every signature and the
/// delegate targets the static method itself, which is what ILGPU compiles.
type internal KernelSet(accelerator: Accelerator) =
    member val MatVecF32 =
        lazy (accelerator.LoadStreamKernel(Action<ArrayView<float32>, ArrayView<float32>, ArrayView<float32>, int, int>(MatVecKernels.matVecF32)))
    member val MatVecWordsShared =
        lazy (accelerator.LoadStreamKernel(Action<ArrayView<float32>, ArrayView<float32>, ArrayView<uint32>, int, int, int>(MatVecKernels.matVecWordsShared)))
    member val MatVecQ4KPairs =
        lazy (accelerator.LoadStreamKernel(Action<ArrayView<float32>, ArrayView<float32>, ArrayView<uint32>, int, int>(MatVecKernels.matVecQ4KPairs)))
    member val MatVecQ6KPairs =
        lazy (accelerator.LoadStreamKernel(Action<ArrayView<float32>, ArrayView<float32>, ArrayView<uint32>, int, int>(MatVecKernels.matVecQ6KPairs)))
    member val MatVecWordsGlobal =
        lazy (accelerator.LoadStreamKernel(Action<ArrayView<float32>, ArrayView<float32>, ArrayView<uint32>, int, int, int>(MatVecKernels.matVecWordsGlobal)))
    member val RmsNorm =
        lazy (accelerator.LoadStreamKernel(Action<ArrayView<float32>, ArrayView<float32>, ArrayView<float32>, int, float32>(AttentionKernels.rmsNorm)))
    member val HeadRmsNorm =
        lazy (accelerator.LoadStreamKernel(Action<ArrayView<float32>, ArrayView<float32>, int, float32>(AttentionKernels.headRmsNorm)))
    member val Attention =
        lazy (accelerator.LoadStreamKernel(Action<ArrayView<float32>, ArrayView<float32>, ArrayView<float32>, ArrayView<float32>, ArrayView<float32>, AttentionArgs>(AttentionKernels.attention)))
    member val Rope =
        lazy (accelerator.LoadAutoGroupedStreamKernel(Action<Index1D, ArrayView<float32>, ArrayView<float32>, ArrayView<float32>, RopeArgs>(ElementwiseKernels.rope)))
    member val SwiGlu =
        lazy (accelerator.LoadAutoGroupedStreamKernel(Action<Index1D, ArrayView<float32>, ArrayView<float32>>(ElementwiseKernels.swiGlu)))
    member val Add =
        lazy (accelerator.LoadAutoGroupedStreamKernel(Action<Index1D, ArrayView<float32>, ArrayView<float32>>(ElementwiseKernels.add)))
    member val AddScaled =
        lazy (accelerator.LoadAutoGroupedStreamKernel(Action<Index1D, ArrayView<float32>, ArrayView<float32>, float32>(ElementwiseKernels.addScaled)))

/// Host -> device weight upload through two page-locked staging buffers. The host side may be a
/// memory-mapped file of any size: rows are converted to the device layout chunk by chunk, and
/// converting chunk i+1 (which also faults its pages in from disk) overlaps the DMA of chunk i.
[<Sealed>]
type internal StagedUploader(accelerator: Accelerator, stream: AcceleratorStream) =
    /// Transfer granularity. Large enough to amortise a DMA launch, small enough to keep the
    /// pinned footprint modest; a single row larger than this gets a staging buffer of its own size.
    [<Literal>]
    let ChunkBytes = 32L * 1024L * 1024L
    let mutable staging = Array.empty<PageLockedArray1D<byte>>

    let ensure (bytes: int64) =
        if staging.Length = 0 || staging.[0].Length < bytes then
            for b in staging do b.Dispose()
            staging <- Array.empty
            let first = accelerator.AllocatePageLocked1D<byte>(bytes)
            try staging <- [| first; accelerator.AllocatePageLocked1D<byte>(bytes) |]
            with _ ->
                first.Dispose()
                reraise ()

    member _.Upload(t: HostTensor, target: ArrayView<byte>) =
        let rowBytes = GpuLayout.deviceRowBytes t
        if target.Length < rowBytes * int64 t.Rows then
            invalidArg (nameof target) $"Device buffer too small for '%s{t.Name}'"
        let rowsPerChunk = int (max 1L (min (int64 t.Rows) (ChunkBytes / rowBytes)))
        ensure (rowBytes * int64 rowsPerChunk)
        let mutable row = 0
        let mutable slot = 0
        while row < t.Rows do
            let count = min rowsPerChunk (t.Rows - row)
            let bytes = rowBytes * int64 count
            let buffer = staging.[slot]
            // The copy that last used this slot was awaited before the previous chunk was queued.
            GpuLayout.writeRows t row count (buffer.Span.Slice(0, int bytes))
            stream.Synchronize() // at most one DMA in flight: the other slot is free again
            let source = buffer.ArrayView.SubView(0L, bytes)
            target.SubView(rowBytes * int64 row, bytes).CopyFrom(stream, &source)
            row <- row + count
            slot <- 1 - slot
        stream.Synchronize()

    /// Staging memory is only needed while loading; release it once the model is resident.
    member _.Release() =
        for b in staging do b.Dispose()
        staging <- Array.empty

    interface IDisposable with
        member this.Dispose() = this.Release()

/// IComputeBackend over one CUDA device. Every operation is enqueued on the accelerator's default
/// stream; downloads, weight uploads and Synchronize are the only points where the host waits.
/// Kernels compile on first use, and the shared-memory limit and grid size come from the device.
/// The backend owns the accelerator: it is released if construction fails, and on Dispose.
[<Sealed>]
type CudaBackend(accelerator: CudaAccelerator, info: CudaDeviceInfo) =
    let kernels = KernelSet(accelerator)

    /// CUDA contexts are per thread: with several accelerators (multi-GPU, or several contexts on
    /// one GPU) the right one must be current before any driver call. Cheap when already bound.
    let bind () =
        if not (obj.ReferenceEquals(Accelerator.Current, accelerator)) then accelerator.Bind()
    let stream = accelerator.DefaultStream
    /// Placeholder for kernels with optional inputs (RoPE factors).
    let dummy =
        try
            bind ()
            accelerator.Allocate1D<float32>(1L)
        with _ ->
            accelerator.Dispose()
            reraise ()
    let uploader = new StagedUploader(accelerator, stream)
    let mutable disposed = false
    /// Longest row (floats) whose activations fit in a group's shared memory.
    let sharedColumns = max 0 (accelerator.MaxSharedMemoryPerGroup / sizeof<float32>)

    static let length (b: IBuffer) = b.Length

    static let require (condition: bool) (message: unit -> string) =
        if not condition then raise (ArgumentException(message ()))

    static let view (b: IBuffer) : ArrayView<float32> =
        match b with
        | :? CudaBuffer as c -> c.View
        | other -> invalidArg (nameof b) $"Expected a CUDA buffer, got %s{other.GetType().Name}"

    static let weight (w: IWeight) =
        match w with
        | :? CudaWeight as c -> c
        | other -> invalidArg (nameof w) $"Expected a CUDA weight, got %s{other.GetType().Name}"

    static let cache (c: IKvCache) =
        match c with
        | :? CudaKvCache as k -> k
        | other -> invalidArg (nameof c) $"Expected a CUDA KV cache, got %s{other.GetType().Name}"

    /// Groups for grid-stride matvec: enough to fill every SM a few times over.
    let strideGroups = max 1 (info.Multiprocessors * 8)

    let matVec (output: IBuffer) (x: IBuffer) (w: IWeight) =
        let cw = weight w
        let rows = w.Rows
        let cols = w.Columns
        require (length x >= cols) (fun () -> $"'%s{w.Name}' needs %d{cols} inputs, buffer has %d{length x}")
        require (length output >= rows) (fun () -> $"'%s{w.Name}' writes %d{rows} outputs, buffer has %d{length output}")
        let o = view output
        let xv = view x
        let rowGroups = Numerics.divUp rows MatVecKernels.MatVecRowsPerGroup
        let words format =
            let groups = min rowGroups strideGroups
            if cols <= sharedColumns then
                let config = KernelConfig(Index1D(groups), Index1D(MatVecKernels.MatVecGroupSize), SharedMemoryConfig.RequestDynamic<float32>(cols))
                kernels.MatVecWordsShared.Value.Invoke(config, o, xv, cw.Words, rows, cols, format)
            else
                let config = KernelConfig(Index1D(groups), Index1D(MatVecKernels.MatVecGroupSize))
                kernels.MatVecWordsGlobal.Value.Invoke(config, o, xv, cw.Words, rows, cols, format)
        /// Two rows per warp: half as many warps, same grid-stride scheme.
        let pairs (kernel: Action<KernelConfig, ArrayView<float32>, ArrayView<float32>, ArrayView<uint32>, int, int>) =
            let pairGroups = Numerics.divUp (Numerics.divUp rows 2) MatVecKernels.MatVecRowsPerGroup
            let config = KernelConfig(Index1D(min pairGroups strideGroups), Index1D(MatVecKernels.MatVecGroupSize), SharedMemoryConfig.RequestDynamic<float32>(cols))
            kernel.Invoke(config, o, xv, cw.Words, rows, cols)
        match w.Type with
        | GgmlType.F32 ->
            let config = KernelConfig(Index1D(rowGroups), Index1D(MatVecKernels.MatVecGroupSize))
            kernels.MatVecF32.Value.Invoke(config, o, xv, cw.Singles, rows, cols)
        | GgmlType.F16 -> words MatVecKernels.FormatF16
        | GgmlType.BF16 -> words MatVecKernels.FormatBF16
        | GgmlType.Q8_0 -> words MatVecKernels.FormatQ8_0
        | GgmlType.Q4_K when cols <= sharedColumns -> pairs kernels.MatVecQ4KPairs.Value
        | GgmlType.Q6_K when cols <= sharedColumns -> pairs kernels.MatVecQ6KPairs.Value
        | GgmlType.Q4_K -> words MatVecKernels.FormatQ4_K
        | GgmlType.Q6_K -> words MatVecKernels.FormatQ6_K
        | other -> raise (NotSupportedException $"CUDA matvec does not support %A{other} ('%s{w.Name}')")

    member _.Info = info

    /// Frees the staging buffers used by LoadWeight (call once every resident weight is loaded).
    member _.ReleaseStaging() =
        bind ()
        uploader.Release()

    interface IWeightStreaming with
        member _.StagingBytes(layers) = CudaStreamedLayerSource.RequiredDeviceBytes layers
        member _.CreateStreamedSource(layers) = new CudaStreamedLayerSource(accelerator, stream, layers) :> ILayerSource

    interface IComputeBackend with
        member _.Name = $"CUDA:%d{info.Index} %s{info.Name} (%s{info.Architecture})"
        member _.Kind = Cuda
        member _.DeviceIndex = info.Index
        member _.MemoryInfo =
            bind ()
            Some { Free = accelerator.GetFreeMemory(); Total = info.TotalMemory }

        member _.Allocate(length) =
            bind ()
            new CudaBuffer(accelerator.Allocate1D<float32>(int64 length)) :> IBuffer

        member _.Upload(source, destination) =
            bind ()
            require (source.Length = length destination) (fun () -> $"Upload of %d{source.Length} values into a buffer of %d{length destination}")
            (view destination).CopyFromCPU(stream, source)

        member _.Download(source, destination) =
            bind ()
            require (destination.Length = length source) (fun () -> $"Download of a buffer of %d{length source} values into %d{destination.Length}")
            (view source).CopyToCPU(stream, destination)
            stream.Synchronize()

        member _.LoadWeight(t) =
            bind ()
            match t.Type with
            | GgmlType.F32 | GgmlType.F16 | GgmlType.BF16 | GgmlType.Q8_0 | GgmlType.Q4_K | GgmlType.Q6_K -> ()
            | other -> raise (NotSupportedException $"CUDA cannot hold '%s{t.Name}' in %A{other}; convert it on load")
            if (t.Type = GgmlType.F16 || t.Type = GgmlType.BF16) && t.Columns % 2 <> 0 then
                raise (NotSupportedException $"'%s{t.Name}': half-precision rows need an even column count")
            let buffer = accelerator.Allocate1D<byte>(GpuLayout.deviceBytes t)
            try
                uploader.Upload(t, buffer.View.AsContiguous())
                new CudaWeight(t.Name, t.Type, t.Rows, t.Columns, buffer.View.AsContiguous(), Some(buffer :> IDisposable)) :> IWeight
            with _ ->
                buffer.Dispose()
                reraise ()

        member _.SliceRows(w, firstRow, rowCount) =
            bind ()
            (weight w).Slice(firstRow, rowCount) :> IWeight

        member _.CreateKvCache(capacity, kvDim) =
            bind ()
            new CudaKvCache(accelerator, capacity, kvDim) :> IKvCache

        member _.RmsNorm(output, x, w, eps) =
            bind ()
            let size = int (view x).Length
            kernels.RmsNorm.Value.Invoke(KernelConfig(Index1D(1), Index1D(AttentionKernels.ReductionGroupSize)), view output, view x, (weight w).Singles, size, eps)

        member _.HeadRmsNorm(x, w, numHeads, headDim, eps) =
            bind ()
            kernels.HeadRmsNorm.Value.Invoke(KernelConfig(Index1D(numHeads), Index1D(32)), view x, (weight w).Singles, headDim, eps)

        member _.MatVec(output, x, w) =
            bind ()
            matVec output x w

        member _.MatVecBatch(inputs, outputs, weights) =
            bind ()
            if inputs.Length <> weights.Length || outputs.Length <> weights.Length then
                invalidArg (nameof weights) "inputs, outputs and weights must have the same length"
            for i in 0 .. weights.Length - 1 do matVec outputs.[i] inputs.[i] weights.[i]

        member _.Add(target, other) =
            bind ()
            require (length other >= length target) (fun () -> "Add: operand shorter than target")
            let t = view target
            kernels.Add.Value.Invoke(Index1D(int t.Length), t, view other)

        member _.AddScaled(target, other, scale) =
            bind ()
            require (length other >= length target) (fun () -> "AddScaled: operand shorter than target")
            let t = view target
            kernels.AddScaled.Value.Invoke(Index1D(int t.Length), t, view other, scale)

        member _.AddBias(target, bias) =
            bind ()
            require (bias.Type = GgmlType.F32 && int64 bias.Rows * int64 bias.Columns >= int64 (length target)) (fun () -> $"Bias '%s{bias.Name}' must be F32 and cover the target")
            let t = view target
            kernels.Add.Value.Invoke(Index1D(int t.Length), t, (weight bias).Singles)

        member _.Rope(q, k, position, spec) =
            bind ()
            require (length q >= spec.NumHeads * spec.HeadDim && length k >= spec.NumKvHeads * spec.HeadDim) (fun () -> "Rope: q/k buffers too small")
            let factors, has =
                match spec.FreqFactors with
                | Some f -> (weight f).Singles, 1
                | None -> dummy.View.AsContiguous(), 0
            let args =
                RopeArgs(position, spec.NumHeads, spec.NumKvHeads, spec.HeadDim, spec.Theta, (if spec.Mode = NeoX then 1 else 0), has)
            let pairs = (spec.NumHeads + spec.NumKvHeads) * (spec.HeadDim / 2)
            kernels.Rope.Value.Invoke(Index1D(pairs), view q, view k, factors, args)

        member _.StoreKv(c, k, v, position) =
            bind ()
            let kc = cache c
            require (position >= 0 && position < c.Capacity) (fun () -> $"KV position %d{position} outside capacity %d{c.Capacity}")
            require (length k >= c.KvDim && length v >= c.KvDim) (fun () -> "StoreKv: k/v buffers too small")
            let kvDim = int64 c.KvDim
            let offset = int64 position * kvDim
            let keyTarget = kc.Keys.SubView(offset, kvDim)
            let valueTarget = kc.Values.SubView(offset, kvDim)
            (view k).SubView(0L, kvDim).CopyTo(stream, &keyTarget)
            (view v).SubView(0L, kvDim).CopyTo(stream, &valueTarget)

        member _.Attention(output, q, c, position, spec, scratch) =
            bind ()
            let kc = cache c
            require (position >= 0 && position < c.Capacity) (fun () -> $"Attention position %d{position} outside capacity %d{c.Capacity}")
            require (length scratch >= spec.NumHeads * c.Capacity) (fun () -> "Attention: score buffer smaller than NumHeads x capacity")
            require (length q >= spec.NumHeads * spec.HeadDim && length output >= spec.NumHeads * spec.HeadDim) (fun () -> "Attention: q/output buffers too small")
            let args = AttentionArgs(position, c.Capacity, spec.KvDim, spec.HeadDim, spec.KvMul, MathF.Sqrt(float32 spec.HeadDim))
            kernels.Attention.Value.Invoke(
                KernelConfig(Index1D(spec.NumHeads), Index1D(AttentionKernels.ReductionGroupSize)),
                view output, view q, kc.Keys, kc.Values, view scratch, args)

        member _.SwiGlu(gate, up) =
            bind ()
            require (length up >= length gate) (fun () -> "SwiGlu: up shorter than gate")
            let g = view gate
            kernels.SwiGlu.Value.Invoke(Index1D(int g.Length), g, view up)

        member _.Synchronize() =
            bind ()
            stream.Synchronize()

        member _.Dispose() =
            if not disposed then
                disposed <- true
                try
                    bind ()
                    stream.Synchronize()
                finally
                    try
                        (uploader :> IDisposable).Dispose()
                        dummy.Dispose()
                    finally
                        accelerator.Dispose()
