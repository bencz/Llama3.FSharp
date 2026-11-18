namespace Llama3.Backend.Cpu

open System
open System.Collections.Generic
open System.IO
open System.Threading
open Llama3.Core

type CpuOptions =
    { /// Worker threads for matrix-vector products (0 = physical cores, see CpuTopology).
      Threads: int
      /// Quantize activations to int8 for Q8_0/Q4_K/Q6_K weights (llama.cpp-style integer dot
      /// products). Disable for bit-closer parity with float reference implementations.
      QuantizedActivations: bool
      /// Kernel implementation; None = measured per format on this machine (see <see cref="Autotune"/>).
      Isa: CpuIsa option }

    static member Default = { Threads = 0; QuantizedActivations = true; Isa = None }

[<Sealed>]
type CpuBuffer(data: float32[]) =
    member _.Data = data
    interface IBuffer with
        member _.Length = data.Length
        member _.Dispose() = ()

/// Weights stay in the host tensor (zero copy). A weight may cover only a row range of its tensor
/// (an expert of a stacked MoE tensor).
[<Sealed>]
type CpuWeight(tensor: HostTensor, firstRow: int, rowCount: int) =
    do
        if firstRow < 0 || rowCount < 0 || firstRow + rowCount > tensor.Rows then
            invalidArg (nameof rowCount) $"Rows [%d{firstRow}, %d{firstRow + rowCount}) outside '%s{tensor.Name}' (%d{tensor.Rows} rows)"

    new(tensor: HostTensor) = new CpuWeight(tensor, 0, tensor.Rows)

    member _.Tensor = tensor
    member _.FirstRow = firstRow
    member _.RowCount = rowCount

    interface IWeight with
        member _.Name = tensor.Name
        member _.Type = tensor.Type
        member _.Rows = rowCount
        member _.Columns = tensor.Columns
        member _.SizeInBytes = int64 rowCount * int64 tensor.RowBytes
        member _.Dispose() = ()

/// Contiguous [capacity x kvDim] key and value planes.
[<Sealed>]
type CpuKvCache(capacity: int, kvDim: int) =
    let keys = Array.zeroCreate<float32> (capacity * kvDim)
    let values = Array.zeroCreate<float32> (capacity * kvDim)
    member _.Keys = keys
    member _.Values = values
    interface IKvCache with
        member _.Capacity = capacity
        member _.KvDim = kvDim
        member _.SizeInBytes = int64 keys.Length * 8L
        member _.Dispose() = ()

/// Per-thread decode scratch for the float reference path.
type internal Scratch private () =
    [<ThreadStatic; DefaultValue>]
    static val mutable private buffer: float32[]

    static member Get() =
        if isNull Scratch.buffer then Scratch.buffer <- Array.zeroCreate 256
        Scratch.buffer

/// A unit of matvec work: rows [Start, End) of matrix Matrix.
[<Struct>]
type internal RowChunk = { Matrix: int; Start: int; End: int }

/// Host memory as seen by the kernel (MemAvailable accounts for reclaimable page cache).
module internal HostMemory =
    let read () : MemoryInfo option =
        if not (OperatingSystem.IsLinux()) then None
        else
            try
                let values =
                    File.ReadLines "/proc/meminfo"
                    |> Seq.choose (fun line ->
                        let parts = line.Split([| ':'; ' ' |], StringSplitOptions.RemoveEmptyEntries)
                        if parts.Length >= 2 then Some(parts.[0], int64 parts.[1] * 1024L) else None)
                    |> Map.ofSeq
                match values.TryFind "MemAvailable", values.TryFind "MemTotal" with
                | Some free, Some total -> Some { Free = free; Total = total }
                | _ -> None
            with :? IOException | :? FormatException -> None

/// IComputeBackend on the host CPU. Not reentrant: one forward pass at a time (concurrent calls
/// are detected and rejected). Parallel kernels run on a persistent <see cref="CpuWorkerPool"/>.
[<Sealed>]
type CpuBackend(options: CpuOptions) =
    let threads = if options.Threads > 0 then options.Threads else CpuTopology.physicalCores.Value
    let pool = new CpuWorkerPool(threads)
    let kernels =
        match options.Isa with
        | Some isa -> CpuKernels.forIsa isa
        | None -> Autotune.best threads
    /// Below this many multiply-adds a job runs on the calling thread: the fork-join cost grows
    /// with the worker count, so the threshold scales with it.
    let parallelThreshold = 32_768L * int64 threads

    /// One quantized-activation slot per distinct input of a batch (grown on demand, reused).
    let quantPool = List<QuantizedActivations>()
    let chunks = List<RowChunk>()
    let mutable batchInputs = Array.empty<float32[]>
    let mutable batchOutputs = Array.empty<float32[]>
    let mutable batchWeights = Array.empty<CpuWeight>
    let mutable batchQuant = Array.empty<QuantizedActivations voption>
    let mutable busy = 0

    /// Inverse RoPE frequencies per (theta, head_dim, factors), computed once.
    let ropeTables = Dictionary<struct (float32 * int * IWeight voption), float32[]>()
    let mutable ropeCos = Array.empty<float32>
    let mutable ropeSin = Array.empty<float32>

    // Attention job state (set per call, read by the pool's chunks = heads).
    let mutable attOutput = Array.empty<float32>
    let mutable attQ = Array.empty<float32>
    let mutable attKeys = Array.empty<float32>
    let mutable attValues = Array.empty<float32>
    let mutable attScores = Array.empty<float32>
    let mutable attPosition = 0
    let mutable attCapacity = 0
    let mutable attSpec = { NumHeads = 1; NumKvHeads = 1; HeadDim = 2 }

    static let data (b: IBuffer) =
        match b with
        | :? CpuBuffer as c -> c.Data
        | other -> invalidArg (nameof b) $"Expected a CPU buffer, got %s{other.GetType().Name}"

    static let cpuWeight (w: IWeight) =
        match w with
        | :? CpuWeight as c -> c
        | other -> invalidArg (nameof w) $"Expected a CPU weight, got %s{other.GetType().Name}"

    /// Whole-tensor F32 view (norm weights, biases, RoPE factors).
    static let singles (w: IWeight) = (cpuWeight w).Tensor.AsSingles()

    static let kv (c: IKvCache) =
        match c with
        | :? CpuKvCache as k -> k
        | other -> invalidArg (nameof c) $"Expected a CPU KV cache, got %s{other.GetType().Name}"

    let exclusive (operation: unit -> unit) =
        if Interlocked.Exchange(&busy, 1) = 1 then
            invalidOp "CpuBackend is not reentrant: two forward passes are using it concurrently"
        try operation ()
        finally Volatile.Write(&busy, 0)

    let ensureBatchCapacity n =
        if batchInputs.Length < n then
            batchInputs <- Array.zeroCreate n
            batchOutputs <- Array.zeroCreate n
            batchWeights <- Array.zeroCreate n
            batchQuant <- Array.zeroCreate n

    let runChunk (c: RowChunk) =
        let w = batchWeights.[c.Matrix]
        let x = batchInputs.[c.Matrix]
        let xq = batchQuant.[c.Matrix]
        let o = batchOutputs.[c.Matrix]
        let scratch = Span<float32>(Scratch.Get())
        for r in c.Start .. c.End - 1 do
            o.[r] <- CpuKernels.dotRow kernels w.Tensor (w.FirstRow + r) x xq scratch

    let matVecJob = { new IChunkJob with member _.Execute i = runChunk chunks.[i] }

    /// outputs[i] = weights[i] * inputs[i] over batchInputs/Outputs/Weights[0..n-1].
    /// Every distinct input (by reference) consumed by an integer kernel is quantized exactly once,
    /// then all rows of all products are scheduled as one balanced job.
    let matVecBatch (n: int) =
        let mutable totalWork = 0L
        for i in 0 .. n - 1 do
            let w = batchWeights.[i]
            let cols = w.Tensor.Columns
            if batchInputs.[i].Length < cols then
                invalidArg "inputs" $"Weight '%s{w.Tensor.Name}' expects %d{cols} inputs, got %d{batchInputs.[i].Length}"
            if batchOutputs.[i].Length < w.RowCount then
                invalidArg "outputs" $"Output for '%s{w.Tensor.Name}' is too small (%d{batchOutputs.[i].Length} < %d{w.RowCount})"
            totalWork <- totalWork + int64 w.RowCount * int64 cols
        let mutable slots = 0
        for i in 0 .. n - 1 do
            batchQuant.[i] <- ValueNone
            let x = batchInputs.[i]
            let dtype = batchWeights.[i].Tensor.Type
            if options.QuantizedActivations && CpuKernels.usesQuantizedActivations dtype then
                if x.Length % 32 <> 0 then
                    invalidOp $"Activations of length %d{x.Length} cannot feed the integer kernels (multiple of 32 required)"
                let mutable j = 0
                while j < i && batchQuant.[i].IsNone do
                    if obj.ReferenceEquals(batchInputs.[j], x) && batchQuant.[j].IsSome then batchQuant.[i] <- batchQuant.[j]
                    j <- j + 1
                if batchQuant.[i].IsNone then
                    if quantPool.Count <= slots then quantPool.Add(QuantizedActivations(max x.Length 4096))
                    elif quantPool.[slots].Quants.Length < x.Length then quantPool.[slots] <- QuantizedActivations(x.Length)
                    let q = quantPool.[slots]
                    q.Quantize(ReadOnlySpan x)
                    batchQuant.[i] <- ValueSome q
                    slots <- slots + 1
        chunks.Clear()
        if totalWork < parallelThreshold || threads = 1 then
            for m in 0 .. n - 1 do
                runChunk { Matrix = m; Start = 0; End = batchWeights.[m].RowCount }
        else
            // Roughly 4 chunks per thread over the whole job, never below 8 rows per chunk.
            let targetChunks = int64 threads * 4L
            for m in 0 .. n - 1 do
                let w = batchWeights.[m]
                let rows = w.RowCount
                let share = max 1 (int (targetChunks * int64 rows * int64 w.Tensor.Columns / totalWork))
                let size = max 8 (Numerics.divUp rows share)
                let mutable s = 0
                while s < rows do
                    chunks.Add { Matrix = m; Start = s; End = min rows (s + size) }
                    s <- s + size
            pool.Run(matVecJob, chunks.Count)
        // Drop references to the caller's buffers and weights once the job is done.
        Array.Clear(batchInputs, 0, n)
        Array.Clear(batchOutputs, 0, n)
        Array.Clear(batchWeights, 0, n)
        Array.Clear(batchQuant, 0, n)

    let attentionHead (h: int) =
        let spec = attSpec
        let headDim = spec.HeadDim
        let kvDim = spec.KvDim
        let scale = MathF.Sqrt(float32 headDim)
        let qh = ReadOnlySpan<float32>(attQ, h * headDim, headDim)
        let kvOffset = (h / spec.KvMul) * headDim
        let scores = Span<float32>(attScores, h * attCapacity, attPosition + 1)
        for t in 0 .. attPosition do
            scores.[t] <- kernels.DotF32(qh, ReadOnlySpan<float32>(attKeys, t * kvDim + kvOffset, headDim)) / scale
        CpuKernels.softmax scores
        let oh = Span<float32>(attOutput, h * headDim, headDim)
        oh.Clear()
        for t in 0 .. attPosition do
            kernels.Axpy(scores.[t], ReadOnlySpan<float32>(attValues, t * kvDim + kvOffset, headDim), oh)

    let attentionJob = { new IChunkJob with member _.Execute h = attentionHead h }

    let inverseFrequencies (spec: RopeSpec) =
        let key = struct (spec.Theta, spec.HeadDim, (match spec.FreqFactors with Some f -> ValueSome f | None -> ValueNone))
        match ropeTables.TryGetValue key with
        | true, table -> table
        | _ ->
            let half = spec.HeadDim / 2
            let factors = spec.FreqFactors |> Option.map (fun f -> (singles f).ToArray())
            let table =
                Array.init half (fun j ->
                    let freq = 1.0f / MathF.Pow(spec.Theta, float32 (2 * j) / float32 spec.HeadDim)
                    match factors with
                    | Some f -> freq / f.[j]
                    | None -> freq)
            ropeTables.[key] <- table
            table

    member _.Threads = threads
    member _.Kernels = kernels
    member _.Options = options

    interface IComputeBackend with
        member _.Name = $"CPU (%d{threads} threads, %s{kernels.Description})"
        member _.Kind = Cpu
        member _.DeviceIndex = 0
        member _.MemoryInfo = HostMemory.read ()

        member _.Allocate(length) = new CpuBuffer(Array.zeroCreate length) :> IBuffer

        member _.Upload(source, destination) =
            let d = data destination
            if source.Length <> d.Length then invalidArg (nameof source) $"Upload of %d{source.Length} values into a buffer of %d{d.Length}"
            Array.Copy(source, d, d.Length)

        member _.Download(source, destination) =
            let s = data source
            if destination.Length <> s.Length then invalidArg (nameof destination) $"Download of a buffer of %d{s.Length} values into %d{destination.Length}"
            Array.Copy(s, destination, s.Length)

        member _.LoadWeight(t) = new CpuWeight(t) :> IWeight

        member _.SliceRows(weight, firstRow, rowCount) =
            let w = cpuWeight weight
            if firstRow + rowCount > w.RowCount then
                invalidArg (nameof rowCount) $"Slice [%d{firstRow}, %d{firstRow + rowCount}) exceeds %d{w.RowCount} rows"
            new CpuWeight(w.Tensor, w.FirstRow + firstRow, rowCount) :> IWeight

        member _.CreateKvCache(capacity, kvDim) = new CpuKvCache(capacity, kvDim) :> IKvCache

        member _.RmsNorm(output, x, weight, eps) =
            let xd = data x
            CpuKernels.rmsNorm kernels (Span(data output, 0, xd.Length)) (ReadOnlySpan xd) (singles weight) eps

        member _.HeadRmsNorm(x, weight, numHeads, headDim, eps) =
            let xd = data x
            let w = singles weight
            for h in 0 .. numHeads - 1 do
                let span = Span<float32>(xd, h * headDim, headDim)
                CpuKernels.rmsNorm kernels span (Span<float32>.op_Implicit span) w eps

        member _.MatVec(output, x, weight) =
            exclusive (fun () ->
                ensureBatchCapacity 1
                batchInputs.[0] <- data x
                batchOutputs.[0] <- data output
                batchWeights.[0] <- cpuWeight weight
                matVecBatch 1)

        member _.MatVecBatch(inputs, outputs, weights) =
            let n = weights.Length
            if inputs.Length <> n || outputs.Length <> n then invalidArg (nameof weights) "inputs, outputs and weights must have the same length"
            exclusive (fun () ->
                ensureBatchCapacity n
                for i in 0 .. n - 1 do
                    batchInputs.[i] <- data inputs.[i]
                    batchOutputs.[i] <- data outputs.[i]
                    batchWeights.[i] <- cpuWeight weights.[i]
                matVecBatch n)

        member _.Add(target, other) = CpuKernels.add (Span(data target)) (ReadOnlySpan(data other))

        member _.AddBias(target, bias) = CpuKernels.add (Span(data target)) (singles bias)

        member _.AddScaled(target, other, scale) = kernels.Axpy(scale, ReadOnlySpan(data other), Span(data target))

        member _.Rope(q, k, position, spec) =
            let half = spec.HeadDim / 2
            if ropeCos.Length < half then
                ropeCos <- Array.zeroCreate half
                ropeSin <- Array.zeroCreate half
            let inverse = inverseFrequencies spec
            for j in 0 .. half - 1 do
                let angle = float32 position * inverse.[j]
                ropeCos.[j] <- MathF.Cos angle
                ropeSin.[j] <- MathF.Sin angle
            let cos = ReadOnlySpan<float32>(ropeCos, 0, half)
            let sin = ReadOnlySpan<float32>(ropeSin, 0, half)
            CpuKernels.applyRope (Span(data q, 0, spec.NumHeads * spec.HeadDim)) spec.NumHeads spec.HeadDim spec.Mode cos sin
            CpuKernels.applyRope (Span(data k, 0, spec.NumKvHeads * spec.HeadDim)) spec.NumKvHeads spec.HeadDim spec.Mode cos sin

        member _.StoreKv(cache, k, v, position) =
            let c = kv cache
            let kvDim = (c :> IKvCache).KvDim
            Array.Copy(data k, 0, c.Keys, position * kvDim, kvDim)
            Array.Copy(data v, 0, c.Values, position * kvDim, kvDim)

        member _.Attention(output, q, cache, position, spec, scratch) =
            let c = kv cache
            exclusive (fun () ->
                attOutput <- data output
                attQ <- data q
                attKeys <- c.Keys
                attValues <- c.Values
                attScores <- data scratch
                attPosition <- position
                attCapacity <- (c :> IKvCache).Capacity
                attSpec <- spec
                if int64 spec.NumHeads * int64 (position + 1) * int64 spec.HeadDim < parallelThreshold then
                    for h in 0 .. spec.NumHeads - 1 do attentionHead h
                else pool.Run(attentionJob, spec.NumHeads))

        member _.SwiGlu(gate, up) = CpuKernels.swiGlu (Span(data gate)) (ReadOnlySpan(data up))
        member _.Synchronize() = ()
        member _.Dispose() = (pool :> IDisposable).Dispose()
