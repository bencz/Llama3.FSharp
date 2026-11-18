namespace Llama3.Backend.Cpu

open System
open System.Numerics
open System.Runtime.CompilerServices
open System.Runtime.InteropServices
open Llama3.Core

/// ISA selection plus the element-wise operations shared by every implementation.
/// Element-wise kernels use <see cref="Vector{T}"/>, whose width the JIT picks for the running
/// CPU (128-bit NEON/SSE, 256-bit AVX2, 512-bit AVX-512), so they are portable without dispatch.
module CpuKernels =

    let private create (isa: CpuIsa) : ICpuKernels =
        match isa with
        | X86Avx512Vnni -> Avx512VnniKernels()
        | X86Avx2 -> Avx2Kernels()
        | ArmNeon -> NeonKernels()
        | Portable -> PortableKernels()

    /// Kernels for a specific ISA (fails if the CPU cannot run it).
    let forIsa (isa: CpuIsa) : ICpuKernels =
        if not (CpuFeatures.isSupported isa) then
            raise (PlatformNotSupportedException $"%A{isa} kernels are not supported on this CPU (%s{CpuFeatures.describe ()})")
        create isa

    /// Every implementation this machine can execute (tests compare them against each other).
    let available () : ICpuKernels list =
        [ Portable; X86Avx2; X86Avx512Vnni; ArmNeon ] |> List.filter CpuFeatures.isSupported |> List.map create

    /// Whether a weight format is served by an integer kernel (and thus wants quantized activations).
    let usesQuantizedActivations (dtype: GgmlType) =
        match dtype with
        | GgmlType.Q8_0 | GgmlType.Q4_K | GgmlType.Q6_K -> true
        | _ -> false

    /// Exact path: decode 256 elements at a time into scratch and dot with float activations.
    [<MethodImpl(MethodImplOptions.AggressiveOptimization)>]
    let dotDecoded (k: ICpuKernels) (dtype: GgmlType) (row: ReadOnlySpan<byte>) (x: ReadOnlySpan<float32>) (scratch: Span<float32>) =
        let bs = GgmlTypes.blockSize dtype
        let ts = GgmlTypes.typeSize dtype
        let chunk = max bs 256
        let mutable acc = 0.0f
        let mutable e = 0
        while e < x.Length do
            let n = min chunk (x.Length - e)
            Dequantize.decode dtype (row.Slice((e / bs) * ts, (n / bs) * ts)) (scratch.Slice(0, n))
            acc <- acc + k.DotF32(Span<float32>.op_Implicit (scratch.Slice(0, n)), x.Slice(e, n))
            e <- e + n
        acc

    /// bfloat16 · float32: a bf16 is the upper half of a float32, so widening is a 16-bit shift.
    [<MethodImpl(MethodImplOptions.AggressiveOptimization)>]
    let dotBF16 (row: ReadOnlySpan<byte>) (x: ReadOnlySpan<float32>) : float32 =
        let halves = MemoryMarshal.Cast<byte, uint16> row
        let n = x.Length
        let width = Vector<uint32>.Count
        let mutable acc = Vector<float32>.Zero
        let mutable i = 0
        if Vector.IsHardwareAccelerated then
            while i + 2 * width <= n do
                let mutable lo = Vector<uint32>.Zero
                let mutable hi = Vector<uint32>.Zero
                Vector.Widen(Vector<uint16>(halves.Slice(i, 2 * width)), &lo, &hi)
                acc <- Vector.FusedMultiplyAdd(Vector.AsVectorSingle(Vector.ShiftLeft(lo, 16)), Vector<float32>(x.Slice(i, width)), acc)
                acc <- Vector.FusedMultiplyAdd(Vector.AsVectorSingle(Vector.ShiftLeft(hi, 16)), Vector<float32>(x.Slice(i + width, width)), acc)
                i <- i + 2 * width
        let mutable result = Vector.Sum acc
        while i < n do
            result <- result + Numerics.bfloat16ToSingle halves.[i] * x.[i]
            i <- i + 1
        result

    /// One output row, dispatched on the storage format. <paramref name="xq"/> = ValueNone selects
    /// the exact float path for quantized formats.
    [<MethodImpl(MethodImplOptions.AggressiveOptimization)>]
    let dotRow (k: ICpuKernels) (weight: HostTensor) (row: int) (x: float32[]) (xq: QuantizedActivations voption) (scratch: Span<float32>) : float32 =
        let bytes = weight.RowSpan row
        let cols = weight.Columns
        match weight.Type with
        | GgmlType.F32 -> k.DotF32(MemoryMarshal.Cast<byte, float32> bytes, ReadOnlySpan<float32>(x, 0, cols))
        | GgmlType.BF16 -> dotBF16 bytes (ReadOnlySpan<float32>(x, 0, cols))
        | dtype ->
            match xq, dtype with
            | ValueSome q, GgmlType.Q8_0 -> k.DotQ8_0(bytes, cols / 32, q)
            | ValueSome q, GgmlType.Q4_K -> k.DotQ4_K(bytes, cols / 256, q)
            | ValueSome q, GgmlType.Q6_K -> k.DotQ6_K(bytes, cols / 256, q)
            | _ -> dotDecoded k dtype bytes (ReadOnlySpan<float32>(x, 0, cols)) scratch

    // ------------------------------------------------------------------------------------------
    // Element-wise / normalisation / positional kernels
    // ------------------------------------------------------------------------------------------

    /// o = w ⊙ (x / sqrt(mean(x²) + eps)) — same evaluation order as llama3.cu.
    [<MethodImpl(MethodImplOptions.AggressiveOptimization)>]
    let rmsNorm (k: ICpuKernels) (o: Span<float32>) (x: ReadOnlySpan<float32>) (w: ReadOnlySpan<float32>) (eps: float32) =
        let ss = k.DotF32(x, x) / float32 x.Length + eps
        let inv = 1.0f / MathF.Sqrt ss
        let n = x.Length
        let width = Vector<float32>.Count
        let mutable i = 0
        if Vector.IsHardwareAccelerated then
            let vinv = Vector<float32>(inv)
            while i + width <= n do
                (Vector<float32>(w.Slice(i, width)) * (vinv * Vector<float32>(x.Slice(i, width)))).CopyTo(o.Slice(i, width))
                i <- i + width
        while i < n do
            o.[i] <- w.[i] * (inv * x.[i])
            i <- i + 1

    /// target += other
    [<MethodImpl(MethodImplOptions.AggressiveOptimization)>]
    let add (target: Span<float32>) (other: ReadOnlySpan<float32>) =
        let width = Vector<float32>.Count
        let mutable i = 0
        if Vector.IsHardwareAccelerated then
            while i + width <= target.Length do
                let t = target.Slice(i, width)
                (Vector<float32>(Span<float32>.op_Implicit t) + Vector<float32>(other.Slice(i, width))).CopyTo t
                i <- i + width
        while i < target.Length do
            target.[i] <- target.[i] + other.[i]
            i <- i + 1

    /// gate = silu(gate) · up
    [<MethodImpl(MethodImplOptions.AggressiveOptimization)>]
    let swiGlu (gate: Span<float32>) (up: ReadOnlySpan<float32>) =
        for i in 0 .. gate.Length - 1 do
            let v = gate.[i]
            gate.[i] <- v * (1.0f / (1.0f + MathF.Exp(-v))) * up.[i]

    /// In-place numerically stable softmax (normalised by division, as in llama3.cu).
    [<MethodImpl(MethodImplOptions.AggressiveOptimization)>]
    let softmax (x: Span<float32>) =
        let mutable maxVal = x.[0]
        for i in 1 .. x.Length - 1 do
            if x.[i] > maxVal then maxVal <- x.[i]
        let mutable sum = 0.0f
        for i in 0 .. x.Length - 1 do
            let e = MathF.Exp(x.[i] - maxVal)
            x.[i] <- e
            sum <- sum + e
        for i in 0 .. x.Length - 1 do
            x.[i] <- x.[i] / sum

    /// Rotates every head of <paramref name="v"/> by precomputed (cos, sin) pairs.
    [<MethodImpl(MethodImplOptions.AggressiveOptimization)>]
    let applyRope (v: Span<float32>) (heads: int) (headDim: int) (mode: RopeMode) (cos: ReadOnlySpan<float32>) (sin: ReadOnlySpan<float32>) =
        let half = headDim / 2
        for h in 0 .. heads - 1 do
            let hv = v.Slice(h * headDim, headDim)
            for j in 0 .. half - 1 do
                let i0 = if mode = NeoX then j else 2 * j
                let i1 = if mode = NeoX then j + half else 2 * j + 1
                let a = hv.[i0]
                let b = hv.[i1]
                hv.[i0] <- a * cos.[j] - b * sin.[j]
                hv.[i1] <- a * sin.[j] + b * cos.[j]
