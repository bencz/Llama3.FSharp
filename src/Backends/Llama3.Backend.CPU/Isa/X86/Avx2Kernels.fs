namespace Llama3.Backend.Cpu

open System
open System.Buffers.Binary
open System.Runtime.CompilerServices
open System.Runtime.InteropServices
open System.Runtime.Intrinsics
open System.Runtime.Intrinsics.X86
open Llama3.Core

/// x86-64 AVX2 + FMA kernels. Integer dot products follow llama.cpp's scheme:
///   VPMADDUBSW (u8 × s8 → s16 pairs) → VPMADDWD (× 1 or × scale → s32) → CVTDQ2PS → VFMADD.
/// Per-block corrections (Q4_K mins, Q6_K −32 offset) are vector dot products of unpacked scales
/// with the pre-scaled activation sums of <see cref="QuantizedActivations"/>.
/// Every hot method is AggressiveOptimization: SIMD code must not start in the unoptimized tier.
/// Only instantiate when <c>CpuFeatures.isSupported X86Avx2</c>.
[<Sealed>]
type Avx2Kernels() =
    static let ones16 = Vector256.Create(1s)
    static let mask0F = Vector256.Create(0x0Fuy)
    static let mask30 = Vector256.Create(0x30uy)

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    static let load (data: ReadOnlySpan<byte>) (offset: int) = Vector256.LoadUnsafe(&MemoryMarshal.GetReference data, unativeint offset)

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    static let loadS (data: sbyte[]) (offset: int) = Vector256.LoadUnsafe(&data.[offset])

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    static let loadF (data: float32[]) (offset: int) = Vector256.LoadUnsafe(&data.[offset])

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    static let half (row: ReadOnlySpan<byte>) (o: int) = Numerics.halfToSingle (uint16 row.[o] ||| (uint16 row.[o + 1] <<< 8))

    /// 32 six-bit values · 32 activation quants, weighted by two int8 scales (one per 16 lanes).
    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    static let q6Accumulate (acc: Vector256<float32>) (q: Vector256<byte>) (sa: int16) (sb: int16) (xv: Vector256<sbyte>) (factor: float32) =
        let scaleVec = Vector256.Create(Vector128.Create sa, Vector128.Create sb)
        let p32 = Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(q, xv), scaleVec)
        Fma.MultiplyAdd(Avx.ConvertToVector256Single p32, Vector256.Create factor, acc)

    [<MethodImpl(MethodImplOptions.AggressiveOptimization)>]
    static member Dot(a: ReadOnlySpan<float32>, b: ReadOnlySpan<float32>) : float32 =
        let n = a.Length
        let mutable i = 0
        let mutable result = 0.0f
        if Avx512F.IsSupported && n >= 64 then
            let mutable acc0 = Vector512<float32>.Zero
            let mutable acc1 = Vector512<float32>.Zero
            let mutable acc2 = Vector512<float32>.Zero
            let mutable acc3 = Vector512<float32>.Zero
            while i + 64 <= n do
                acc0 <- Avx512F.FusedMultiplyAdd(Vector512.Create<float32>(a.Slice(i, 16)), Vector512.Create<float32>(b.Slice(i, 16)), acc0)
                acc1 <- Avx512F.FusedMultiplyAdd(Vector512.Create<float32>(a.Slice(i + 16, 16)), Vector512.Create<float32>(b.Slice(i + 16, 16)), acc1)
                acc2 <- Avx512F.FusedMultiplyAdd(Vector512.Create<float32>(a.Slice(i + 32, 16)), Vector512.Create<float32>(b.Slice(i + 32, 16)), acc2)
                acc3 <- Avx512F.FusedMultiplyAdd(Vector512.Create<float32>(a.Slice(i + 48, 16)), Vector512.Create<float32>(b.Slice(i + 48, 16)), acc3)
                i <- i + 64
            result <- Vector512.Sum((acc0 + acc1) + (acc2 + acc3))
        else if n >= 32 then
            let mutable acc0 = Vector256<float32>.Zero
            let mutable acc1 = Vector256<float32>.Zero
            let mutable acc2 = Vector256<float32>.Zero
            let mutable acc3 = Vector256<float32>.Zero
            while i + 32 <= n do
                acc0 <- Fma.MultiplyAdd(Vector256.Create<float32>(a.Slice(i, 8)), Vector256.Create<float32>(b.Slice(i, 8)), acc0)
                acc1 <- Fma.MultiplyAdd(Vector256.Create<float32>(a.Slice(i + 8, 8)), Vector256.Create<float32>(b.Slice(i + 8, 8)), acc1)
                acc2 <- Fma.MultiplyAdd(Vector256.Create<float32>(a.Slice(i + 16, 8)), Vector256.Create<float32>(b.Slice(i + 16, 8)), acc2)
                acc3 <- Fma.MultiplyAdd(Vector256.Create<float32>(a.Slice(i + 24, 8)), Vector256.Create<float32>(b.Slice(i + 24, 8)), acc3)
                i <- i + 32
            result <- Vector256.Sum((acc0 + acc1) + (acc2 + acc3))
        let mutable acc = Vector256<float32>.Zero
        while i + 8 <= n do
            acc <- Fma.MultiplyAdd(Vector256.Create<float32>(a.Slice(i, 8)), Vector256.Create<float32>(b.Slice(i, 8)), acc)
            i <- i + 8
        result <- result + Vector256.Sum acc
        while i < n do
            result <- result + a.[i] * b.[i]
            i <- i + 1
        result

    [<MethodImpl(MethodImplOptions.AggressiveOptimization)>]
    static member Q8_0(row: ReadOnlySpan<byte>, blocks: int, xq: QuantizedActivations) =
        KernelGuards.check row blocks 34 32 xq
        let xs = xq.Quants
        let xd = xq.Scales
        let mutable acc = Vector256<float32>.Zero
        for b in 0 .. blocks - 1 do
            let o = b * 34
            let wq = (load row (o + 2)).AsSByte()
            // Signed × signed via |w| (unsigned) × (x · sign(w)).
            let prod16 = Avx2.MultiplyAddAdjacent(Avx2.Abs wq, Avx2.Sign(loadS xs (b * 32), wq))
            let prod32 = Avx2.MultiplyAddAdjacent(prod16, ones16)
            acc <- Fma.MultiplyAdd(Avx.ConvertToVector256Single prod32, Vector256.Create(half row o * xd.[b]), acc)
        Vector256.Sum acc

    /// Per 32-element sub-block j: d·sc_j·dx_j·Σ(q4·qx) − dmin·m_j·(dx_j·Σqx).
    [<MethodImpl(MethodImplOptions.AggressiveOptimization)>]
    static member Q4_K(row: ReadOnlySpan<byte>, superBlocks: int, xq: QuantizedActivations) =
        KernelGuards.check row superBlocks 144 256 xq
        let xs = xq.Quants
        let xd = xq.Scales
        let ss32 = xq.ScaledSums32
        let mutable acc = Vector256<float32>.Zero
        let mutable mins = Vector256<float32>.Zero
        for sb in 0 .. superBlocks - 1 do
            let o = sb * 144
            let d = half row o
            let dmin = half row (o + 2)
            let struct (scales, minBytes) = Q4KScales.unpack (row.Slice(o + 4, 12))
            let blk0 = sb * 8
            // Eight mins (zero-extended) · eight pre-scaled activation sums, one FMA.
            let m = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(Vector128.CreateScalar(minBytes).AsByte()))
            mins <- Fma.MultiplyAdd(m * Vector256.Create dmin, loadF ss32 blk0, mins)
            for p in 0 .. 3 do
                let v = load row (o + 16 + p * 32)
                let lo = Avx2.And(v, mask0F)
                let hi = Avx2.And(Avx2.ShiftRightLogical(v.AsUInt16(), 4uy).AsByte(), mask0F)
                let e = sb * 256 + p * 64
                let pLo = Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(lo, loadS xs e), ones16)
                let pHi = Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(hi, loadS xs (e + 32)), ones16)
                let fLo = d * float32 (Q4KScales.byteAt scales (2 * p)) * xd.[blk0 + 2 * p]
                let fHi = d * float32 (Q4KScales.byteAt scales (2 * p + 1)) * xd.[blk0 + 2 * p + 1]
                acc <- Fma.MultiplyAdd(Avx.ConvertToVector256Single pLo, Vector256.Create fLo, acc)
                acc <- Fma.MultiplyAdd(Avx.ConvertToVector256Single pHi, Vector256.Create fHi, acc)
        Vector256.Sum acc - Vector256.Sum mins

    /// q6 ∈ [0,63] used unsigned; the −32 bias becomes Σ_g 32·d·sc_g·(dx·Σx)_g, computed as two
    /// 8-lane FMAs per super-block (scale g ↔ 16-element group g of the super-block).
    /// Byte shifts do not exist in AVX2: 16-bit shifts + masks extract the 2-bit high planes.
    [<MethodImpl(MethodImplOptions.AggressiveOptimization)>]
    static member Q6_K(row: ReadOnlySpan<byte>, superBlocks: int, xq: QuantizedActivations) =
        KernelGuards.check row superBlocks 210 256 xq
        let xs = xq.Quants
        let xd = xq.Scales
        let ss16 = xq.ScaledSums16
        let mutable acc = Vector256<float32>.Zero
        let mutable correction = Vector256<float32>.Zero
        for sb in 0 .. superBlocks - 1 do
            let o = sb * 210
            let d = half row (o + 208)
            let scales = Vector128.LoadUnsafe(&MemoryMarshal.GetReference row, unativeint (o + 192)).AsSByte()
            let factor = Vector256.Create(32.0f * d)
            let g0 = sb * 16
            let sLo = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32 scales)
            let sHi = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(Sse2.ShiftRightLogical128BitLane(scales, 8uy)))
            correction <- Fma.MultiplyAdd(sLo * factor, loadF ss16 g0, correction)
            correction <- Fma.MultiplyAdd(sHi * factor, loadF ss16 (g0 + 8), correction)
            for n in 0 .. 1 do
                let qlOff = o + n * 64
                let scOff = o + 192 + n * 8
                let eBase = sb * 256 + n * 128
                let ql0 = load row qlOff
                let ql1 = load row (qlOff + 32)
                let qh = load row (o + 128 + n * 32)
                let q0 = Avx2.Or(Avx2.And(ql0, mask0F), Avx2.And(Avx2.ShiftLeftLogical(qh.AsUInt16(), 4uy).AsByte(), mask30))
                let q1 = Avx2.Or(Avx2.And(ql1, mask0F), Avx2.And(Avx2.ShiftLeftLogical(qh.AsUInt16(), 2uy).AsByte(), mask30))
                let q2 = Avx2.Or(Avx2.And(Avx2.ShiftRightLogical(ql0.AsUInt16(), 4uy).AsByte(), mask0F), Avx2.And(qh, mask30))
                let q3 = Avx2.Or(Avx2.And(Avx2.ShiftRightLogical(ql1.AsUInt16(), 4uy).AsByte(), mask0F), Avx2.And(Avx2.ShiftRightLogical(qh.AsUInt16(), 2uy).AsByte(), mask30))
                acc <- q6Accumulate acc q0 (int16 (sbyte row.[scOff])) (int16 (sbyte row.[scOff + 1])) (loadS xs eBase) (d * xd.[eBase / 32])
                acc <- q6Accumulate acc q1 (int16 (sbyte row.[scOff + 2])) (int16 (sbyte row.[scOff + 3])) (loadS xs (eBase + 32)) (d * xd.[eBase / 32 + 1])
                acc <- q6Accumulate acc q2 (int16 (sbyte row.[scOff + 4])) (int16 (sbyte row.[scOff + 5])) (loadS xs (eBase + 64)) (d * xd.[eBase / 32 + 2])
                acc <- q6Accumulate acc q3 (int16 (sbyte row.[scOff + 6])) (int16 (sbyte row.[scOff + 7])) (loadS xs (eBase + 96)) (d * xd.[eBase / 32 + 3])
        Vector256.Sum acc - Vector256.Sum correction

    interface ICpuKernels with
        member _.Isa = X86Avx2
        member _.Description = $"x86-64 %s{CpuFeatures.describe ()}"
        member _.DotF32(a, b) = Avx2Kernels.Dot(a, b)
        member _.Axpy(alpha, x, y) = PortableKernels.AxpyImpl(alpha, x, y)
        member _.DotQ8_0(row, blocks, xq) = Avx2Kernels.Q8_0(row, blocks, xq)
        member _.DotQ4_K(row, superBlocks, xq) = Avx2Kernels.Q4_K(row, superBlocks, xq)
        member _.DotQ6_K(row, superBlocks, xq) = Avx2Kernels.Q6_K(row, superBlocks, xq)
