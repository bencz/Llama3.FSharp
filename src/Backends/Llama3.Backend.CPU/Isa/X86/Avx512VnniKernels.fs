namespace Llama3.Backend.Cpu

open System
open System.Buffers.Binary
open System.Runtime.CompilerServices
open System.Runtime.InteropServices
open System.Runtime.Intrinsics
open System.Runtime.Intrinsics.X86
open Llama3.Core

/// x86-64 AVX-512 (F + BW) with VNNI kernels: VPDPBUSD multiplies 64 unsigned x signed byte pairs
/// and accumulates groups of four into 16 int32 lanes in one instruction (replacing the
/// VPMADDUBSW + VPMADDWD pair of the AVX2 path, at twice the width).
///
///   Q4_K / Q6_K: quant values are unsigned (0..15, 0..63) -> first operand as-is.
///   Q8_0:        signed weights are biased by XOR 0x80 (w + 128 as u8); the extra 128*sum(x) is
///                removed with the per-block activation sums computed at quantization time.
///
/// Available on Ice Lake / Rocket Lake / Sapphire Rapids and newer Intel, and AMD Zen 4 / Zen 5.
/// Only instantiate when <c>CpuFeatures.isSupported X86Avx512Vnni</c>.
[<Sealed>]
type Avx512VnniKernels() =
    static let mask0F = Vector512.Create(0x0Fuy)
    static let mask30 = Vector512.Create(0x30uy)
    static let bias = Vector512.Create(0x80uy)
    /// pshufb masks replicating scale bytes 0..3 (resp. 4..7) four times each.
    static let replicateLow = Vector128.Create(0uy, 0uy, 0uy, 0uy, 1uy, 1uy, 1uy, 1uy, 2uy, 2uy, 2uy, 2uy, 3uy, 3uy, 3uy, 3uy)
    static let replicateHigh = Vector128.Create(4uy, 4uy, 4uy, 4uy, 5uy, 5uy, 5uy, 5uy, 6uy, 6uy, 6uy, 6uy, 7uy, 7uy, 7uy, 7uy)

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    static let load512 (data: ReadOnlySpan<byte>) (offset: int) = Vector512.LoadUnsafe(&MemoryMarshal.GetReference data, unativeint offset)

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    static let load256 (data: ReadOnlySpan<byte>) (offset: int) = Vector256.LoadUnsafe(&MemoryMarshal.GetReference data, unativeint offset)

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    static let load512s (data: sbyte[]) (offset: int) = Vector512.LoadUnsafe(&data.[offset])

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    static let load256s (data: sbyte[]) (offset: int) = Vector256.LoadUnsafe(&data.[offset])

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    static let half (row: ReadOnlySpan<byte>) (o: int) = Numerics.halfToSingle (uint16 row.[o] ||| (uint16 row.[o + 1] <<< 8))

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    static let dpbusd (u: Vector512<byte>) (s: Vector512<sbyte>) =
        AvxVnni.V512.MultiplyWideningAndAdd(Vector512<int>.Zero, u, s)

    /// Two scalars, each broadcast to one 256-bit half.
    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    static let halves (a: float32) (b: float32) = Vector512.Create(Vector256.Create a, Vector256.Create b)

    [<MethodImpl(MethodImplOptions.AggressiveOptimization)>]
    static member Q8_0(row: ReadOnlySpan<byte>, blocks: int, xq: QuantizedActivations) =
        KernelGuards.check row blocks 34 32 xq
        let xs = xq.Quants
        let xd = xq.Scales
        let ss32 = xq.ScaledSums32
        let mutable acc = Vector512<float32>.Zero
        let mutable correction = 0.0f
        let mutable b = 0
        while b + 1 < blocks do
            let o0 = b * 34
            let o1 = o0 + 34
            let biased = Vector512.Create(load256 row (o0 + 2), load256 row (o1 + 2)) ^^^ bias
            let isum = dpbusd biased (Vector512.Create(load256s xs (b * 32), load256s xs (b * 32 + 32)))
            let d0 = half row o0
            let d1 = half row o1
            acc <- Avx512F.FusedMultiplyAdd(Avx512F.ConvertToVector512Single isum, halves (d0 * xd.[b]) (d1 * xd.[b + 1]), acc)
            correction <- correction + 128.0f * (d0 * ss32.[b] + d1 * ss32.[b + 1])
            b <- b + 2
        let mutable tail = 0.0f
        if b < blocks then
            let o = b * 34
            let mutable isum = 0
            for i in 0 .. 31 do
                isum <- isum + int (sbyte row.[o + 2 + i]) * int xs.[b * 32 + i]
            tail <- float32 isum * half row o * xd.[b]
        Vector512.Sum acc - correction + tail

    /// One 64-byte load covers the quants of sub-block pairs p and p+1; low nibbles feed sub-blocks
    /// 2p / 2p+2, high nibbles 2p+1 / 2p+3 (activation halves gathered to match).
    [<MethodImpl(MethodImplOptions.AggressiveOptimization)>]
    static member Q4_K(row: ReadOnlySpan<byte>, superBlocks: int, xq: QuantizedActivations) =
        KernelGuards.check row superBlocks 144 256 xq
        let xs = xq.Quants
        let xd = xq.Scales
        let ss32 = xq.ScaledSums32
        let mutable acc = Vector512<float32>.Zero
        let mutable mins = Vector256<float32>.Zero
        for sb in 0 .. superBlocks - 1 do
            let o = sb * 144
            let d = half row o
            let dmin = half row (o + 2)
            let struct (scales, minBytes) = Q4KScales.unpack (row.Slice(o + 4, 12))
            let blk0 = sb * 8
            let m = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(Vector128.CreateScalar(minBytes).AsByte()))
            mins <- Fma.MultiplyAdd(m * Vector256.Create dmin, Vector256.LoadUnsafe(&ss32.[blk0]), mins)
            for pp in 0 .. 1 do
                let v = load512 row (o + 16 + pp * 64)
                let lo = v &&& mask0F
                let hi = Avx512BW.ShiftRightLogical(v.AsUInt16(), 4uy).AsByte() &&& mask0F
                let e0 = sb * 256 + 128 * pp
                let xLo = Vector512.Create(load256s xs e0, load256s xs (e0 + 64))
                let xHi = Vector512.Create(load256s xs (e0 + 32), load256s xs (e0 + 96))
                let j = 4 * pp
                let b = blk0 + j
                let sc k = float32 (Q4KScales.byteAt scales (j + k))
                let fLo = halves (d * sc 0 * xd.[b]) (d * sc 2 * xd.[b + 2])
                let fHi = halves (d * sc 1 * xd.[b + 1]) (d * sc 3 * xd.[b + 3])
                acc <- Avx512F.FusedMultiplyAdd(Avx512F.ConvertToVector512Single(dpbusd lo xLo), fLo, acc)
                acc <- Avx512F.FusedMultiplyAdd(Avx512F.ConvertToVector512Single(dpbusd hi xHi), fHi, acc)
        Vector512.Sum acc - Vector256.Sum mins

    /// Per half super-block: ql0||ql1 is one 64-byte load; the 2-bit planes of qh are merged per
    /// 256-bit half so that q01 covers elements [0, 64) and q23 [64, 128) of the half, both
    /// contiguous in the activations. Each group of 4 int32 lanes is one 16-element scale group.
    [<MethodImpl(MethodImplOptions.AggressiveOptimization)>]
    static member Q6_K(row: ReadOnlySpan<byte>, superBlocks: int, xq: QuantizedActivations) =
        KernelGuards.check row superBlocks 210 256 xq
        let xs = xq.Quants
        let xd = xq.Scales
        let ss16 = xq.ScaledSums16
        let mutable acc = Vector512<float32>.Zero
        let mutable correction = Vector512<float32>.Zero
        for sb in 0 .. superBlocks - 1 do
            let o = sb * 210
            let d = half row (o + 208)
            // sum((q - 32)*x) = sum(q*x) - 32*sum(x): all 16 scale groups of the super-block in one FMA.
            let allScales = Avx512F.ConvertToVector512Single(Avx512F.ConvertToVector512Int32(Vector128.LoadUnsafe(&MemoryMarshal.GetReference row, unativeint (o + 192)).AsSByte()))
            correction <- Avx512F.FusedMultiplyAdd(allScales * Vector512.Create(32.0f * d), Vector512.LoadUnsafe(&ss16.[sb * 16]), correction)
            for n in 0 .. 1 do
                let scOff = o + 192 + n * 8
                let e0 = sb * 256 + n * 128
                let l = load512 row (o + n * 64)
                let qh = load256 row (o + 128 + n * 32)
                let h = Vector512.Create(qh, qh)
                let a = Avx512BW.ShiftLeftLogical(h.AsUInt16(), 4uy).AsByte() &&& mask30
                let b2 = Avx512BW.ShiftLeftLogical(h.AsUInt16(), 2uy).AsByte() &&& mask30
                let c = h &&& mask30
                let dd = Avx512BW.ShiftRightLogical(h.AsUInt16(), 2uy).AsByte() &&& mask30
                let q01 = (l &&& mask0F) ||| Vector512.Create(a.GetLower(), b2.GetUpper())
                let q23 = (Avx512BW.ShiftRightLogical(l.AsUInt16(), 4uy).AsByte() &&& mask0F) ||| Vector512.Create(c.GetLower(), dd.GetUpper())
                // Eight int8 scales -> two vectors of 16 int32 (each scale replicated over 4 lanes).
                let scales = Vector128.CreateScalar(BinaryPrimitives.ReadInt64LittleEndian(row.Slice(scOff, 8))).AsByte()
                let s01 = Avx512F.ConvertToVector512Int32(Ssse3.Shuffle(scales, replicateLow).AsSByte())
                let s23 = Avx512F.ConvertToVector512Int32(Ssse3.Shuffle(scales, replicateHigh).AsSByte())
                let blk = e0 / 32
                let p01 = Avx512F.MultiplyLow(dpbusd q01 (load512s xs e0), s01)
                let p23 = Avx512F.MultiplyLow(dpbusd q23 (load512s xs (e0 + 64)), s23)
                acc <- Avx512F.FusedMultiplyAdd(Avx512F.ConvertToVector512Single p01, halves (d * xd.[blk]) (d * xd.[blk + 1]), acc)
                acc <- Avx512F.FusedMultiplyAdd(Avx512F.ConvertToVector512Single p23, halves (d * xd.[blk + 2]) (d * xd.[blk + 3]), acc)
        Vector512.Sum acc - Vector512.Sum correction

    interface ICpuKernels with
        member _.Isa = X86Avx512Vnni
        member _.Description = $"x86-64 %s{CpuFeatures.describe ()}"
        member _.DotF32(a, b) = Avx2Kernels.Dot(a, b)
        member _.Axpy(alpha, x, y) = PortableKernels.AxpyImpl(alpha, x, y)
        member _.DotQ8_0(row, blocks, xq) = Avx512VnniKernels.Q8_0(row, blocks, xq)
        member _.DotQ4_K(row, superBlocks, xq) = Avx512VnniKernels.Q4_K(row, superBlocks, xq)
        member _.DotQ6_K(row, superBlocks, xq) = Avx512VnniKernels.Q6_K(row, superBlocks, xq)
