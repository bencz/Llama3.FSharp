namespace Llama3.Backend.Cpu

open System
open System.Buffers.Binary
open System.Runtime.CompilerServices
open System.Runtime.InteropServices
open System.Runtime.Intrinsics
open System.Runtime.Intrinsics.Arm
open Llama3.Core

/// ARM64 AdvSimd (NEON) kernels. Every quantized value fits in a signed byte (Q8_0 in [-127,127],
/// Q4_K nibbles in [0,15], Q6_K in [0,63]), so all formats use one signed int8 dot primitive:
///   - SDOT (Dp.DotProduct, ARMv8.2 DotProd: Apple M*, Graviton2+, Cortex-A76+) when available,
///   - otherwise SMULL/SMULL2 + SADALP (widening multiply, pairwise accumulate).
/// Only instantiate when <c>CpuFeatures.isSupported ArmNeon</c>.
[<Sealed>]
type NeonKernels() =
    static let mask0F = Vector128.Create(0x0Fuy)
    static let mask30 = Vector128.Create(0x30uy)

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    static let load (data: ReadOnlySpan<byte>) (offset: int) = Vector128.LoadUnsafe(&MemoryMarshal.GetReference data, unativeint offset)

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    static let loadS (data: sbyte[]) (offset: int) = Vector128.LoadUnsafe(&data.[offset])

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    static let half (row: ReadOnlySpan<byte>) (o: int) = Numerics.halfToSingle (uint16 row.[o] ||| (uint16 row.[o + 1] <<< 8))

    /// acc[i] += sum_{k<4} a[4i+k]*b[4i+k]  (16 signed bytes -> 4 int32 lanes).
    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    static let dot16 (acc: Vector128<int>) (a: Vector128<sbyte>) (b: Vector128<sbyte>) : Vector128<int> =
        if Dp.IsSupported then Dp.DotProduct(acc, a, b)
        else
            let lo = AdvSimd.MultiplyWideningLower(a.GetLower(), b.GetLower())
            let hi = AdvSimd.MultiplyWideningUpper(a, b)
            AdvSimd.AddPairwiseWideningAndAdd(AdvSimd.AddPairwiseWideningAndAdd(acc, lo), hi)

    /// One 16-value Q6_K group (unsigned 6-bit quants) * activations, scaled into the accumulator.
    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    static let q6Step (acc: Vector128<float32>) (q: Vector128<byte>) (x: Vector128<sbyte>) (factor: float32) =
        AdvSimd.FusedMultiplyAdd(acc, AdvSimd.ConvertToSingle(dot16 Vector128<int>.Zero (q.AsSByte()) x), Vector128.Create factor)

    [<MethodImpl(MethodImplOptions.AggressiveOptimization)>]
    static member Dot(a: ReadOnlySpan<float32>, b: ReadOnlySpan<float32>) : float32 =
        let n = a.Length
        let mutable i = 0
        let mutable acc0 = Vector128<float32>.Zero
        let mutable acc1 = Vector128<float32>.Zero
        let mutable acc2 = Vector128<float32>.Zero
        let mutable acc3 = Vector128<float32>.Zero
        while i + 16 <= n do
            acc0 <- AdvSimd.FusedMultiplyAdd(acc0, Vector128.Create<float32>(a.Slice(i, 4)), Vector128.Create<float32>(b.Slice(i, 4)))
            acc1 <- AdvSimd.FusedMultiplyAdd(acc1, Vector128.Create<float32>(a.Slice(i + 4, 4)), Vector128.Create<float32>(b.Slice(i + 4, 4)))
            acc2 <- AdvSimd.FusedMultiplyAdd(acc2, Vector128.Create<float32>(a.Slice(i + 8, 4)), Vector128.Create<float32>(b.Slice(i + 8, 4)))
            acc3 <- AdvSimd.FusedMultiplyAdd(acc3, Vector128.Create<float32>(a.Slice(i + 12, 4)), Vector128.Create<float32>(b.Slice(i + 12, 4)))
            i <- i + 16
        while i + 4 <= n do
            acc0 <- AdvSimd.FusedMultiplyAdd(acc0, Vector128.Create<float32>(a.Slice(i, 4)), Vector128.Create<float32>(b.Slice(i, 4)))
            i <- i + 4
        let mutable result = Vector128.Sum((acc0 + acc1) + (acc2 + acc3))
        while i < n do
            result <- result + a.[i] * b.[i]
            i <- i + 1
        result

    [<MethodImpl(MethodImplOptions.AggressiveOptimization)>]
    static member Q8_0(row: ReadOnlySpan<byte>, blocks: int, xq: QuantizedActivations) =
        KernelGuards.check row blocks 34 32 xq
        let xs = xq.Quants
        let xd = xq.Scales
        let mutable acc = Vector128<float32>.Zero
        for b in 0 .. blocks - 1 do
            let o = b * 34
            let isum = dot16 (dot16 Vector128<int>.Zero ((load row (o + 2)).AsSByte()) (loadS xs (b * 32))) ((load row (o + 18)).AsSByte()) (loadS xs (b * 32 + 16))
            acc <- AdvSimd.FusedMultiplyAdd(acc, AdvSimd.ConvertToSingle isum, Vector128.Create(half row o * xd.[b]))
        Vector128.Sum acc

    [<MethodImpl(MethodImplOptions.AggressiveOptimization)>]
    static member Q4_K(row: ReadOnlySpan<byte>, superBlocks: int, xq: QuantizedActivations) =
        KernelGuards.check row superBlocks 144 256 xq
        let xs = xq.Quants
        let xd = xq.Scales
        let sums = xq.Sums16
        let mutable acc = Vector128<float32>.Zero
        let mutable minCorrection = 0.0f
        for sb in 0 .. superBlocks - 1 do
            let o = sb * 144
            let d = half row o
            let dmin = half row (o + 2)
            let scaleBytes = row.Slice(o + 4, 12)
            for p in 0 .. 3 do
                let struct (sc0, m0) = Dequantize.scaleMinK4 (2 * p) scaleBytes
                let struct (sc1, m1) = Dequantize.scaleMinK4 (2 * p + 1) scaleBytes
                let e = sb * 256 + p * 64
                let blk = e / 32
                minCorrection <-
                    minCorrection
                    + dmin * (float32 m0 * xd.[blk] * float32 (sums.[2 * blk] + sums.[2 * blk + 1])
                              + float32 m1 * xd.[blk + 1] * float32 (sums.[2 * blk + 2] + sums.[2 * blk + 3]))
                let v0 = load row (o + 16 + p * 32)
                let v1 = load row (o + 32 + p * 32)
                let lo = dot16 (dot16 Vector128<int>.Zero (AdvSimd.And(v0, mask0F).AsSByte()) (loadS xs e)) (AdvSimd.And(v1, mask0F).AsSByte()) (loadS xs (e + 16))
                let hi = dot16 (dot16 Vector128<int>.Zero (AdvSimd.ShiftRightLogical(v0, 4uy).AsSByte()) (loadS xs (e + 32))) (AdvSimd.ShiftRightLogical(v1, 4uy).AsSByte()) (loadS xs (e + 48))
                acc <- AdvSimd.FusedMultiplyAdd(acc, AdvSimd.ConvertToSingle lo, Vector128.Create(d * float32 sc0 * xd.[blk]))
                acc <- AdvSimd.FusedMultiplyAdd(acc, AdvSimd.ConvertToSingle hi, Vector128.Create(d * float32 sc1 * xd.[blk + 1]))
        Vector128.Sum acc - minCorrection

    /// Each 16-byte vector is exactly one Q6_K scale group, so scales apply as an int multiply-add.
    [<MethodImpl(MethodImplOptions.AggressiveOptimization)>]
    static member Q6_K(row: ReadOnlySpan<byte>, superBlocks: int, xq: QuantizedActivations) =
        KernelGuards.check row superBlocks 210 256 xq
        let xs = xq.Quants
        let xd = xq.Scales
        let sums = xq.Sums16
        let mutable acc = Vector128<float32>.Zero
        let mutable correction = 0.0f
        for sb in 0 .. superBlocks - 1 do
            let o = sb * 210
            let d = half row (o + 208)
            for n in 0 .. 1 do
                let qlOff = o + n * 64
                let qhOff = o + 128 + n * 32
                let scOff = o + 192 + n * 8
                let eBase = sb * 256 + n * 128
                for half16 in 0 .. 1 do
                    let h = 16 * half16
                    let ql0 = load row (qlOff + h)
                    let ql1 = load row (qlOff + 32 + h)
                    let qh = load row (qhOff + h)
                    let q0 = AdvSimd.Or(AdvSimd.And(ql0, mask0F), AdvSimd.And(AdvSimd.ShiftLeftLogical(qh, 4uy), mask30))
                    let q1 = AdvSimd.Or(AdvSimd.And(ql1, mask0F), AdvSimd.And(AdvSimd.ShiftLeftLogical(qh, 2uy), mask30))
                    let q2 = AdvSimd.Or(AdvSimd.ShiftRightLogical(ql0, 4uy), AdvSimd.And(qh, mask30))
                    let q3 = AdvSimd.Or(AdvSimd.ShiftRightLogical(ql1, 4uy), AdvSimd.And(AdvSimd.ShiftRightLogical(qh, 2uy), mask30))
                    // Unrolled by hand: a local helper would capture the row span (illegal) or allocate.
                    let e0 = eBase + h
                    let f0 = d * xd.[e0 / 32] * float32 (int (sbyte row.[scOff + half16]))
                    let f1 = d * xd.[(e0 + 32) / 32] * float32 (int (sbyte row.[scOff + 2 + half16]))
                    let f2 = d * xd.[(e0 + 64) / 32] * float32 (int (sbyte row.[scOff + 4 + half16]))
                    let f3 = d * xd.[(e0 + 96) / 32] * float32 (int (sbyte row.[scOff + 6 + half16]))
                    acc <- q6Step acc q0 (loadS xs e0) f0
                    acc <- q6Step acc q1 (loadS xs (e0 + 32)) f1
                    acc <- q6Step acc q2 (loadS xs (e0 + 64)) f2
                    acc <- q6Step acc q3 (loadS xs (e0 + 96)) f3
                    correction <-
                        correction
                        + 32.0f * (f0 * float32 sums.[e0 / 16] + f1 * float32 sums.[(e0 + 32) / 16]
                                   + f2 * float32 sums.[(e0 + 64) / 16] + f3 * float32 sums.[(e0 + 96) / 16])
        Vector128.Sum acc - correction

    interface ICpuKernels with
        member _.Isa = ArmNeon
        member _.Description = $"ARM64 %s{CpuFeatures.describe ()}"
        member _.DotF32(a, b) = NeonKernels.Dot(a, b)
        member _.Axpy(alpha, x, y) = PortableKernels.AxpyImpl(alpha, x, y)
        member _.DotQ8_0(row, blocks, xq) = NeonKernels.Q8_0(row, blocks, xq)
        member _.DotQ4_K(row, superBlocks, xq) = NeonKernels.Q4_K(row, superBlocks, xq)
        member _.DotQ6_K(row, superBlocks, xq) = NeonKernels.Q6_K(row, superBlocks, xq)
