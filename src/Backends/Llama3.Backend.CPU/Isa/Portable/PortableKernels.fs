namespace Llama3.Backend.Cpu

open System
open System.Buffers.Binary
open System.Numerics
open System.Runtime.CompilerServices
open System.Runtime.InteropServices
open Llama3.Core

/// Architecture-neutral kernels: Vector&lt;T&gt; (sized by the JIT to the machine: 128-bit on
/// NEON/SSE, 256-bit on AVX2, 512-bit with AVX-512) for float math and exact scalar integer math
/// for quantized rows. Also serves as the oracle the ISA-specific kernels are tested against.
[<Sealed>]
type PortableKernels() =

    [<MethodImpl(MethodImplOptions.AggressiveOptimization)>]
    static member Dot(a: ReadOnlySpan<float32>, b: ReadOnlySpan<float32>) : float32 =
        let n = a.Length
        let w = Vector<float32>.Count
        let mutable i = 0
        let mutable result = 0.0f
        if Vector.IsHardwareAccelerated && n >= 4 * w then
            let mutable acc0 = Vector<float32>.Zero
            let mutable acc1 = Vector<float32>.Zero
            let mutable acc2 = Vector<float32>.Zero
            let mutable acc3 = Vector<float32>.Zero
            while i + 4 * w <= n do
                acc0 <- Vector.FusedMultiplyAdd(Vector<float32>(a.Slice(i, w)), Vector<float32>(b.Slice(i, w)), acc0)
                acc1 <- Vector.FusedMultiplyAdd(Vector<float32>(a.Slice(i + w, w)), Vector<float32>(b.Slice(i + w, w)), acc1)
                acc2 <- Vector.FusedMultiplyAdd(Vector<float32>(a.Slice(i + 2 * w, w)), Vector<float32>(b.Slice(i + 2 * w, w)), acc2)
                acc3 <- Vector.FusedMultiplyAdd(Vector<float32>(a.Slice(i + 3 * w, w)), Vector<float32>(b.Slice(i + 3 * w, w)), acc3)
                i <- i + 4 * w
            result <- Vector.Sum((acc0 + acc1) + (acc2 + acc3))
        if Vector.IsHardwareAccelerated then
            let mutable acc = Vector<float32>.Zero
            while i + w <= n do
                acc <- Vector.FusedMultiplyAdd(Vector<float32>(a.Slice(i, w)), Vector<float32>(b.Slice(i, w)), acc)
                i <- i + w
            result <- result + Vector.Sum acc
        while i < n do
            result <- result + a.[i] * b.[i]
            i <- i + 1
        result

    [<MethodImpl(MethodImplOptions.AggressiveOptimization)>]
    static member AxpyImpl(alpha: float32, x: ReadOnlySpan<float32>, y: Span<float32>) =
        let n = x.Length
        let w = Vector<float32>.Count
        let mutable i = 0
        if Vector.IsHardwareAccelerated then
            let va = Vector<float32>(alpha)
            while i + w <= n do
                let target = y.Slice(i, w)
                Vector.FusedMultiplyAdd(va, Vector<float32>(x.Slice(i, w)), Vector<float32>(Span<float32>.op_Implicit target)).CopyTo(target)
                i <- i + w
        while i < n do
            y.[i] <- y.[i] + alpha * x.[i]
            i <- i + 1

    static member Half(row: ReadOnlySpan<byte>, o: int) = Numerics.halfToSingle (uint16 row.[o] ||| (uint16 row.[o + 1] <<< 8))

    [<MethodImpl(MethodImplOptions.AggressiveOptimization)>]
    static member Q8_0(row: ReadOnlySpan<byte>, blocks: int, xq: QuantizedActivations) =
        KernelGuards.check row blocks 34 32 xq
        let xs = xq.Quants
        let xd = xq.Scales
        let mutable acc = 0.0f
        for b in 0 .. blocks - 1 do
            let o = b * 34
            let mutable isum = 0
            for i in 0 .. 31 do
                isum <- isum + int (sbyte row.[o + 2 + i]) * int xs.[b * 32 + i]
            acc <- acc + float32 isum * PortableKernels.Half(row, o) * xd.[b]
        acc

    [<MethodImpl(MethodImplOptions.AggressiveOptimization)>]
    static member Q4_K(row: ReadOnlySpan<byte>, superBlocks: int, xq: QuantizedActivations) =
        KernelGuards.check row superBlocks 144 256 xq
        let xs = xq.Quants
        let xd = xq.Scales
        let sums = xq.Sums16
        let mutable acc = 0.0f
        for sb in 0 .. superBlocks - 1 do
            let o = sb * 144
            let d = PortableKernels.Half(row, o)
            let dmin = PortableKernels.Half(row, o + 2)
            let scales = row.Slice(o + 4, 12)
            for p in 0 .. 3 do
                let struct (sc0, m0) = Dequantize.scaleMinK4 (2 * p) scales
                let struct (sc1, m1) = Dequantize.scaleMinK4 (2 * p + 1) scales
                let e = sb * 256 + p * 64
                let blk = e / 32
                let mutable sLo = 0
                let mutable sHi = 0
                for l in 0 .. 31 do
                    let q = row.[o + 16 + p * 32 + l]
                    sLo <- sLo + int (q &&& 0xFuy) * int xs.[e + l]
                    sHi <- sHi + int (q >>> 4) * int xs.[e + 32 + l]
                acc <-
                    acc
                    + d * (float32 sc0 * xd.[blk] * float32 sLo + float32 sc1 * xd.[blk + 1] * float32 sHi)
                    - dmin * (float32 m0 * xd.[blk] * float32 (sums.[2 * blk] + sums.[2 * blk + 1])
                              + float32 m1 * xd.[blk + 1] * float32 (sums.[2 * blk + 2] + sums.[2 * blk + 3]))
        acc

    [<MethodImpl(MethodImplOptions.AggressiveOptimization)>]
    static member Q6_K(row: ReadOnlySpan<byte>, superBlocks: int, xq: QuantizedActivations) =
        KernelGuards.check row superBlocks 210 256 xq
        let xs = xq.Quants
        let xd = xq.Scales
        let mutable acc = 0.0f
        for sb in 0 .. superBlocks - 1 do
            let o = sb * 210
            let d = PortableKernels.Half(row, o + 208)
            for n in 0 .. 1 do
                let qlOff = o + n * 64
                let qhOff = o + 128 + n * 32
                let scOff = o + 192 + n * 8
                for k in 0 .. 3 do
                    let e = sb * 256 + n * 128 + 32 * k
                    let mutable sA = 0
                    let mutable sB = 0
                    for l in 0 .. 31 do
                        let h = int row.[qhOff + l]
                        let q =
                            match k with
                            | 0 -> int (row.[qlOff + l] &&& 0xFuy) ||| ((h &&& 3) <<< 4)
                            | 1 -> int (row.[qlOff + 32 + l] &&& 0xFuy) ||| (((h >>> 2) &&& 3) <<< 4)
                            | 2 -> int (row.[qlOff + l] >>> 4) ||| (((h >>> 4) &&& 3) <<< 4)
                            | _ -> int (row.[qlOff + 32 + l] >>> 4) ||| (((h >>> 6) &&& 3) <<< 4)
                        let p = (q - 32) * int xs.[e + l]
                        if l < 16 then sA <- sA + p else sB <- sB + p
                    let sa = int (sbyte row.[scOff + 2 * k])
                    let sbv = int (sbyte row.[scOff + 2 * k + 1])
                    acc <- acc + d * xd.[e / 32] * float32 (sa * sA + sbv * sB)
        acc

    interface ICpuKernels with
        member _.Isa = Portable
        member _.Description = $"portable (Vector<T> %d{Vector<byte>.Count * 8}-bit + scalar int8)"
        member _.DotF32(a, b) = PortableKernels.Dot(a, b)
        member _.Axpy(alpha, x, y) = PortableKernels.AxpyImpl(alpha, x, y)
        member _.DotQ8_0(row, blocks, xq) = PortableKernels.Q8_0(row, blocks, xq)
        member _.DotQ4_K(row, superBlocks, xq) = PortableKernels.Q4_K(row, superBlocks, xq)
        member _.DotQ6_K(row, superBlocks, xq) = PortableKernels.Q6_K(row, superBlocks, xq)
