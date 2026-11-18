namespace Llama3.Backend.Cpu

open System
open System.Numerics
open System.Runtime.CompilerServices

/// Activation vector quantized to symmetric int8 blocks of 32 (scale = max|x| / 127), plus the
/// sum of quants of every 16-element group. The sums let K-quant kernels apply their per-group
/// offsets (Q4_K mins, Q6_K's −32 bias) with one scalar FMA instead of touching the data again.
[<Sealed>]
type QuantizedActivations(capacity: int) =
    let quants = Array.zeroCreate<sbyte> capacity
    let scales = Array.zeroCreate<float32> (capacity / 32)
    let sums16 = Array.zeroCreate<int> (capacity / 16)
    let scaledSums32 = Array.zeroCreate<float32> (capacity / 32)
    let scaledSums16 = Array.zeroCreate<float32> (capacity / 16)
    let mutable length = 0

    member _.Quants = quants
    member _.Scales = scales
    member _.Sums16 = sums16
    /// dx_b · Σq over each 32-element block (Q4_K min correction, Q8_0 bias removal).
    member _.ScaledSums32 = scaledSums32
    /// dx · Σq over each 16-element group (Q6_K −32 offset correction).
    member _.ScaledSums16 = scaledSums16
    member _.Length = length

    [<MethodImpl(MethodImplOptions.AggressiveOptimization)>]
    member _.Quantize(x: ReadOnlySpan<float32>) =
        if x.Length % 32 <> 0 || x.Length > capacity then
            invalidArg (nameof x) $"Cannot quantize %d{x.Length} activations (capacity %d{capacity}, block 32)"
        length <- x.Length
        let width = Vector<float32>.Count
        for b in 0 .. x.Length / 32 - 1 do
            let block = x.Slice(b * 32, 32)
            let mutable amax = 0.0f
            if Vector.IsHardwareAccelerated && 32 % width = 0 then
                let mutable m = Vector<float32>.Zero
                for i in 0 .. width .. 31 do
                    m <- Vector.Max(m, Vector.Abs(Vector<float32>(block.Slice(i, width))))
                for k in 0 .. width - 1 do amax <- MathF.Max(amax, m.[k])
            else
                for v in block do amax <- MathF.Max(amax, MathF.Abs v)
            let d = amax / 127.0f
            let id = if d > 0.0f then 1.0f / d else 0.0f
            scales.[b] <- d
            let mutable s0 = 0
            let mutable s1 = 0
            for i in 0 .. 31 do
                let q = int (MathF.Round(block.[i] * id, MidpointRounding.AwayFromZero))
                quants.[b * 32 + i] <- sbyte q
                if i < 16 then s0 <- s0 + q else s1 <- s1 + q
            sums16.[2 * b] <- s0
            sums16.[2 * b + 1] <- s1
            scaledSums16.[2 * b] <- d * float32 s0
            scaledSums16.[2 * b + 1] <- d * float32 s1
            scaledSums32.[b] <- d * float32 (s0 + s1)

/// Argument validation shared by every integer kernel: the vector loads are unchecked, so the row
/// and the activations must be proven long enough once per call.
module KernelGuards =
    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    let check (row: ReadOnlySpan<byte>) (blocks: int) (blockBytes: int) (blockElements: int) (xq: QuantizedActivations) =
        if blocks < 0 || row.Length < blocks * blockBytes || xq.Length < blocks * blockElements then
            raise (ArgumentException $"Kernel arguments too short: %d{blocks} blocks need %d{blocks * blockBytes} weight bytes (got %d{row.Length}) and %d{blocks * blockElements} activations (got %d{xq.Length})")
