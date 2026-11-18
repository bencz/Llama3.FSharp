namespace Llama3.Core

open System
open System.Runtime.InteropServices

/// Reference decoders (scalar, exact). They define the semantics every SIMD/GPU kernel must match.
module Dequantize =
    open Numerics

    /// Q4_K scale/min extraction (<c>get_scale_min_k4</c>): 8 six-bit scales and mins packed in 12 bytes.
    let inline scaleMinK4 (j: int) (scales: ReadOnlySpan<byte>) : struct (int * int) =
        if j < 4 then
            struct (int (scales.[j] &&& 63uy), int (scales.[j + 4] &&& 63uy))
        else
            let d = int (scales.[j + 4] &&& 0xFuy) ||| (int (scales.[j - 4] >>> 6) <<< 4)
            let m = int (scales.[j + 4] >>> 4) ||| (int (scales.[j] >>> 6) <<< 4)
            struct (d, m)

    let blockQ8_0 (src: ReadOnlySpan<byte>) (dst: Span<float32>) =
        let d = readHalf src 0
        for i in 0 .. 31 do
            dst.[i] <- d * float32 (sbyte src.[2 + i])

    let blockQ4_0 (src: ReadOnlySpan<byte>) (dst: Span<float32>) =
        let d = readHalf src 0
        for j in 0 .. 15 do
            let q = src.[2 + j]
            dst.[j] <- d * float32 (int (q &&& 0xFuy) - 8)
            dst.[j + 16] <- d * float32 (int (q >>> 4) - 8)

    let blockQ4_K (src: ReadOnlySpan<byte>) (dst: Span<float32>) =
        let d = readHalf src 0
        let dmin = readHalf src 2
        let scales = src.Slice(4, 12)
        let qs = src.Slice(16, 128)
        for p in 0 .. 3 do
            let struct (sc0, m0) = scaleMinK4 (2 * p) scales
            let struct (sc1, m1) = scaleMinK4 (2 * p + 1) scales
            let d0 = d * float32 sc0
            let d1 = d * float32 sc1
            let mm0 = dmin * float32 m0
            let mm1 = dmin * float32 m1
            for l in 0 .. 31 do
                let q = qs.[p * 32 + l]
                dst.[p * 64 + l] <- d0 * float32 (q &&& 0xFuy) - mm0
                dst.[p * 64 + 32 + l] <- d1 * float32 (q >>> 4) - mm1

    let blockQ6_K (src: ReadOnlySpan<byte>) (dst: Span<float32>) =
        let d = readHalf src 208
        for n in 0 .. 1 do
            let ql = src.Slice(n * 64, 64)
            let qh = src.Slice(128 + n * 32, 32)
            let sc = src.Slice(192 + n * 8, 8)
            let y = dst.Slice(n * 128, 128)
            for l in 0 .. 31 do
                let is' = l / 16
                let h = int qh.[l]
                let q1 = (int (ql.[l] &&& 0xFuy) ||| ((h &&& 3) <<< 4)) - 32
                let q2 = (int (ql.[l + 32] &&& 0xFuy) ||| (((h >>> 2) &&& 3) <<< 4)) - 32
                let q3 = (int (ql.[l] >>> 4) ||| (((h >>> 4) &&& 3) <<< 4)) - 32
                let q4 = (int (ql.[l + 32] >>> 4) ||| (((h >>> 6) &&& 3) <<< 4)) - 32
                y.[l] <- d * float32 (sbyte sc.[is']) * float32 q1
                y.[l + 32] <- d * float32 (sbyte sc.[is' + 2]) * float32 q2
                y.[l + 64] <- d * float32 (sbyte sc.[is' + 4]) * float32 q3
                y.[l + 96] <- d * float32 (sbyte sc.[is' + 6]) * float32 q4

    /// Decodes <paramref name="count"/> consecutive elements starting at a block boundary.
    let decode (dtype: GgmlType) (src: ReadOnlySpan<byte>) (dst: Span<float32>) =
        let count = dst.Length
        match dtype with
        | GgmlType.F32 -> MemoryMarshal.Cast<byte, float32>(src.Slice(0, count * 4)).CopyTo(dst)
        | GgmlType.F16 ->
            let halves = MemoryMarshal.Cast<byte, uint16>(src.Slice(0, count * 2))
            for i in 0 .. count - 1 do dst.[i] <- halfToSingle halves.[i]
        | GgmlType.BF16 ->
            let halves = MemoryMarshal.Cast<byte, uint16>(src.Slice(0, count * 2))
            for i in 0 .. count - 1 do dst.[i] <- bfloat16ToSingle halves.[i]
        | _ ->
            let bs = GgmlTypes.blockSize dtype
            let ts = GgmlTypes.typeSize dtype
            if count % bs <> 0 then invalidArg (nameof dst) $"%d{count} is not a multiple of %d{bs}"
            for b in 0 .. count / bs - 1 do
                let block = src.Slice(b * ts, ts)
                let out = dst.Slice(b * bs, bs)
                match dtype with
                | GgmlType.Q8_0 -> blockQ8_0 block out
                | GgmlType.Q4_0 -> blockQ4_0 block out
                | GgmlType.Q4_K -> blockQ4_K block out
                | GgmlType.Q6_K -> blockQ6_K block out
                | other -> raise (NotSupportedException $"Decoding %A{other} is not implemented")

    /// Decodes one row of a 2-D tensor (embedding lookup, tests).
    let row (tensor: HostTensor) (rowIndex: int) (dst: Span<float32>) =
        decode tensor.Type (tensor.RowSpan rowIndex) (dst.Slice(0, tensor.Columns))

    /// Decodes a whole tensor.
    let toSingles (tensor: HostTensor) : float32[] =
        let result = Array.zeroCreate<float32> (int tensor.ElementCount)
        decode tensor.Type tensor.Bytes (Span<float32>(result))
        result

/// Load-time format normalisation. Formats without a native kernel are re-encoded losslessly
/// where possible (Q4_0 -> Q8_0: the values (q - 8)*d are exactly representable with the same d),
/// otherwise widened to F32.
module TensorConversion =

    /// Exact Q4_0 -> Q8_0 re-encoding: 18-byte blocks become 34-byte blocks with the same scale.
    let q4_0ToQ8_0 (tensor: HostTensor) : HostTensor =
        let source = tensor.Bytes
        let blocks = source.Length / 18
        let output = GC.AllocateUninitializedArray<byte>(blocks * 34)
        for b in 0 .. blocks - 1 do
            let s = b * 18
            let d = b * 34
            output.[d] <- source.[s]
            output.[d + 1] <- source.[s + 1]
            for j in 0 .. 15 do
                let q = source.[s + 2 + j]
                output.[d + 2 + j] <- byte (sbyte (int (q &&& 0xFuy) - 8))
                output.[d + 2 + 16 + j] <- byte (sbyte (int (q >>> 4) - 8))
        HostTensor(tensor.Name, GgmlType.Q8_0, tensor.Shape, output)

    /// Widens to F32 directly into the tensor's byte storage (no intermediate float array).
    let private toF32 (tensor: HostTensor) : HostTensor =
        let bytes = tensor.ElementCount * 4L
        if bytes > int64 Array.MaxLength then
            raise (NotSupportedException $"Tensor '%s{tensor.Name}' (%A{tensor.Type}) would need %d{bytes} bytes as F32, beyond the 2 GiB host tensor limit")
        let output = GC.AllocateUninitializedArray<byte>(int bytes)
        Dequantize.decode tensor.Type tensor.Bytes (MemoryMarshal.Cast<byte, float32>(Span<byte>(output)))
        HostTensor(tensor.Name, GgmlType.F32, tensor.Shape, output)

    let toComputeFormat (tensor: HostTensor) : HostTensor =
        if GgmlTypes.isNativeComputeType tensor.Type then tensor
        elif tensor.Type = GgmlType.Q4_0 then q4_0ToQ8_0 tensor
        elif GgmlTypes.isDecodable tensor.Type then toF32 tensor
        else raise (NotSupportedException $"Tensor '%s{tensor.Name}' uses %A{tensor.Type}, which this runtime cannot decode yet")

    /// Vectors consumed element-wise (norm weights, biases, RoPE factors) are always stored as F32.
    let toSingleVector (tensor: HostTensor) : HostTensor =
        if tensor.Type = GgmlType.F32 then tensor else toF32 tensor
