namespace Llama3.Backend.Cuda

open System
open Llama3.Core

/// Device storage formats. Q8_0 (34 B) and Q6_K (210 B) blocks are padded to a multiple of
/// 4 bytes at upload so kernels can read whole 32-bit words; every other format is stored as-is.
///   Q8_0 -> [d:f16][pad:2][qs:32]                       36 B
///   Q4_K -> [d:f16][dmin:f16][scales:8xu8][mins:8xu8][qs:128]   148 B  (6-bit scales unpacked)
///   Q6_K -> [ql:128][qh:64][scales:16][d:f16][pad:2]   212 B
module GpuLayout =

    let private paddedBlock (t: GgmlType) =
        match t with
        | GgmlType.Q8_0 -> Some 36
        | GgmlType.Q4_K -> Some 148
        | GgmlType.Q6_K -> Some 212
        | _ -> None

    /// Bytes of one row on the device (rows are whole blocks, so padding is per row too).
    let deviceRowBytes (t: HostTensor) : int64 =
        match paddedBlock t.Type with
        | Some padded -> int64 t.Columns / int64 (GgmlTypes.blockSize t.Type) * int64 padded
        | None -> int64 t.RowBytes

    /// Bytes a tensor occupies on the device.
    let deviceBytes (t: HostTensor) : int64 = deviceRowBytes t * int64 t.Rows

    /// Converts host blocks (a whole number of them) into the device layout.
    let private convert (dtype: GgmlType) (src: ReadOnlySpan<byte>) (destination: Span<byte>) =
        match dtype with
        | GgmlType.Q8_0 ->
            for b in 0 .. src.Length / 34 - 1 do
                let s = src.Slice(b * 34, 34)
                let d = destination.Slice(b * 36, 36)
                s.Slice(0, 2).CopyTo d
                d.[2] <- 0uy
                d.[3] <- 0uy
                s.Slice(2, 32).CopyTo(d.Slice 4)
        | GgmlType.Q4_K ->
            // get_scale_min_k4 resolved once at upload: the kernel then reads plain bytes instead of
            // reassembling 6-bit fields with a lane-divergent branch.
            for b in 0 .. src.Length / 144 - 1 do
                let s = src.Slice(b * 144, 144)
                let d = destination.Slice(b * 148, 148)
                s.Slice(0, 4).CopyTo d
                let packed = s.Slice(4, 12)
                for j in 0 .. 7 do
                    let struct (sc, m) = Dequantize.scaleMinK4 j packed
                    d.[4 + j] <- byte sc
                    d.[12 + j] <- byte m
                s.Slice(16, 128).CopyTo(d.Slice 20)
        | GgmlType.Q6_K ->
            for b in 0 .. src.Length / 210 - 1 do
                let d = destination.Slice(b * 212, 212)
                src.Slice(b * 210, 210).CopyTo d
                d.[210] <- 0uy
                d.[211] <- 0uy
        | _ -> src.CopyTo destination

    /// Writes rows [firstRow, firstRow + rowCount) of <paramref name="t"/> in device layout. The
    /// source may be a memory-mapped file: pages are faulted in as the rows are read.
    let writeRows (t: HostTensor) (firstRow: int) (rowCount: int) (destination: Span<byte>) =
        let needed = deviceRowBytes t * int64 rowCount
        if int64 destination.Length < needed then
            invalidArg (nameof destination) $"Destination of %d{destination.Length} bytes cannot hold %d{needed}"
        convert t.Type (t.RowsSpan(firstRow, rowCount)) destination

    /// Whole tensor in device layout (the destination must be below 2 GiB, e.g. a packed layer).
    let write (t: HostTensor) (destination: Span<byte>) = writeRows t 0 t.Rows destination
