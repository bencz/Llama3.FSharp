namespace Llama3.Backend.Cuda

open System
open ILGPU
open ILGPU.Runtime
open Llama3.Core

[<Sealed>]
type CudaBuffer(buffer: MemoryBuffer1D<float32, Stride1D.Dense>) =
    member _.View: ArrayView<float32> = buffer.View.AsContiguous()
    interface IBuffer with
        member _.Length = int buffer.Length
        member _.Dispose() = buffer.Dispose()

/// A weight in device memory. Either owns its allocation (resident) or is a view into a
/// streaming slot (owner = None).
[<Sealed>]
type CudaWeight(name: string, dtype: GgmlType, rows: int, cols: int, bytes: ArrayView<byte>, owner: IDisposable option) =
    member _.Bytes = bytes
    member _.Singles: ArrayView<float32> = bytes.Cast<float32>()
    /// Word view consumed by the F16 / quantized kernels (all device layouts are 4-byte aligned).
    member _.Words: ArrayView<uint32> = bytes.Cast<uint32>()

    /// Non-owning view of rows [firstRow, firstRow + rowCount). Device rows are whole blocks
    /// (padded layouts pad per block), so a row range is a contiguous, word-aligned byte range.
    member _.Slice(firstRow: int, rowCount: int) : CudaWeight =
        if firstRow < 0 || rowCount < 0 || firstRow + rowCount > rows then
            invalidArg (nameof rowCount) $"Rows [%d{firstRow}, %d{firstRow + rowCount}) outside '%s{name}' (%d{rows} rows)"
        let rowBytes = bytes.Length / int64 rows
        new CudaWeight(name, dtype, rowCount, cols, bytes.SubView(int64 firstRow * rowBytes, int64 rowCount * rowBytes), None)

    interface IWeight with
        member _.Name = name
        member _.Type = dtype
        member _.Rows = rows
        member _.Columns = cols
        member _.SizeInBytes = bytes.LengthInBytes
        member _.Dispose() =
            match owner with
            | Some o -> o.Dispose() // MemoryBuffer.Dispose is idempotent
            | None -> ()

[<Sealed>]
type CudaKvCache(accelerator: Accelerator, capacity: int, kvDim: int) =
    let keys = accelerator.Allocate1D<float32>(int64 capacity * int64 kvDim)
    let values =
        try accelerator.Allocate1D<float32>(int64 capacity * int64 kvDim)
        with _ ->
            keys.Dispose()
            reraise ()
    let mutable disposed = false
    member _.Keys: ArrayView<float32> = keys.View.AsContiguous()
    member _.Values: ArrayView<float32> = values.View.AsContiguous()
    interface IKvCache with
        member _.Capacity = capacity
        member _.KvDim = kvDim
        member _.SizeInBytes = keys.LengthInBytes + values.LengthInBytes
        member _.Dispose() =
            if not disposed then
                disposed <- true
                keys.Dispose()
                values.Dispose()
