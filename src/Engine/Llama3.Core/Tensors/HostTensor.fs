namespace Llama3.Core

open System
open System.Runtime.InteropServices

/// A tensor resident in host memory (or mapped from a file), stored in its on-disk format.
/// <c>Shape</c> follows ggml order: <c>Shape[0]</c> is the contiguous (column) dimension,
/// so a weight matrix W(out, in) has Shape = [| in; out |] and each row is one output neuron.
[<Sealed>]
type HostTensor(name: string, dtype: GgmlType, shape: int64[], storage: TensorStorage) =
    let elements = shape |> Array.fold (*) 1L
    let sizeInBytes = GgmlTypes.rowBytes dtype elements
    let columns = int shape.[0]
    let rows = elements / shape.[0]
    let rowBytes = GgmlTypes.rowBytes dtype (int64 columns)
    do
        if storage.Length <> sizeInBytes then
            invalidArg (nameof storage) $"Tensor '%s{name}' (%A{dtype} %A{shape}) needs %d{sizeInBytes} bytes, got %d{storage.Length}"
        if rows > int64 Int32.MaxValue then
            invalidArg (nameof shape) $"Tensor '%s{name}' has %d{rows} rows (at most %d{Int32.MaxValue})"

    new(name: string, dtype: GgmlType, shape: int64[], data: byte[]) = HostTensor(name, dtype, shape, ArrayStorage data)

    member _.Name = name
    member _.Type = dtype
    member _.Shape = shape
    member _.Storage = storage
    member _.Columns = columns
    member _.Rows = int rows
    member _.ElementCount = elements
    member _.SizeInBytes = sizeInBytes
    member _.RowBytes = int rowBytes

    /// Raw bytes of one row.
    member _.RowSpan(row: int) : ReadOnlySpan<byte> = storage.Slice(int64 row * rowBytes, int rowBytes)

    /// Raw bytes of rows [first, first + count) (at most 2 GiB per call).
    member _.RowsSpan(first: int, count: int) : ReadOnlySpan<byte> =
        let bytes = int64 count * rowBytes
        if bytes > int64 Int32.MaxValue then invalidArg (nameof count) $"%d{count} rows of '%s{name}' exceed one span"
        storage.Slice(int64 first * rowBytes, int bytes)

    /// The whole payload as one span (tensors up to 2 GiB; use RowsSpan for larger ones).
    member _.Bytes: ReadOnlySpan<byte> =
        if sizeInBytes > int64 Int32.MaxValue then
            invalidOp $"Tensor '%s{name}' (%d{sizeInBytes} bytes) does not fit in one span; read it by rows"
        storage.Slice(0L, int sizeInBytes)

    /// Float32 view (only valid for F32 tensors).
    member this.AsSingles() : ReadOnlySpan<float32> =
        if dtype <> GgmlType.F32 then invalidOp $"Tensor '%s{name}' is %A{dtype}, not F32"
        MemoryMarshal.Cast<byte, float32>(this.Bytes)

    override _.ToString() = $"%s{name} %A{dtype} %A{shape}"

    /// Builds an F32 tensor from float values.
    static member OfSingles(name: string, shape: int64[], values: ReadOnlySpan<float32>) =
        HostTensor(name, GgmlType.F32, shape, MemoryMarshal.AsBytes(values).ToArray())
