namespace Llama3.Core

open System
open System.Collections.Generic
open System.IO
open System.Text
open Microsoft.Win32.SafeHandles

/// A GGUF metadata value. Arrays are stored as typed, unboxed .NET arrays because tokenizer
/// vocabularies (150k+ entries) live in them.
type GgufValue =
    | GInt of int64
    | GUInt of uint64
    | GFloat of float
    | GBool of bool
    | GString of string
    | GStringArray of string[]
    | GIntArray of int64[]
    | GFloatArray of float[]
    | GBoolArray of bool[]

/// Wire-level decoding of GGUF primitives. Every length and count read from the file is bounded
/// by the bytes that remain in the stream before anything is allocated, so a corrupt or hostile
/// header fails with <see cref="InvalidDataException"/> instead of exhausting memory.
module internal GgufWire =
    let magic = 0x46554747u // "GGUF" little-endian

    /// ggml's GGML_MAX_DIMS.
    [<Literal>]
    let MaxDims = 4

    // GGUF value type ids.
    [<Literal>]
    let TUInt8 = 0u
    [<Literal>]
    let TInt8 = 1u
    [<Literal>]
    let TUInt16 = 2u
    [<Literal>]
    let TInt16 = 3u
    [<Literal>]
    let TUInt32 = 4u
    [<Literal>]
    let TInt32 = 5u
    [<Literal>]
    let TFloat32 = 6u
    [<Literal>]
    let TBool = 7u
    [<Literal>]
    let TString = 8u
    [<Literal>]
    let TArray = 9u
    [<Literal>]
    let TUInt64 = 10u
    [<Literal>]
    let TInt64 = 11u
    [<Literal>]
    let TFloat64 = 12u

    let remaining (r: BinaryReader) = r.BaseStream.Length - r.BaseStream.Position

    let invalid (message: string) = raise (InvalidDataException message)

    /// Validates a count read from the file: each element needs at least <paramref name="minBytes"/>.
    let count (r: BinaryReader) (value: uint64) (minBytes: int64) (what: string) : int =
        if value > uint64 Int32.MaxValue || int64 value > remaining r / minBytes then
            invalid $"GGUF %s{what} count %d{value} exceeds what the file can hold"
        int value

    let readString (r: BinaryReader) =
        let len = r.ReadUInt64()
        if len > uint64 (remaining r) then invalid $"GGUF string length %d{len} exceeds the file"
        let bytes = r.ReadBytes(int len)
        if bytes.Length <> int len then raise (EndOfStreamException "Truncated GGUF string")
        Encoding.UTF8.GetString bytes

    let readScalarAsInt (r: BinaryReader) (t: uint32) : int64 =
        match t with
        | TUInt8 -> int64 (r.ReadByte())
        | TInt8 -> int64 (r.ReadSByte())
        | TUInt16 -> int64 (r.ReadUInt16())
        | TInt16 -> int64 (r.ReadInt16())
        | TUInt32 -> int64 (r.ReadUInt32())
        | TInt32 -> int64 (r.ReadInt32())
        | TUInt64 -> int64 (r.ReadUInt64())
        | TInt64 -> r.ReadInt64()
        | _ -> invalid $"GGUF type %d{t} is not an integer"

    let isIntType t =
        t = TUInt8 || t = TInt8 || t = TUInt16 || t = TInt16 || t = TUInt32 || t = TInt32 || t = TUInt64 || t = TInt64

    /// Encoded size of one array element (strings: their 8-byte length prefix at least).
    let elementBytes (t: uint32) : int64 =
        match t with
        | TUInt8 | TInt8 | TBool -> 1L
        | TUInt16 | TInt16 -> 2L
        | TUInt32 | TInt32 | TFloat32 -> 4L
        | TUInt64 | TInt64 | TFloat64 | TString -> 8L
        | other -> invalid $"Unsupported GGUF array element type %d{other}"

    let readValue (r: BinaryReader) (t: uint32) : GgufValue =
        match t with
        | TUInt64 -> GUInt(r.ReadUInt64())
        | t when isIntType t -> GInt(readScalarAsInt r t)
        | TFloat32 -> GFloat(float (r.ReadSingle()))
        | TFloat64 -> GFloat(r.ReadDouble())
        | TBool -> GBool(r.ReadByte() <> 0uy)
        | TString -> GString(readString r)
        | TArray ->
            let elemType = r.ReadUInt32()
            let n = count r (r.ReadUInt64()) (elementBytes elemType) "array"
            match elemType with
            | TString -> GStringArray(Array.init n (fun _ -> readString r))
            | TFloat32 -> GFloatArray(Array.init n (fun _ -> float (r.ReadSingle())))
            | TFloat64 -> GFloatArray(Array.init n (fun _ -> r.ReadDouble()))
            | TBool -> GBoolArray(Array.init n (fun _ -> r.ReadByte() <> 0uy))
            | et when isIntType et -> GIntArray(Array.init n (fun _ -> readScalarAsInt r et))
            | et -> invalid $"Unsupported GGUF array element type %d{et}"
        | _ -> invalid $"Unsupported GGUF value type %d{t}"

/// Location of one tensor inside the file.
type GgufTensorInfo =
    { Name: string
      Type: GgmlType
      /// ggml order: Shape[0] is contiguous.
      Shape: int64[]
      /// Absolute byte offset in the file.
      FileOffset: int64
      SizeInBytes: int64 }

    member t.ElementCount = t.Shape |> Array.fold (*) 1L

/// Parsed GGUF header (metadata + tensor directory). Tensor payloads are read on demand.
[<Sealed>]
type GgufFile private (path: string, version: uint32, metadata: IReadOnlyDictionary<string, GgufValue>, tensors: GgufTensorInfo[]) =
    let byName = tensors |> Seq.map (fun t -> KeyValuePair(t.Name, t)) |> Dictionary

    /// Parses and validates the header of a GGUF v2/v3 file.
    static member Open(path: string) : GgufFile =
        use stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 <<< 20)
        use r = new BinaryReader(stream, Encoding.UTF8, leaveOpen = false)
        try
            if stream.Length < 24L || r.ReadUInt32() <> GgufWire.magic then GgufWire.invalid $"'%s{path}' is not a GGUF file"
            let version = r.ReadUInt32()
            if version < 2u || version > 3u then GgufWire.invalid $"Unsupported GGUF version %d{version}"
            let tensorCountRaw = r.ReadUInt64()
            let kvCountRaw = r.ReadUInt64()
            // A key/value pair needs ≥ 12 bytes (key length + type), a tensor entry ≥ 24.
            let kvCount = GgufWire.count r kvCountRaw 12L "metadata"
            let metadata = Dictionary<string, GgufValue>(kvCount)
            for _ in 1 .. kvCount do
                let key = GgufWire.readString r
                let t = r.ReadUInt32()
                metadata.[key] <- GgufWire.readValue r t
            let tensorCount = GgufWire.count r tensorCountRaw 24L "tensor"
            let raw =
                Array.init tensorCount (fun _ ->
                    let name = GgufWire.readString r
                    let nDims = r.ReadUInt32()
                    if nDims = 0u || nDims > uint32 GgufWire.MaxDims then
                        GgufWire.invalid $"Tensor '%s{name}' has %d{nDims} dimensions (1..%d{GgufWire.MaxDims} supported)"
                    let shape =
                        Array.init (int nDims) (fun _ ->
                            let d = r.ReadUInt64()
                            if d = 0UL || d > uint64 Int64.MaxValue then GgufWire.invalid $"Tensor '%s{name}' has dimension %d{d}"
                            int64 d)
                    let dtype = enum<GgmlType> (int (r.ReadUInt32()))
                    if not (GgmlTypes.isKnown dtype) then GgufWire.invalid $"Tensor '%s{name}' uses unknown ggml type %d{int dtype}"
                    let offset = r.ReadUInt64()
                    struct (name, shape, dtype, offset))
            let alignment =
                match metadata.TryGetValue "general.alignment" with
                | true, GInt a -> a
                | true, GUInt a when a <= uint64 Int32.MaxValue -> int64 a
                | true, other -> GgufWire.invalid $"Invalid general.alignment %A{other}"
                | _ -> 32L
            if alignment <= 0L || (alignment &&& (alignment - 1L)) <> 0L then
                GgufWire.invalid $"general.alignment must be a positive power of two, got %d{alignment}"
            let dataStart = Numerics.alignUp stream.Position alignment
            let fileLength = stream.Length
            let tensors =
                raw
                |> Array.map (fun struct (name, shape, dtype, offset) ->
                    let elements =
                        try shape |> Array.fold (fun acc d -> Checked.(*) acc d) 1L
                        with :? OverflowException -> GgufWire.invalid $"Tensor '%s{name}' element count overflows"
                    let bytes =
                        try GgmlTypes.rowBytes dtype elements
                        with :? ArgumentException as e -> GgufWire.invalid $"Tensor '%s{name}': %s{e.Message}"
                    if offset % uint64 alignment <> 0UL then GgufWire.invalid $"Tensor '%s{name}' offset %d{offset} is not aligned to %d{alignment}"
                    if offset > uint64 (fileLength - dataStart) || int64 offset + bytes > fileLength - dataStart then
                        GgufWire.invalid $"Tensor '%s{name}' (%d{bytes} bytes at %d{offset}) extends past the end of the file"
                    { Name = name
                      Type = dtype
                      Shape = shape
                      FileOffset = dataStart + int64 offset
                      SizeInBytes = bytes })
            GgufFile(path, version, metadata, tensors)
        with :? EndOfStreamException as e ->
            raise (InvalidDataException($"'%s{path}': truncated GGUF header", e))

    member _.Path = path
    member _.Version = version
    member _.Metadata = metadata
    member _.Tensors = tensors

    member _.TryGet(key: string) =
        match metadata.TryGetValue key with
        | true, v -> Some v
        | _ -> None

    member this.TryGetInt(key: string) =
        match this.TryGet key with
        | Some(GInt v) -> Some v
        | Some(GUInt v) when v <= uint64 Int64.MaxValue -> Some(int64 v)
        | _ -> None

    member this.TryGetFloat(key: string) =
        match this.TryGet key with
        | Some(GFloat v) -> Some v
        | Some(GInt v) -> Some(float v)
        | _ -> None

    member this.TryGetString(key: string) =
        match this.TryGet key with
        | Some(GString s) -> Some s
        | _ -> None

    member this.TryGetBool(key: string) =
        match this.TryGet key with
        | Some(GBool b) -> Some b
        | _ -> None

    member this.GetInt(key: string) =
        this.TryGetInt key |> Option.defaultWith (fun () -> raise (KeyNotFoundException $"GGUF key '%s{key}' missing"))

    member this.GetString(key: string) =
        this.TryGetString key |> Option.defaultWith (fun () -> raise (KeyNotFoundException $"GGUF key '%s{key}' missing"))

    member _.TryFindTensor(name: string) =
        match byName.TryGetValue name with
        | true, t -> Some t
        | _ -> None

    static member private Buffer(info: GgufTensorInfo) =
        if info.SizeInBytes > int64 Array.MaxLength then
            raise (NotSupportedException $"Tensor '%s{info.Name}' (%d{info.SizeInBytes} bytes) exceeds the 2 GiB managed array limit of host tensors")
        GC.AllocateUninitializedArray<byte>(int info.SizeInBytes)

    /// Reads one tensor payload into a freshly allocated host tensor (positional I/O, thread-safe).
    member _.ReadTensor(handle: SafeFileHandle, info: GgufTensorInfo) : HostTensor =
        let buffer = GgufFile.Buffer info
        let mutable read = 0
        while read < buffer.Length do
            let n = RandomAccess.Read(handle, Span<byte>(buffer, read, buffer.Length - read), info.FileOffset + int64 read)
            if n <= 0 then raise (EndOfStreamException $"Unexpected end of file while reading '%s{info.Name}'")
            read <- read + n
        HostTensor(info.Name, info.Type, info.Shape, buffer)

    /// Asynchronous positional read of one tensor payload (concurrent reads on one handle are
    /// safe). Continuations do not capture the caller's synchronization context.
    member _.ReadTensorAsync(handle: SafeFileHandle, info: GgufTensorInfo, cancellation: Threading.CancellationToken) : Threading.Tasks.Task<HostTensor> =
        backgroundTask {
            let buffer = GgufFile.Buffer info
            let mutable read = 0
            while read < buffer.Length do
                let! n = RandomAccess.ReadAsync(handle, Memory<byte>(buffer, read, buffer.Length - read), info.FileOffset + int64 read, cancellation)
                if n <= 0 then raise (EndOfStreamException $"Unexpected end of file while reading '%s{info.Name}'")
                read <- read + n
            return HostTensor(info.Name, info.Type, info.Shape, buffer)
        }
