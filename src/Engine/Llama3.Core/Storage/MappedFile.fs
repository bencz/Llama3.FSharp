namespace Llama3.Core

open System
open System.IO
open System.IO.MemoryMappedFiles
open System.Runtime.CompilerServices
open System.Runtime.InteropServices

/// OS hints for mapped memory (llama.cpp's llama_mmap prefetch). Interop only; no pointer types.
module internal MemoryAdvice =

    [<Literal>]
    let private PosixMadvWillNeed = 3

    [<DllImport("libc", EntryPoint = "posix_madvise", SetLastError = false)>]
    extern int private posixMadvise(nativeint address, unativeint length, int advice)

    /// MADV_POPULATE_READ (Linux 5.14+): synchronously fault in the page-table entries of a range.
    [<Literal>]
    let private MadvPopulateRead = 22

    [<DllImport("libc", EntryPoint = "madvise", SetLastError = false)>]
    extern int private madvise(nativeint address, unativeint length, int advice)

    /// Populates [address, address + length) (page-aligned address). False when the kernel or the
    /// platform does not support it; the caller then touches the pages itself.
    let populateRead (address: nativeint) (length: int64) =
        OperatingSystem.IsLinux()
        && length > 0L
        && (try madvise (address, unativeint length, MadvPopulateRead) = 0
            with :? DllNotFoundException | :? EntryPointNotFoundException -> false)

    /// Asks the kernel to start reading [address, address + length) into the page cache
    /// asynchronously. Best effort: failures and unsupported platforms are ignored.
    let willNeed (address: nativeint) (length: int64) =
        if OperatingSystem.IsLinux() && length > 0L then
            try posixMadvise (address, unativeint length, PosixMadvWillNeed) |> ignore
            with :? DllNotFoundException | :? EntryPointNotFoundException -> ()

/// A whole file mapped read-only into the address space — the llama.cpp loading model: tensors
/// are views into the mapping (zero copy, no 2 GiB limit, pages shared with the OS page cache),
/// and device uploads read straight from it.
[<Sealed>]
type MappedFile(path: string) =
    let length = FileInfo(path).Length
    let file = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0L, MemoryMappedFileAccess.Read)
    let view = file.CreateViewAccessor(0L, 0L, MemoryMappedFileAccess.Read)
    let handle = view.SafeMemoryMappedViewHandle
    let mutable referenced = false
    do handle.DangerousAddRef(&referenced)
    let baseAddress = handle.DangerousGetHandle() + nativeint view.PointerOffset
    let mutable disposed = false

    member _.Path = path
    member _.Length = length

    /// Read-only window of the mapping. The span is valid while this object is alive.
    member _.Slice(offset: int64, count: int) : ReadOnlySpan<byte> =
        if disposed then raise (ObjectDisposedException path)
        if offset < 0L || count < 0 || offset + int64 count > length then
            raise (ArgumentOutOfRangeException(nameof offset, $"[%d{offset}, +%d{count}) outside '%s{path}' (%d{length} bytes)"))
        // Byref arithmetic from the mapping's base address: no pointer types, no unverifiable IL.
        let start = &Unsafe.AddByteOffset(&Unsafe.NullRef<byte>(), unativeint (baseAddress + nativeint offset))
        MemoryMarshal.CreateReadOnlySpan(&start, count)

    /// Starts asynchronous readahead of a byte range (posix_madvise WILLNEED on Linux).
    member _.Prefetch(offset: int64, count: int64) =
        if not disposed then MemoryAdvice.willNeed (baseAddress + nativeint offset) (min count (length - offset))

    /// Faults the whole mapping in before inference starts — llama.cpp's MAP_POPULATE default:
    /// the file is read into the page cache (in parallel chunks, which keeps NVMe queues busy) and
    /// this process's page tables are filled, so decoding never stalls on page faults. Pages stay
    /// shared with the page cache, so a second process or run pays almost nothing.
    member this.Populate(parallelism: int) =
        if not disposed && length > 0L then
            let chunk = 64L <<< 20 // multiple of every page size; mapping base is page-aligned
            let chunks = int ((length + chunk - 1L) / chunk)
            let page = int64 Environment.SystemPageSize
            Threading.Tasks.Parallel.For(
                0,
                chunks,
                Threading.Tasks.ParallelOptions(MaxDegreeOfParallelism = max 1 parallelism),
                fun i ->
                    let offset = int64 i * chunk
                    let count = min chunk (length - offset)
                    if not (MemoryAdvice.populateRead (baseAddress + nativeint offset) count) then
                        // Older kernels / other systems: touch one byte per page.
                        let mutable sink = 0uy
                        let mutable p = offset
                        while p < offset + count do
                            sink <- sink ^^^ (this.Slice(p, 1)).[0]
                            p <- p + page
                        GC.KeepAlive sink)
            |> ignore

    interface IDisposable with
        member _.Dispose() =
            if not disposed then
                disposed <- true
                if referenced then handle.DangerousRelease()
                view.Dispose()
                file.Dispose()

/// Backing bytes of a host tensor: a managed array or a window of a mapped file. Offsets are
/// 64-bit so a tensor may exceed 2 GiB; any single span is at most Int32.MaxValue bytes.
[<AbstractClass>]
type TensorStorage() =
    abstract Length: int64
    abstract Slice: offset: int64 * count: int -> ReadOnlySpan<byte>

[<Sealed>]
type ArrayStorage(data: byte[]) =
    inherit TensorStorage()
    member _.Array = data
    override _.Length = int64 data.Length
    override _.Slice(offset, count) = ReadOnlySpan<byte>(data, int offset, count)

[<Sealed>]
type MappedStorage(file: MappedFile, offset: int64, length: int64) =
    inherit TensorStorage()
    do
        if offset < 0L || length < 0L || offset + length > file.Length then
            invalidArg (nameof length) $"Tensor range [%d{offset}, +%d{length}) outside '%s{file.Path}'"
    member _.File = file
    member _.Offset = offset
    override _.Length = length
    override _.Slice(start, count) =
        if start < 0L || start + int64 count > length then
            raise (ArgumentOutOfRangeException(nameof start, $"[%d{start}, +%d{count}) outside a %d{length}-byte tensor"))
        file.Slice(offset + start, count)
