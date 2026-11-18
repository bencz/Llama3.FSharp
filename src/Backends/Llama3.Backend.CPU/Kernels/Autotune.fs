namespace Llama3.Backend.Cpu

open System
open System.Collections.Concurrent
open System.Diagnostics
open Llama3.Core

/// Routes every operation to the implementation selected for it on this machine.
[<Sealed>]
type TunedKernels(f32: ICpuKernels, q8_0: ICpuKernels, q4_K: ICpuKernels, q6_K: ICpuKernels) =
    let name (k: ICpuKernels) = sprintf "%A" k.Isa

    interface ICpuKernels with
        /// A tuned set mixes ISAs; this reports the one serving the Q4_K path (the most common format).
        member _.Isa = q4_K.Isa
        member _.Description =
            $"autotuned (F32=%s{name f32}, Q8_0=%s{name q8_0}, Q4_K=%s{name q4_K}, Q6_K=%s{name q6_K}; %s{CpuFeatures.describe ()})"
        member _.DotF32(a, b) = f32.DotF32(a, b)
        member _.Axpy(alpha, x, y) = f32.Axpy(alpha, x, y)
        member _.DotQ8_0(row, blocks, xq) = q8_0.DotQ8_0(row, blocks, xq)
        member _.DotQ4_K(row, superBlocks, xq) = q4_K.DotQ4_K(row, superBlocks, xq)
        member _.DotQ6_K(row, superBlocks, xq) = q6_K.DotQ6_K(row, superBlocks, xq)

/// Per-machine kernel selection by measurement. Instruction-set availability does not imply speed
/// (512-bit VNNI loses to AVX2 on some client cores and wins on server cores), so each candidate
/// runs the same work under decode-like conditions: a weight set larger than typical last-level
/// caches, swept by the real number of threads. The cost is the best of a few runs; candidates
/// within <c>Margin</c> of the fastest resolve by a fixed preference (widest vectors first), so
/// the choice is stable across runs instead of flipping on noise.
module Autotune =

    [<Literal>]
    let private Columns = 4096

    /// Bytes of synthetic weights per format (well above a client L3, streamed like real decoding).
    [<Literal>]
    let private WorkingSetBytes = 32 * 1024 * 1024

    [<Literal>]
    let private RowsPerChunk = 64

    let private Margin = 1.10

    let private preference (isa: CpuIsa) =
        match isa with
        | X86Avx512Vnni -> 0
        | X86Avx2 -> 1
        | ArmNeon -> 2
        | Portable -> 3

    /// Deterministic random blocks with sane scale fields (content is irrelevant; every candidate
    /// does identical work).
    let private synthetic (dtype: GgmlType) =
        let rowBytes = int (GgmlTypes.rowBytes dtype (int64 Columns))
        let rows = WorkingSetBytes / rowBytes
        let bytes = GC.AllocateUninitializedArray<byte>(rows * rowBytes)
        Random(1234).NextBytes bytes
        let setHalf (o: int) (v: float32) =
            let h = Numerics.singleToHalfBits v
            bytes.[o] <- byte h
            bytes.[o + 1] <- byte (h >>> 8)
        match dtype with
        | GgmlType.Q8_0 -> for b in 0 .. bytes.Length / 34 - 1 do setHalf (b * 34) 0.01f
        | GgmlType.Q4_K ->
            for b in 0 .. bytes.Length / 144 - 1 do
                setHalf (b * 144) 0.001f
                setHalf (b * 144 + 2) 0.001f
        | GgmlType.Q6_K -> for b in 0 .. bytes.Length / 210 - 1 do setHalf (b * 210 + 208) 0.001f
        | _ ->
            // F32: replace the random bit patterns by finite values.
            let floats = Runtime.InteropServices.MemoryMarshal.Cast<byte, float32>(Span<byte>(bytes))
            for i in 0 .. floats.Length - 1 do floats.[i] <- float32 (i % 7) * 0.1f
        bytes, rows, rowBytes

    /// Sweeps every row of a synthetic matrix with one candidate on the pool.
    [<Sealed>]
    type private Sweep(data: byte[], rows: int, rowBytes: int, dtype: GgmlType, x: float32[], xq: QuantizedActivations) =
        let sinks = Array.zeroCreate<float32> (Numerics.divUp rows RowsPerChunk)
        member val Kernels = Unchecked.defaultof<ICpuKernels> with get, set
        member _.Chunks = sinks.Length
        member _.Sink = Array.sum sinks
        interface IChunkJob with
            member this.Execute chunk =
                let mutable s = 0.0f
                let k = this.Kernels
                for r in chunk * RowsPerChunk .. min rows ((chunk + 1) * RowsPerChunk) - 1 do
                    let row = ReadOnlySpan<byte>(data, r * rowBytes, rowBytes)
                    s <-
                        s
                        + match dtype with
                          | GgmlType.Q8_0 -> k.DotQ8_0(row, Columns / 32, xq)
                          | GgmlType.Q4_K -> k.DotQ4_K(row, Columns / 256, xq)
                          | GgmlType.Q6_K -> k.DotQ6_K(row, Columns / 256, xq)
                          | _ -> k.DotF32(Runtime.InteropServices.MemoryMarshal.Cast<byte, float32> row, ReadOnlySpan x)
                sinks.[chunk] <- s

    let private tune (threads: int) : ICpuKernels =
        let all = CpuKernels.available ()
        // The portable path is only a contender when no vector ISA exists.
        let candidates =
            match all |> List.filter (fun k -> k.Isa <> Portable) with
            | [] -> all
            | vector -> vector
        match candidates with
        | [ only ] -> only
        | _ ->
            use pool = new CpuWorkerPool(threads)
            let rng = Random 99
            let x = Array.init Columns (fun _ -> float32 (rng.NextDouble() * 2.0 - 1.0))
            let xq = QuantizedActivations(Columns)
            xq.Quantize(ReadOnlySpan x)
            let select (dtype: GgmlType) =
                let data, rows, rowBytes = synthetic dtype
                let sweep = Sweep(data, rows, rowBytes, dtype, x, xq)
                let cost (k: ICpuKernels) =
                    sweep.Kernels <- k
                    pool.Run(sweep, sweep.Chunks) // warm-up: pages, caches, JIT
                    let mutable best = Int64.MaxValue
                    for _ in 1 .. 3 do
                        let start = Stopwatch.GetTimestamp()
                        pool.Run(sweep, sweep.Chunks)
                        best <- min best (Stopwatch.GetTimestamp() - start)
                    if Single.IsNaN sweep.Sink then best + 1L else best
                let timed = candidates |> List.map (fun k -> k, float (cost k))
                let fastest = timed |> List.minBy snd |> snd
                timed
                |> List.filter (fun (_, t) -> t <= fastest * Margin)
                |> List.minBy (fun (k, _) -> preference k.Isa)
                |> fst
            TunedKernels(select GgmlType.F32, select GgmlType.Q8_0, select GgmlType.Q4_K, select GgmlType.Q6_K) :> ICpuKernels

    let private cache = ConcurrentDictionary<int, Lazy<ICpuKernels>>()

    /// Kernels tuned for <paramref name="threads"/> workers (measured once per process and count).
    let best (threads: int) : ICpuKernels = cache.GetOrAdd(threads, fun t -> lazy (tune t)).Value
