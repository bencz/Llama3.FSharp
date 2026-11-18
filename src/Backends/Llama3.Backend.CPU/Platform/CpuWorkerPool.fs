namespace Llama3.Backend.Cpu

open System
open System.Diagnostics
open System.Threading

/// A unit of parallel work split into independent, indexed chunks.
type IChunkJob =
    abstract Execute: chunk: int -> unit

/// Persistent fork-join pool for the per-token kernels (hundreds of jobs per token, each a few
/// microseconds to a few milliseconds). Compared with Parallel.For it has no task allocation or
/// thread-pool wake-up per job: <c>threads − 1</c> dedicated workers plus the calling thread pull
/// chunk indices from one atomic counter (dynamic load balancing, so a preempted worker delays at
/// most one chunk). Idle workers spin briefly, then sleep on a monitor until the next job.
///
/// Protocol (safe on weakly ordered CPUs such as ARM64): <c>generation</c> is odd while a job is
/// open and even once it is closed. The job fields are written before the odd generation is
/// published (release) and read only after observing it (acquire). A worker joins a job by
/// incrementing <c>active</c> and then re-checking that the same odd generation is still current;
/// the caller closes the job (even generation) and then waits for <c>active</c> = 0. Both sides
/// use full fences between their write and their read (Dekker), so either the worker sees the
/// job closed and backs off, or the caller sees the worker and waits for it. A straggler can
/// therefore never claim chunks of the next job with the previous job's delegate.
[<Sealed>]
type CpuWorkerPool(threads: int) =
    let workerCount = max 0 (threads - 1)
    let gate = obj ()
    /// Spin this long for the next job before sleeping (jobs of one token arrive back to back).
    let spinTicks = Stopwatch.Frequency / 20_000L // 50 µs
    let mutable job = ValueNone: IChunkJob voption
    let mutable chunkCount = 0
    let mutable nextChunk = 0
    let mutable pending = 0
    let mutable active = 0
    let mutable generation = 0L
    let mutable stopping = false
    let mutable running = 0
    let mutable disposed = 0
    let mutable failure = ValueNone: exn voption

    let runChunks (work: IChunkJob) (count: int) =
        let mutable chunk = Interlocked.Increment(&nextChunk) - 1
        while chunk < count do
            try work.Execute chunk
            with e -> lock gate (fun () -> if failure.IsNone then failure <- ValueSome e)
            Interlocked.Decrement(&pending) |> ignore
            chunk <- Interlocked.Increment(&nextChunk) - 1

    /// True when an open job other than <paramref name="seen"/> is published.
    let hasNewJob (seen: int64) =
        let g = Volatile.Read(&generation)
        g <> seen && (g &&& 1L) = 1L

    let workerLoop () =
        let mutable seen = 0L
        while not (Volatile.Read(&stopping)) do
            // Short spin: the next job of the same token usually follows within microseconds.
            let start = Stopwatch.GetTimestamp()
            while not (hasNewJob seen) && not (Volatile.Read(&stopping)) && Stopwatch.GetTimestamp() - start < spinTicks do
                Thread.SpinWait 64
            if not (hasNewJob seen) && not (Volatile.Read(&stopping)) then
                lock gate (fun () ->
                    while not (hasNewJob seen) && not stopping do
                        Monitor.Wait gate |> ignore)
            let g = Volatile.Read(&generation)
            if not (Volatile.Read(&stopping)) && g <> seen && (g &&& 1L) = 1L then
                seen <- g
                Interlocked.Increment(&active) |> ignore // full fence
                // Still the same open job? Then the caller cannot close it without seeing us.
                if Volatile.Read(&generation) = g then
                    match job with
                    | ValueSome work -> runChunks work chunkCount
                    | ValueNone -> ()
                Interlocked.Decrement(&active) |> ignore

    let workers =
        Array.init workerCount (fun i ->
            let t = Thread(workerLoop, IsBackground = true, Name = $"llama3-cpu-%d{i}")
            t.Start()
            t)

    member _.Threads = workerCount + 1

    /// Runs chunks 0 .. count−1 of <paramref name="work"/> on the pool and the calling thread;
    /// returns when every chunk has finished. The first exception thrown by a chunk is rethrown.
    /// One job at a time: concurrent callers are rejected.
    member _.Run(work: IChunkJob, count: int) =
        if count <= 0 then ()
        elif workerCount = 0 || count = 1 then
            for c in 0 .. count - 1 do work.Execute c
        else
            if Interlocked.Exchange(&running, 1) = 1 then invalidOp "CpuWorkerPool.Run is not reentrant"
            try
                job <- ValueSome work
                chunkCount <- count
                pending <- count
                failure <- ValueNone
                nextChunk <- 0
                lock gate (fun () ->
                    // Odd generation = open job; the volatile write releases the fields above.
                    Volatile.Write(&generation, generation + 1L)
                    Monitor.PulseAll gate)
                runChunks work count
                let mutable spin = SpinWait()
                while Volatile.Read(&pending) > 0 do
                    spin.SpinOnce(-1)
                // Close (even generation, full fence), then wait for workers that joined.
                Interlocked.Increment(&generation) |> ignore
                spin.Reset()
                while Volatile.Read(&active) > 0 do
                    spin.SpinOnce(-1)
                job <- ValueNone
                match failure with
                | ValueSome e -> raise (AggregateException("A parallel CPU kernel failed", e))
                | ValueNone -> ()
            finally
                Volatile.Write(&running, 0)

    interface IDisposable with
        member _.Dispose() =
            if Interlocked.Exchange(&disposed, 1) = 0 then
                lock gate (fun () ->
                    Volatile.Write(&stopping, true)
                    Monitor.PulseAll gate)
                for w in workers do w.Join()
