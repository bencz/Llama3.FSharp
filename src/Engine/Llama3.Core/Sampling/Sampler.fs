namespace Llama3.Core

open System
open System.Collections.Generic

type SamplingSettings =
    { /// 0 = greedy argmax (the only strategy of the original llama3.cu).
      Temperature: float32
      /// Keep the K most likely tokens (0 = disabled).
      TopK: int
      /// Nucleus threshold (1 = disabled).
      TopP: float32
      /// Drop tokens whose probability is below MinP x p(max) (0 = disabled).
      MinP: float32
      /// CTRL-style penalty applied to recently generated tokens (1 = disabled).
      RepetitionPenalty: float32
      RepetitionWindow: int
      Seed: int option }

    static member Greedy =
        { Temperature = 0.0f
          TopK = 0
          TopP = 1.0f
          MinP = 0.0f
          RepetitionPenalty = 1.0f
          RepetitionWindow = 64
          Seed = None }

    /// Qwen3 recommended settings for non-thinking mode.
    static member Chat =
        { SamplingSettings.Greedy with
            Temperature = 0.7f
            TopK = 20
            TopP = 0.8f }

/// Stateful sampler (owns the RNG and scratch buffers; not thread-safe). Steady state allocates
/// nothing per token: candidate buffers are reused and sorted by key arrays (no comparer closure).
[<Sealed>]
type Sampler(settings: SamplingSettings) =
    let rng = match settings.Seed with Some s -> Random(s) | None -> Random()
    let seen = HashSet<int>()
    let heap = PriorityQueue<int, float32>()
    let mutable keys = Array.empty<float32>
    let mutable ids = Array.empty<int>
    let mutable probs = Array.empty<float>

    let ensure (n: int) =
        if keys.Length < n then
            keys <- Array.zeroCreate n
            ids <- Array.zeroCreate n
            probs <- Array.zeroCreate n

    /// Index of the largest value; ties resolve to the lowest index (same as sample_argmax in C).
    static member Argmax(logits: ReadOnlySpan<float32>) =
        let mutable best = 0
        let mutable bestValue = logits.[0]
        for i in 1 .. logits.Length - 1 do
            if logits.[i] > bestValue then
                best <- i
                bestValue <- logits.[i]
        best

    member _.Settings = settings

    /// Samples the next token. <paramref name="logits"/> may be modified in place (penalties).
    member _.Sample(logits: float32[], history: IReadOnlyList<int>) : int =
        if settings.RepetitionPenalty <> 1.0f && history.Count > 0 then
            seen.Clear()
            let start = max 0 (history.Count - settings.RepetitionWindow)
            for i in start .. history.Count - 1 do
                let t = history.[i]
                if seen.Add t && t >= 0 && t < logits.Length then
                    let l = logits.[t]
                    logits.[t] <- if l > 0.0f then l / settings.RepetitionPenalty else l * settings.RepetitionPenalty

        if settings.Temperature <= 0.0f then Sampler.Argmax(ReadOnlySpan logits)
        else
            let n = logits.Length
            let invT = 1.0f / settings.Temperature
            let mutable maxLogit = Single.NegativeInfinity
            for l in logits do
                if l > maxLogit then maxLogit <- l
            let truncates = (settings.TopK > 0 && settings.TopK < n) || settings.TopP < 1.0f || settings.MinP > 0.0f
            if not truncates then
                // Plain temperature sampling: one pass for Z, one cumulative pass to draw.
                let mutable z = 0.0
                for l in logits do z <- z + exp (float ((l - maxLogit) * invT))
                let r = rng.NextDouble() * z
                let mutable acc = 0.0
                let mutable chosen = n - 1
                let mutable i = 0
                while i < n do
                    acc <- acc + exp (float ((logits.[i] - maxLogit) * invT))
                    if r < acc then
                        chosen <- i
                        i <- n
                    else i <- i + 1
                chosen
            else
                ensure (if settings.TopK > 0 && settings.TopK < n then settings.TopK else n)
                let mutable count = 0
                if settings.TopK > 0 && settings.TopK < n then
                    // Bounded min-heap: O(n log k) instead of sorting the vocabulary.
                    heap.Clear()
                    for i in 0 .. n - 1 do
                        if heap.Count < settings.TopK then heap.Enqueue(i, logits.[i])
                        else
                            let mutable smallest = 0
                            let mutable smallestLogit = 0.0f
                            if heap.TryPeek(&smallest, &smallestLogit) && logits.[i] > smallestLogit then
                                heap.DequeueEnqueue(i, logits.[i]) |> ignore
                    while heap.Count > 0 do
                        let i = heap.Dequeue()
                        ids.[count] <- i
                        keys.[count] <- -logits.[i]
                        count <- count + 1
                else
                    // Pre-filter tokens that cannot reach the nucleus (llama2.c trick).
                    let mutable z = 0.0
                    for l in logits do z <- z + exp (float ((l - maxLogit) * invT))
                    let cutoff = if settings.TopP < 1.0f then (1.0 - float settings.TopP) / float (n - 1) else 0.0
                    for i in 0 .. n - 1 do
                        if exp (float ((logits.[i] - maxLogit) * invT)) / z >= cutoff then
                            ids.[count] <- i
                            keys.[count] <- -logits.[i]
                            count <- count + 1
                // Descending logits (ascending negated keys); ties keep the lower id first.
                Array.Sort(keys, ids, 0, count)
                let mutable sum = 0.0
                for c in 0 .. count - 1 do
                    let p = exp (float ((-keys.[c] - maxLogit) * invT))
                    probs.[c] <- p
                    sum <- sum + p
                let mutable keep = count
                if settings.MinP > 0.0f then
                    let threshold = float settings.MinP * probs.[0]
                    let mutable k = 1
                    while k < keep && probs.[k] >= threshold do k <- k + 1
                    keep <- k
                if settings.TopP < 1.0f then
                    let mutable cumulative = 0.0
                    let mutable k = 0
                    let target = float settings.TopP * sum
                    while k < keep && cumulative < target do
                        cumulative <- cumulative + probs.[k]
                        k <- k + 1
                    keep <- max 1 k
                let mutable total = 0.0
                for k in 0 .. keep - 1 do total <- total + probs.[k]
                let r = rng.NextDouble() * total
                let mutable acc = 0.0
                let mutable chosen = keep - 1
                let mutable k = 0
                while k < keep do
                    acc <- acc + probs.[k]
                    if r < acc then
                        chosen <- k
                        k <- keep
                    else k <- k + 1
                ids.[chosen]
