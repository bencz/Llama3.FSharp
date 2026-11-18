namespace Llama3.Core

open System

/// Top-k expert selection (Qwen3-MoE / Mixtral semantics).
module MoeRouting =

    /// Selects the <c>experts.Length</c> highest-scoring experts and their gate weights:
    /// probabilities are the softmax over <b>all</b> router logits; with <paramref name="normalize"/>
    /// the selected probabilities are rescaled to sum to 1 (<c>norm_topk_prob</c>).
    /// Ties resolve to the lower expert index. Allocation-free; O(n*k) with k << n.
    let select (logits: ReadOnlySpan<float32>) (normalize: bool) (experts: Span<int>) (weights: Span<float32>) =
        let k = experts.Length
        if k = 0 || k > logits.Length || weights.Length <> k then
            invalidArg (nameof experts) $"Cannot route to %d{k} of %d{logits.Length} experts"
        // Insertion into a descending top-k list kept in (experts, weights=logits for now).
        for i in 0 .. k - 1 do
            experts.[i] <- -1
            weights.[i] <- Single.NegativeInfinity
        for e in 0 .. logits.Length - 1 do
            // NaN logits rank below every number (as -inf); empty slots (-1) below everything.
            let l = if Single.IsNaN logits.[e] then Single.NegativeInfinity else logits.[e]
            if experts.[k - 1] < 0 || l > weights.[k - 1] then
                let mutable j = k - 1
                while j > 0 && (experts.[j - 1] < 0 || l > weights.[j - 1]) do
                    weights.[j] <- weights.[j - 1]
                    experts.[j] <- experts.[j - 1]
                    j <- j - 1
                weights.[j] <- l
                experts.[j] <- e
        // Softmax over the full router distribution (max is the top-1 logit).
        let maxLogit = weights.[0]
        if Single.IsNegativeInfinity maxLogit then
            // No usable logit at all: route uniformly instead of producing NaN weights.
            for i in 0 .. k - 1 do weights.[i] <- 1.0f / float32 k
        else
            let mutable total = 0.0
            for e in 0 .. logits.Length - 1 do
                if not (Single.IsNaN logits.[e]) then total <- total + exp (float (logits.[e] - maxLogit))
            let mutable selected = 0.0
            for i in 0 .. k - 1 do
                let p = exp (float (weights.[i] - maxLogit)) / total
                weights.[i] <- float32 p
                selected <- selected + p
            if normalize then
                for i in 0 .. k - 1 do
                    weights.[i] <- float32 (float weights.[i] / selected)
