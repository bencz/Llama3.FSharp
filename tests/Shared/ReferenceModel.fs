/// Synthetic models and an independent float64 reference forward pass (linked into test projects).
/// The reference is written directly from the model definition (pre-norm decoder, GQA, RoPE,
/// QK-norm, SwiGLU, softmax->top-k->renormalised MoE) and shares no code with the runtime except
/// weight decoding (Dequantize), which has its own exactness tests.
module Llama3.Tests.ReferenceModel

open System
open Llama3.Core
open Llama3.Tests.TestSupport

/// Architectural options of a synthetic model (each one changes the forward pass).
type SyntheticSpec =
    { Seed: int
      DType: GgmlType
      Moe: bool
      Layers: int
      RopeMode: RopeMode
      /// Llama 3.1-style rope_freqs correction factors.
      FreqFactors: bool
      /// Qwen2-style Q/K/V projection biases.
      QkvBias: bool
      /// Qwen3 per-head RMSNorm on q and k.
      QkNorm: bool
      /// Classifier shares the (F32) token embedding table.
      TiedClassifier: bool }

    static member Create(seed: int, dtype: GgmlType, moe: bool) =
        { Seed = seed
          DType = dtype
          Moe = moe
          Layers = 3
          RopeMode = NeoX
          FreqFactors = false
          QkvBias = false
          QkNorm = true
          TiedClassifier = false }

/// Builds a random model. Dims are multiples of 256 so every quantized format is legal.
let syntheticWith (spec: SyntheticSpec) : HostModel =
    let rng = Random spec.Seed
    let dtype = spec.DType
    let dim, hidden, layers, heads, kvHeads, headDim, vocab = 256, 512, spec.Layers, 4, 2, 64, 384
    let moeConfig =
        if spec.Moe then Some { ExpertCount = 8; ExpertsPerToken = 2; ExpertHiddenDim = 256; NormalizeTopK = true }
        else None
    let config =
        { Architecture = (if spec.Moe then Qwen3Moe else Qwen3)
          Dim = dim
          HiddenDim = hidden
          NumLayers = layers
          NumHeads = heads
          NumKvHeads = kvHeads
          HeadDim = headDim
          VocabSize = vocab
          ContextLength = 256
          RopeTheta = 10000.0f
          RopeMode = spec.RopeMode
          NormEps = 1e-5f
          Moe = moeConfig }.Validate()
    let vector (n: int) (center: float) =
        HostTensor.OfSingles("v", [| int64 n |], ReadOnlySpan(Array.init n (fun _ -> float32 (center + (rng.NextDouble() - 0.5) * 0.2))))
    let matrix rows cols = randomTensor rng dtype rows cols
    /// Stacked expert tensor: [cols; rows; experts].
    let stacked (rows: int) (cols: int) (experts: int) =
        let flat = randomTensor rng dtype (rows * experts) cols
        HostTensor("experts", dtype, [| int64 cols; int64 rows; int64 experts |], flat.Storage)
    let bias n = if spec.QkvBias then Some(vector n 0.0) else None
    let layer () =
        let ffnBlock =
            match moeConfig with
            | Some m ->
                Moe
                    { Router = HostTensor.OfSingles("router", [| int64 dim; int64 m.ExpertCount |], ReadOnlySpan(randomVector rng (dim * m.ExpertCount)))
                      GateExperts = stacked m.ExpertHiddenDim dim m.ExpertCount
                      UpExperts = stacked m.ExpertHiddenDim dim m.ExpertCount
                      DownExperts = stacked dim m.ExpertHiddenDim m.ExpertCount }
            | None -> Dense { Gate = matrix hidden dim; Up = matrix hidden dim; Down = matrix dim hidden }
        { Attention =
            { Norm = vector dim 1.0
              Wq = matrix (heads * headDim) dim
              Wk = matrix (kvHeads * headDim) dim
              Wv = matrix (kvHeads * headDim) dim
              Wo = matrix dim (heads * headDim)
              Bq = bias (heads * headDim)
              Bk = bias (kvHeads * headDim)
              Bv = bias (kvHeads * headDim)
              QNorm = (if spec.QkNorm then Some(vector headDim 1.0) else None)
              KNorm = (if spec.QkNorm then Some(vector headDim 1.0) else None) }
          Ffn = { Norm = vector dim 1.0; Block = ffnBlock } }
    let embedding = HostTensor.OfSingles("embd", [| int64 dim; int64 vocab |], ReadOnlySpan(randomVector rng (dim * vocab)))
    let layerArray = Array.init layers (fun _ -> layer ())
    { Config = config
      TokenEmbedding = embedding
      Layers = layerArray
      OutputNorm = vector dim 1.0
      Output = (if spec.TiedClassifier then embedding else matrix vocab dim)
      RopeFreqFactors =
        if spec.FreqFactors then
            // Factors in [1, 8], growing with the pair index like Llama 3.1's low-frequency scaling.
            Some(HostTensor.OfSingles("rope_freqs", [| int64 (headDim / 2) |], ReadOnlySpan(Array.init (headDim / 2) (fun j -> 1.0f + 7.0f * float32 j / float32 (headDim / 2 - 1)))))
        else None }

let synthetic (seed: int) (dtype: GgmlType) (moe: bool) : HostModel = syntheticWith (SyntheticSpec.Create(seed, dtype, moe))

/// Float64 reference forward over a whole token sequence; returns logits per position.
let forward (model: HostModel) (tokens: int[]) : float[][] =
    let c = model.Config
    let decoded = Collections.Generic.Dictionary<HostTensor, float[]>(HashIdentity.Reference)
    let values (t: HostTensor) =
        match decoded.TryGetValue t with
        | true, v -> v
        | _ ->
            let v = Dequantize.toSingles t |> Array.map float
            decoded.[t] <- v
            v
    /// rows [first, first+count) of W * x
    let matvecRows (w: HostTensor) (first: int) (count: int) (x: float[]) =
        let data = values w
        let cols = w.Columns
        Array.init count (fun r ->
            let o = (first + r) * cols
            let mutable acc = 0.0
            for j in 0 .. cols - 1 do acc <- acc + data.[o + j] * x.[j]
            acc)
    let matvec (w: HostTensor) x = matvecRows w 0 w.Rows x
    let rmsNorm (x: float[]) (w: HostTensor) =
        let wv = values w
        let ms = (Array.sumBy (fun v -> v * v) x) / float x.Length
        let inv = 1.0 / sqrt (ms + float c.NormEps)
        Array.mapi (fun i v -> wv.[i] * v * inv) x
    let silu v = v / (1.0 + exp -v)
    let half = c.HeadDim / 2
    let factors = model.RopeFreqFactors |> Option.map values
    let rope (v: float[]) (heads: int) (pos: int) =
        let out = Array.copy v
        for h in 0 .. heads - 1 do
            for j in 0 .. half - 1 do
                let freq =
                    1.0 / Math.Pow(float c.RopeTheta, float (2 * j) / float c.HeadDim)
                    / (match factors with Some f -> f.[j] | None -> 1.0)
                let a = float pos * freq
                let i0, i1 = if c.RopeMode = NeoX then j, j + half else 2 * j, 2 * j + 1
                let o = h * c.HeadDim
                out.[o + i0] <- v.[o + i0] * cos a - v.[o + i1] * sin a
                out.[o + i1] <- v.[o + i0] * sin a + v.[o + i1] * cos a
        out
    let headNorm (v: float[]) (heads: int) (w: HostTensor) =
        let wv = values w
        let out = Array.copy v
        for h in 0 .. heads - 1 do
            let o = h * c.HeadDim
            let ms = (Seq.sum (seq { for i in 0 .. c.HeadDim - 1 -> v.[o + i] * v.[o + i] })) / float c.HeadDim
            let inv = 1.0 / sqrt (ms + float c.NormEps)
            for i in 0 .. c.HeadDim - 1 do out.[o + i] <- wv.[i] * v.[o + i] * inv
        out
    let keys = Array.init c.NumLayers (fun _ -> ResizeArray<float[]>())
    let vals = Array.init c.NumLayers (fun _ -> ResizeArray<float[]>())
    let embedding = values model.TokenEmbedding
    [| for pos in 0 .. tokens.Length - 1 do
           let mutable x = Array.init c.Dim (fun i -> embedding.[tokens.[pos] * c.Dim + i])
           for l in 0 .. c.NumLayers - 1 do
               let a = model.Layers.[l].Attention
               let h = rmsNorm x a.Norm
               let withBias (b: HostTensor option) (y: float[]) =
                   match b with
                   | Some t -> Array.map2 (+) y (values t)
                   | None -> y
               let mutable q = matvec a.Wq h |> withBias a.Bq
               let mutable k = matvec a.Wk h |> withBias a.Bk
               let v = matvec a.Wv h |> withBias a.Bv
               a.QNorm |> Option.iter (fun n -> q <- headNorm q c.NumHeads n)
               a.KNorm |> Option.iter (fun n -> k <- headNorm k c.NumKvHeads n)
               q <- rope q c.NumHeads pos
               k <- rope k c.NumKvHeads pos
               keys.[l].Add k
               vals.[l].Add v
               let att = Array.zeroCreate<float> c.QDim
               for head in 0 .. c.NumHeads - 1 do
                   let kv = head / c.KvMul
                   let scores =
                       Array.init (pos + 1) (fun t ->
                           let mutable s = 0.0
                           for i in 0 .. c.HeadDim - 1 do
                               s <- s + q.[head * c.HeadDim + i] * keys.[l].[t].[kv * c.HeadDim + i]
                           s / sqrt (float c.HeadDim))
                   let m = Array.max scores
                   let e = scores |> Array.map (fun s -> exp (s - m))
                   let z = Array.sum e
                   for t in 0 .. pos do
                       for i in 0 .. c.HeadDim - 1 do
                           att.[head * c.HeadDim + i] <- att.[head * c.HeadDim + i] + e.[t] / z * vals.[l].[t].[kv * c.HeadDim + i]
               x <- Array.map2 (+) x (matvec a.Wo att)
               let f = model.Layers.[l].Ffn
               let h = rmsNorm x f.Norm
               match f.Block with
               | Dense d ->
                   let g = matvec d.Gate h
                   let u = matvec d.Up h
                   x <- Array.map2 (+) x (matvec d.Down (Array.map2 (fun gi ui -> silu gi * ui) g u))
               | Moe m ->
                   let mc = c.Moe.Value
                   let logits = matvec m.Router h
                   let mx = Array.max logits
                   let probs = logits |> Array.map (fun v -> exp (v - mx))
                   let z = Array.sum probs
                   // top-k by probability, ties -> lower index (stable sort on descending value).
                   let chosen = probs |> Array.mapi (fun e p -> e, p / z) |> Array.sortBy (fun (e, p) -> -p, e) |> Array.take mc.ExpertsPerToken
                   let total = if mc.NormalizeTopK then chosen |> Array.sumBy snd else 1.0
                   let hiddenRows = mc.ExpertHiddenDim
                   for (e, p) in chosen do
                       let g = matvecRows m.GateExperts (e * hiddenRows) hiddenRows h
                       let u = matvecRows m.UpExperts (e * hiddenRows) hiddenRows h
                       let down = matvecRows m.DownExperts (e * c.Dim) c.Dim (Array.map2 (fun gi ui -> silu gi * ui) g u)
                       x <- Array.map2 (fun xi di -> xi + p / total * di) x down
           yield matvec model.Output (rmsNorm x model.OutputNorm) |]
