namespace Llama3.Core

open System
open System.Collections.Generic

/// Execution of one decoder block. Attention and FFN are placed independently (op-level hybrid):
/// e.g. attention + KV on the GPU while a MoE's experts run on the CPU.
type LayerStage =
    { Layer: int
      AttentionBackend: IComputeBackend
      AttentionWeights: ILayerSource
      KvBackend: IComputeBackend
      Cache: IKvCache
      FfnBackend: IComputeBackend
      FfnWeights: ILayerSource }

/// Final norm + classifier.
type OutputStage =
    { Backend: IComputeBackend
      Norm: IWeight
      Classifier: IWeight }

/// Introspection hooks. Hidden states and routing decisions are only materialised on the host when
/// an observer asks for them, so observers that only want logits cost nothing per layer.
type IForwardObserver =
    abstract CaptureHiddenStates: bool
    /// Residual stream after <paramref name="layer"/> (copied; safe to retain).
    abstract OnHiddenState: layer: int * position: int * hidden: float32[] -> unit
    /// Experts chosen by a MoE layer and their gate weights (copies; safe to retain).
    abstract OnExpertRouting: layer: int * position: int * experts: int[] * weights: float32[] -> unit
    /// Raw logits before sampling (the array is reused by the next step: copy to retain).
    abstract OnLogits: position: int * logits: float32[] -> unit

/// Phases timed by the profiler.
type ForwardPhase =
    | Embedding = 0
    | Attention = 1
    | Transfer = 2
    | Routing = 3
    | Ffn = 4
    | Output = 5

/// Wall-clock breakdown of forward passes per phase and device. While enabled, devices are
/// synchronised at every phase boundary so asynchronous GPU work is attributed to the right
/// phase (this serialises the pipeline: use it to find bottlenecks, not to measure peak speed).
[<Sealed>]
type ForwardProfile() =
    let totals = Dictionary<struct (ForwardPhase * string), int64>()
    let mutable tokens = 0

    member _.Add(phase: ForwardPhase, device: string, ticks: int64) =
        let key = struct (phase, device)
        match totals.TryGetValue key with
        | true, t -> totals.[key] <- t + ticks
        | _ -> totals.[key] <- ticks

    member _.CountToken() = tokens <- tokens + 1
    member _.Tokens = tokens

    /// (phase, device, milliseconds per token), largest first.
    member _.PerToken() =
        [ for KeyValue(struct (phase, device), ticks) in totals ->
              phase, device, float ticks / float Diagnostics.Stopwatch.Frequency * 1000.0 / float (max 1 tokens) ]
        |> List.sortByDescending (fun (_, _, ms) -> ms)

    member _.Reset() =
        totals.Clear()
        tokens <- 0

/// The decoder forward pass over a heterogeneous pipeline of stages.
/// Activations stay on a device while consecutive stages run there; crossing a device boundary
/// costs one Dim-sized transfer through host memory.
[<Sealed>]
type Transformer
    (
        config: ModelConfig,
        embedding: HostTensor,
        ropeFreqFactors: HostTensor option,
        stages: LayerStage[],
        output: OutputStage,
        cacheCapacity: int
    ) =
    let workspaces = Dictionary<IComputeBackend, Workspace>(HashIdentity.Reference)
    let ropeFactors = Dictionary<IComputeBackend, IWeight>(HashIdentity.Reference)
    let staging = HostStaging(config)
    let logits = Array.zeroCreate<float32> config.VocabSize
    let observers = ResizeArray<IForwardObserver>()
    let mutable profile: ForwardProfile option = None
    let mutable mark = 0L

    /// Closes the current phase: waits for <paramref name="device"/> and charges the elapsed time.
    let lap (phase: ForwardPhase) (device: IComputeBackend) =
        match profile with
        | Some p ->
            device.Synchronize()
            let now = Diagnostics.Stopwatch.GetTimestamp()
            p.Add(phase, device.Name, now - mark)
            mark <- now
        | None -> ()

    let workspace (b: IComputeBackend) =
        match workspaces.TryGetValue b with
        | true, ws -> ws
        | _ ->
            let ws = new Workspace(b, config, cacheCapacity)
            workspaces.[b] <- ws
            ws

    let ropeSpec (b: IComputeBackend) =
        let factors =
            ropeFreqFactors
            |> Option.map (fun t ->
                match ropeFactors.TryGetValue b with
                | true, w -> w
                | _ ->
                    let w = b.LoadWeight t
                    ropeFactors.[b] <- w
                    w)
        { Mode = config.RopeMode
          Theta = config.RopeTheta
          HeadDim = config.HeadDim
          NumHeads = config.NumHeads
          NumKvHeads = config.NumKvHeads
          FreqFactors = factors }

    let attentionSpec =
        { NumHeads = config.NumHeads
          NumKvHeads = config.NumKvHeads
          HeadDim = config.HeadDim }

    let release () =
        for ws in workspaces.Values do (ws :> IDisposable).Dispose()
        for w in ropeFactors.Values do w.Dispose()
        workspaces.Clear()
        ropeFactors.Clear()

    // Pre-create every workspace and RoPE table (logits and MoE scratch are allocated on first
    // use). Everything already allocated is released if a later allocation fails.
    do
        try
            for s in stages do
                workspace s.AttentionBackend |> ignore
                workspace s.KvBackend |> ignore
                workspace s.FfnBackend |> ignore
                ropeSpec s.AttentionBackend |> ignore
            workspace output.Backend |> ignore
        with _ ->
            release ()
            reraise ()
    let ropeSpecs = stages |> Array.map (fun s -> ropeSpec s.AttentionBackend)

    /// Moves the residual stream from <paramref name="source"/> (None = host staging) to <paramref name="target"/>.
    let transfer (source: Workspace option) (target: Workspace) =
        match source with
        | Some s when obj.ReferenceEquals(s, target) -> ()
        | Some s ->
            s.Backend.Download(s.X, staging.X)
            target.Backend.Upload(staging.X, target.X)
            lap ForwardPhase.Transfer target.Backend
        | None -> target.Backend.Upload(staging.X, target.X)

    let runAttention (stage: LayerStage) (ws: Workspace) (position: int) (rope: RopeSpec) =
        let b = stage.AttentionBackend
        let w = stage.AttentionWeights.AcquireAttention stage.Layer
        let eps = config.NormEps
        b.RmsNorm(ws.Xb, ws.X, w.Norm, eps)
        let weights = ws.QkvWeights
        weights.[0] <- w.Wq
        weights.[1] <- w.Wk
        weights.[2] <- w.Wv
        b.MatVecBatch(ws.QkvInputs, ws.QkvOutputs, weights)
        match w.Bq with Some bias -> b.AddBias(ws.Q, bias) | None -> ()
        match w.Bk with Some bias -> b.AddBias(ws.K, bias) | None -> ()
        match w.Bv with Some bias -> b.AddBias(ws.V, bias) | None -> ()
        match w.QNorm with Some n -> b.HeadRmsNorm(ws.Q, n, config.NumHeads, config.HeadDim, eps) | None -> ()
        match w.KNorm with Some n -> b.HeadRmsNorm(ws.K, n, config.NumKvHeads, config.HeadDim, eps) | None -> ()
        b.Rope(ws.Q, ws.K, position, rope)
        Attention.step ws (workspace stage.KvBackend) stage.Cache staging position attentionSpec
        b.MatVec(ws.Xb2, ws.AttOut, w.Wo)
        b.Add(ws.X, ws.Xb2)

    /// Sparse FFN: route on the host (ExpertCount floats), then two batched matvec jobs.
    let runMoe (stage: LayerStage) (ws: Workspace) (moe: DeviceMoe) (position: int) =
        let b = stage.FfnBackend
        let m = ws.Moe
        let moeConfig = config.Moe.Value
        b.MatVec(m.RouterLogits, ws.Xb, moe.Router)
        b.Download(m.RouterLogits, m.RouterHost)
        MoeRouting.select (ReadOnlySpan m.RouterHost) moeConfig.NormalizeTopK (Span m.Experts) (Span m.ExpertWeights)
        lap ForwardPhase.Routing b
        if observers.Count > 0 then
            for o in observers do
                o.OnExpertRouting(stage.Layer, position, Array.copy m.Experts, Array.copy m.ExpertWeights)
        for i in 0 .. m.Experts.Length - 1 do
            let e = m.Experts.[i]
            m.GateUpWeights.[2 * i] <- moe.Gate.[e]
            m.GateUpWeights.[2 * i + 1] <- moe.Up.[e]
            m.DownWeights.[i] <- moe.Down.[e]
        b.MatVecBatch(m.GateUpInputs, m.GateUpOutputs, m.GateUpWeights)
        for i in 0 .. m.Experts.Length - 1 do
            b.SwiGlu(m.Gate.[i], m.Up.[i])
        b.MatVecBatch(m.Gate, m.Output, m.DownWeights)
        for i in 0 .. m.Experts.Length - 1 do
            b.AddScaled(ws.X, m.Output.[i], m.ExpertWeights.[i])

    let runFfn (stage: LayerStage) (ws: Workspace) (position: int) =
        let b = stage.FfnBackend
        let f = stage.FfnWeights.AcquireFfn stage.Layer
        b.RmsNorm(ws.Xb, ws.X, f.Norm, config.NormEps)
        match f.Block with
        | DenseBlock(gate, up, down) ->
            let weights = ws.GateUpWeights
            weights.[0] <- gate
            weights.[1] <- up
            b.MatVecBatch(ws.GateUpInputs, ws.GateUpOutputs, weights)
            b.SwiGlu(ws.Hb, ws.Hb2)
            b.MatVec(ws.Xb, ws.Hb, down)
            b.Add(ws.X, ws.Xb)
        | MoeBlock moe -> runMoe stage ws moe position

    member _.Config = config
    member _.Stages = stages
    member _.Output = output
    member _.CacheCapacity = cacheCapacity

    /// Enables (Some) or disables (None) per-phase profiling.
    member _.Profile
        with get () = profile
        and set (value: ForwardProfile option) = profile <- value

    member _.AddObserver(o: IForwardObserver) = observers.Add o
    member _.RemoveObserver(o: IForwardObserver) = observers.Remove o |> ignore

    /// Runs one token through the network at <paramref name="position"/>. Returns the logits
    /// buffer (reused across calls) or ValueNone when <paramref name="computeLogits"/> is false
    /// (prompt prefill only needs the KV cache side effects).
    member _.Forward(token: int, position: int, computeLogits: bool) : float32[] voption =
        if token < 0 || token >= config.VocabSize then
            raise (ArgumentOutOfRangeException(nameof token, $"Token %d{token} outside vocabulary (%d{config.VocabSize})"))
        if position < 0 || position >= cacheCapacity then
            raise (ArgumentOutOfRangeException(nameof position, $"Position %d{position} outside KV cache capacity %d{cacheCapacity}"))
        mark <- Diagnostics.Stopwatch.GetTimestamp()
        Dequantize.row embedding token (Span staging.X)
        profile |> Option.iter (fun p ->
            let now = Diagnostics.Stopwatch.GetTimestamp()
            p.Add(ForwardPhase.Embedding, "host", now - mark)
            mark <- now)
        let mutable captureHidden = false
        for o in observers do
            if o.CaptureHiddenStates then captureHidden <- true
        let mutable current: Workspace option = None
        for i in 0 .. stages.Length - 1 do
            let stage = stages.[i]
            let attention = workspace stage.AttentionBackend
            transfer current attention
            runAttention stage attention position ropeSpecs.[i]
            lap ForwardPhase.Attention stage.AttentionBackend
            let ffn = workspace stage.FfnBackend
            transfer (Some attention) ffn
            runFfn stage ffn position
            lap ForwardPhase.Ffn stage.FfnBackend
            current <- Some ffn
            if captureHidden then
                let hidden = Array.zeroCreate<float32> config.Dim
                stage.FfnBackend.Download(ffn.X, hidden)
                for o in observers do
                    if o.CaptureHiddenStates then o.OnHiddenState(stage.Layer, position, hidden)
        if computeLogits then
            let ws = workspace output.Backend
            transfer current ws
            let b = output.Backend
            b.RmsNorm(ws.Xb, ws.X, output.Norm, config.NormEps)
            b.MatVec(ws.Logits, ws.Xb, output.Classifier)
            b.Download(ws.Logits, logits)
            lap ForwardPhase.Output b
            profile |> Option.iter (fun p -> p.CountToken())
            for o in observers do o.OnLogits(position, logits)
            ValueSome logits
        else
            current |> Option.iter (fun ws -> ws.Backend.Synchronize())
            profile |> Option.iter (fun p -> p.CountToken())
            ValueNone

    interface IDisposable with
        member _.Dispose() = release ()
