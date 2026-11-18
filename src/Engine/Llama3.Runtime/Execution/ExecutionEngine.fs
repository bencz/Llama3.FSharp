namespace Llama3.Runtime

open System
open Llama3.Core

/// Materialises an <see cref="ExecutionPlan"/>: uploads weights, creates KV caches and builds the
/// transformer pipeline. Owns every device resource except the backends themselves.
[<Sealed>]
type ExecutionEngine private (plan: ExecutionPlan, transformer: Transformer, resources: IDisposable list, backends: IComputeBackend list) =
    let mutable disposed = false

    member _.Plan = plan
    member _.Transformer = transformer
    member _.Backends = backends

    member _.Forward(token: int, position: int, computeLogits: bool) = transformer.Forward(token, position, computeLogits)

    /// <paramref name="gpus"/> is aligned with <c>plan.Gpus</c>; entries the plan does not use may be None.
    static member Create(model: HostModel, plan: ExecutionPlan, cpu: IComputeBackend, gpus: IComputeBackend option[]) =
        let config = model.Config
        let backendOf placement =
            match placement with
            | OnCpu -> cpu
            | OnGpu g when g < gpus.Length && gpus.[g].IsSome -> gpus.[g].Value
            | OnGpu g -> invalidOp $"The plan places work on GPU%d{g} but no backend was opened for it"
        let resources = ResizeArray<IDisposable>()
        try
            // One resident source per device holding exactly the parts placed there; streamed
            // (dense) layers go to the GPU's streaming source with both of their parts.
            let sourceFor = Collections.Generic.Dictionary<struct (int * LayerPart), ILayerSource>()
            let register (source: ILayerSource) =
                resources.Add source
                for (layer, part) in source.Parts do sourceFor.[struct (layer, part)] <- source
            for placement in OnCpu :: (plan.UsedGpus |> List.map OnGpu) do
                let parts =
                    [| for l in plan.Layers do
                           if l.Weights = Resident then
                               let host = model.Layers.[l.Layer]
                               if l.Attention = placement then yield l.Layer, host, AttentionPart
                               if l.Ffn = placement then yield l.Layer, host, FfnPart |]
                if parts.Length > 0 then register (new ResidentLayerSource(backendOf placement, parts))
            plan.Layers
            |> Array.filter (fun l -> l.Weights = Streamed)
            |> Array.groupBy (fun l ->
                match l.Attention, l.Ffn with
                | OnGpu a, OnGpu f when a = f -> a
                | _ -> invalidOp $"Streamed layer %d{l.Layer} must compute both parts on one GPU")
            |> Array.iter (fun (g, layers) ->
                match backendOf (OnGpu g) with
                | :? IWeightStreaming as s -> register (s.CreateStreamedSource(layers |> Array.map (fun l -> l.Layer, model.Layers.[l.Layer])))
                | b -> invalidOp $"Backend %s{b.Name} cannot stream weights")
            let stages =
                plan.Layers
                |> Array.map (fun l ->
                    let kvBackend = backendOf l.KvCache
                    let cache = kvBackend.CreateKvCache(plan.Capacity, config.KvDim)
                    resources.Add cache
                    { Layer = l.Layer
                      AttentionBackend = backendOf l.Attention
                      AttentionWeights = sourceFor.[struct (l.Layer, AttentionPart)]
                      KvBackend = kvBackend
                      Cache = cache
                      FfnBackend = backendOf l.Ffn
                      FfnWeights = sourceFor.[struct (l.Layer, FfnPart)] })
            let outputBackend = backendOf plan.Output
            let norm = outputBackend.LoadWeight model.OutputNorm
            let classifier = outputBackend.LoadWeight model.Output
            resources.Add norm
            resources.Add classifier
            let transformer =
                new Transformer(
                    config,
                    model.TokenEmbedding,
                    model.RopeFreqFactors,
                    stages,
                    { Backend = outputBackend; Norm = norm; Classifier = classifier },
                    plan.Capacity)
            // The transformer is disposed before the weights and caches it uses (same order as
            // ExecutionEngine.Dispose), including when the warm-up below fails.
            resources.Add transformer
            // Warm-up: JIT-compiles every kernel the plan uses (ILGPU compiles on first launch) and
            // waits for all uploads. The KV slot written here is overwritten by the first real token.
            transformer.Forward(0, 0, true) |> ignore
            let backends = cpu :: (plan.UsedGpus |> List.map (fun g -> backendOf (OnGpu g)))
            for b in backends do b.Synchronize()
            new ExecutionEngine(plan, transformer, List.ofSeq resources, backends)
        with _ ->
            for r in Seq.rev resources do r.Dispose()
            reraise ()

    interface IDisposable with
        member _.Dispose() =
            if not disposed then
                disposed <- true
                for r in List.rev resources do r.Dispose()
