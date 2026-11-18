namespace Llama3.Core

open System

/// All-or-nothing device allocation: if any allocation fails (e.g. out of device memory), the
/// buffers already obtained are released before the exception propagates.
module Allocation =
    let all (backend: IComputeBackend) (lengths: int[]) : IBuffer[] =
        let owned = Collections.Generic.List<IBuffer>(lengths.Length)
        try
            for n in lengths do owned.Add(backend.Allocate n)
            owned.ToArray()
        with _ ->
            for b in owned do b.Dispose()
            reraise ()

/// Scratch for one MoE FFN evaluation on a device: router logits, one gate/up/output buffer per
/// routed expert, and the batch descriptors handed to <see cref="IComputeBackend.MatVecBatch"/>
/// (preallocated so that routing a token allocates nothing).
[<Sealed>]
type MoeScratch(backend: IComputeBackend, config: ModelConfig, moe: MoeConfig, xb: IBuffer) =
    let k = moe.ExpertsPerToken
    let buffers =
        Allocation.all backend
            [| yield moe.ExpertCount
               yield! Array.create k moe.ExpertHiddenDim
               yield! Array.create k moe.ExpertHiddenDim
               yield! Array.create k config.Dim |]
    let routerLogits = buffers.[0]
    let gate = buffers.[1 .. k]
    let up = buffers.[k + 1 .. 2 * k]
    let output = buffers.[2 * k + 1 .. 3 * k]

    member _.RouterLogits = routerLogits
    member val RouterHost = Array.zeroCreate<float32> moe.ExpertCount
    member val Experts = Array.zeroCreate<int> k
    member val ExpertWeights = Array.zeroCreate<float32> k
    /// Gate/up products of every routed expert share the normalised input.
    member val GateUpInputs = Array.create (2 * k) xb
    member val GateUpOutputs = Array.init (2 * k) (fun i -> if i % 2 = 0 then gate.[i / 2] else up.[i / 2])
    member val GateUpWeights = Array.zeroCreate<IWeight> (2 * k)
    member _.Gate = gate
    member _.Up = up
    member _.Output = output
    member val DownWeights = Array.zeroCreate<IWeight> k

    interface IDisposable with
        member _.Dispose() =
            for b in buffers do b.Dispose()

/// Per-device activation buffers of the forward pass. One workspace exists for every backend that
/// executes at least one stage (attention, FFN, KV attention or classifier).
[<Sealed>]
type Workspace(backend: IComputeBackend, config: ModelConfig, cacheCapacity: int) =
    // Dense FFN buffers are not needed by sparse models (experts use MoeScratch).
    let hiddenDim = if config.Moe.IsSome then 1 else config.HiddenDim
    let buffers =
        Allocation.all backend
            [| config.Dim; config.Dim; config.Dim; config.QDim; config.KvDim; config.KvDim; config.QDim
               config.NumHeads * cacheCapacity; hiddenDim; hiddenDim |]
    let x = buffers.[0]
    let xb = buffers.[1]
    let xb2 = buffers.[2]
    let q = buffers.[3]
    let k = buffers.[4]
    let v = buffers.[5]
    let attOut = buffers.[6]
    let att = buffers.[7]
    let hb = buffers.[8]
    let hb2 = buffers.[9]
    let mutable logits: IBuffer option = None
    let moe = lazy (config.Moe |> Option.map (fun m -> new MoeScratch(backend, config, m, xb)))

    member _.Backend = backend
    /// Residual stream.
    member _.X = x
    member _.Xb = xb
    member _.Xb2 = xb2
    member _.Q = q
    member _.K = k
    member _.V = v
    member _.AttOut = attOut
    /// Attention scores, NumHeads × capacity.
    member _.Att = att
    member _.Hb = hb
    member _.Hb2 = hb2
    member val QkvInputs = [| xb; xb; xb |]
    member val QkvOutputs = [| q; k; v |]
    member val GateUpInputs = [| xb; xb |]
    member val GateUpOutputs = [| hb; hb2 |]
    /// Per-call weight descriptors, refilled in place (no allocation per layer).
    member val QkvWeights = Array.zeroCreate<IWeight> 3
    member val GateUpWeights = Array.zeroCreate<IWeight> 2

    /// MoE scratch, created on first use by an FFN stage of this device.
    member _.Moe =
        match moe.Value with
        | Some m -> m
        | None -> invalidOp "Model has no mixture-of-experts configuration"

    member _.Logits =
        match logits with
        | Some l -> l
        | None ->
            let l = backend.Allocate config.VocabSize
            logits <- Some l
            l

    interface IDisposable with
        member _.Dispose() =
            for b in buffers do b.Dispose()
            logits |> Option.iter (fun l -> l.Dispose())
            if moe.IsValueCreated then moe.Value |> Option.iter (fun m -> (m :> IDisposable).Dispose())

/// Host-side staging used when activations cross devices.
[<Sealed>]
type HostStaging(config: ModelConfig) =
    member val X = Array.zeroCreate<float32> config.Dim
    member val Q = Array.zeroCreate<float32> config.QDim
    member val K = Array.zeroCreate<float32> config.KvDim
    member val V = Array.zeroCreate<float32> config.KvDim
    member val AttOut = Array.zeroCreate<float32> config.QDim

module Attention =

    /// Appends k/v at <paramref name="position"/> and attends q over the cache.
    /// When the cache lives on another device than the projections, q/k/v cross over through
    /// host memory (a few KiB per layer) and only the attention output comes back.
    let step
        (compute: Workspace)
        (kvSide: Workspace)
        (cache: IKvCache)
        (staging: HostStaging)
        (position: int)
        (spec: AttentionSpec)
        =
        let cb = compute.Backend
        if obj.ReferenceEquals(compute, kvSide) then
            cb.StoreKv(cache, compute.K, compute.V, position)
            cb.Attention(compute.AttOut, compute.Q, cache, position, spec, compute.Att)
        else
            let kb = kvSide.Backend
            cb.Download(compute.Q, staging.Q)
            cb.Download(compute.K, staging.K)
            cb.Download(compute.V, staging.V)
            kb.Upload(staging.Q, kvSide.Q)
            kb.Upload(staging.K, kvSide.K)
            kb.Upload(staging.V, kvSide.V)
            kb.StoreKv(cache, kvSide.K, kvSide.V, position)
            kb.Attention(kvSide.AttOut, kvSide.Q, cache, position, spec, kvSide.Att)
            kb.Download(kvSide.AttOut, staging.AttOut)
            cb.Upload(staging.AttOut, compute.AttOut)
