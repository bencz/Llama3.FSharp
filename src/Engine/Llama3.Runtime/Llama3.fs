namespace Llama3

open System
open System.Collections.Generic
open System.Threading
open Llama3.Core
open Llama3.Runtime

/// A loaded model ready for inference. Not thread-safe: one generation at a time.
[<Sealed>]
type Llama3Model internal
    (info: ModelInfo, engine: ExecutionEngine, devices: DeviceManager, backends: IComputeBackend list, resources: IDisposable list) =
    /// Identifies who last wrote the KV cache, so chat sessions know when to replay their history.
    let mutable cacheOwner = 0L
    let mutable disposed = false
    static let mutable nextOwner = 0L

    member _.Info = info
    member _.Name = info.Name
    member _.Config = info.Config
    member _.Tokenizer = info.Tokenizer
    member _.Plan = engine.Plan
    member _.Backends = backends |> List.map (fun b -> b.Name)

    /// Sensible defaults for this model (llama3.cu behaviour for llama2.c checkpoints).
    member _.DefaultSettings =
        match info.Config.Architecture with
        | Llama2C -> GenerationSettings.Legacy
        | _ -> GenerationSettings.Default

    /// Per-phase timing of forward passes (None = disabled). Enabling it synchronises devices at
    /// every phase boundary.
    member _.Profile
        with get () = engine.Transformer.Profile
        and set (value: ForwardProfile option) = engine.Transformer.Profile <- value

    /// Adds an introspection observer (hidden states / routing / logits).
    member _.AddObserver(observer: IForwardObserver) = engine.Transformer.AddObserver observer
    member _.RemoveObserver(observer: IForwardObserver) = engine.Transformer.RemoveObserver observer

    member internal _.Engine = engine
    member internal _.NewOwner() = Interlocked.Increment(&nextOwner)
    member internal _.CacheOwner with get () = cacheOwner and set v = cacheOwner <- v

    /// Prompt tokens for a single-shot generation.
    member _.BuildPrompt(prompt: string, settings: GenerationSettings) : int[] =
        let useChat =
            match settings.UseChatTemplate, info.ChatTemplate with
            | Some false, _ | _, None -> false
            | _, Some _ -> true
        if useChat then
            let template = info.ChatTemplate.Value
            PromptEncoding.encode info.Tokenizer (template.Begin settings.SystemPrompt @ template.UserTurn(prompt, settings.EnableThinking))
        else
            let body = info.Tokenizer.Encode prompt
            let tokens =
                match info.AddBos, info.Tokenizer.BosId with
                | true, Some bos -> Array.append [| bos |] body
                | _ -> body
            if settings.LegacyPromptPatch && info.Config.Architecture = Llama2C && tokens.Length > 1 && tokens.[1] = 306 then
                tokens.[1] <- 76
            if tokens.Length = 0 then [| defaultArg info.Tokenizer.BosId 0 |] else tokens

    /// Streams a completion of <paramref name="prompt"/>; the KV cache starts empty.
    member this.GenerateStream(prompt: string, settings: GenerationSettings, onToken: GeneratedToken -> unit, ?cancellation: CancellationToken) =
        let tokens = this.BuildPrompt(prompt, settings)
        cacheOwner <- this.NewOwner()
        let outcome =
            Decoding.run engine info 0 tokens settings onToken (defaultArg cancellation CancellationToken.None) (List<int>())
        outcome.Result

    /// Generates a completion and returns the answer text.
    member this.Generate(prompt: string, ?settings: GenerationSettings) : string =
        let result = this.GenerateStream(prompt, defaultArg settings this.DefaultSettings, ignore)
        result.Text

    /// Raw forward pass (tests, introspection). Returns a copy of the logits.
    member _.Forward(token: int, position: int) : float32[] =
        cacheOwner <- -1L
        match engine.Forward(token, position, true) with
        | ValueSome logits -> Array.copy logits
        | ValueNone -> Array.empty

    /// Starts a multi-turn conversation that keeps its KV cache between turns.
    member this.CreateChat(?systemPrompt: string, ?enableThinking: bool) =
        match info.ChatTemplate with
        | None -> invalidOp $"Model '%s{info.Name}' has no chat template"
        | Some template -> new ChatSession(this, template, systemPrompt, defaultArg enableThinking false)

    interface IDisposable with
        member _.Dispose() =
            if not disposed then
                disposed <- true
                // Device resources first, then the backends, the driver and finally the mappings
                // the host weights live in.
                try
                    try (engine :> IDisposable).Dispose()
                    finally
                        for b in backends do b.Dispose()
                        (devices :> IDisposable).Dispose()
                finally
                    for r in resources do r.Dispose()

/// A conversation. Each turn feeds only the new tokens; if another generation used the model in
/// between, the session transparently replays its history.
and [<Sealed>] ChatSession internal (model: Llama3Model, template: ChatTemplate, systemPrompt: string option, enableThinking: bool) =
    let owner = model.NewOwner()
    let history = List<int>()
    let mutable pending: int option = None
    let mutable started = false
    let info = model.Info

    member _.History = history :> IReadOnlyList<int>
    member _.Position = history.Count

    member _.Reset() =
        history.Clear()
        pending <- None
        started <- false

    /// Feeds the new turn and generates the reply. The turn is transactional: if it stops before
    /// the whole prompt reached the model (cancellation, token limit) or throws, the session is
    /// left exactly as before the call, so the message can simply be sent again.
    member _.Send(message: string, ?settings: GenerationSettings, ?onToken: GeneratedToken -> unit, ?cancellation: CancellationToken) =
        let settings = defaultArg settings GenerationSettings.Default
        let encode parts = PromptEncoding.encode info.Tokenizer parts
        let endOfTurn = info.Tokenizer.TryGetSpecial template.EndOfTurnMarker
        let newTokens =
            [| if not started then
                   yield! encode (template.Begin systemPrompt)
               else
                   // Close the previous assistant turn. Its final token was sampled but never fed:
                   // a stop token (end of turn or end of text) is replaced by the template's close.
                   let close = encode template.EndAssistantTurn
                   match pending with
                   | Some p when Some p = endOfTurn || info.StopTokens.Contains p -> yield! close
                   | Some p ->
                       yield p
                       yield! close
                   | None -> yield! close
               yield! encode (template.UserTurn(message, enableThinking)) |]
        let savedCount = history.Count
        let savedPending = pending
        let savedStarted = started
        let replay = model.CacheOwner <> owner
        let savedHistory = if replay then history.ToArray() else Array.empty
        let rollback () =
            if replay then
                history.Clear()
                history.AddRange savedHistory
                // The cache now holds a partial replay: force a full one next time.
                model.CacheOwner <- -1L
            else history.RemoveRange(savedCount, history.Count - savedCount)
            pending <- savedPending
            started <- savedStarted
        let startPosition, feed =
            if replay then
                // The cache was overwritten by someone else: replay the whole conversation.
                let all = Array.append savedHistory newTokens
                history.Clear()
                0, all
            else history.Count, newTokens
        started <- true
        model.CacheOwner <- owner
        let outcome =
            try
                Decoding.run model.Engine info startPosition feed settings (defaultArg onToken ignore)
                    (defaultArg cancellation CancellationToken.None) history
            with _ ->
                rollback ()
                reraise ()
        if outcome.PromptComplete feed then pending <- outcome.Pending
        else rollback ()
        outcome.Result

    interface IDisposable with
        member _.Dispose() = history.Clear()

/// Entry point: <c>Llama3.load options "model.gguf"</c>.
[<RequireQualifiedAccess>]
module Llama3 =

    let private loadOptions (options: ModelOptions) : LoadOptions =
        { UseMmap = options.UseMmap
          Populate = true
          Concurrency = options.LoadConcurrency |> Option.defaultValue LoadOptions.Default.Concurrency }

    /// GPUs opened for planning, with the warnings gathered on the way (they end up in the plan
    /// notes instead of being printed by the library).
    type private Prepared =
        { Plan: ExecutionPlan
          /// Aligned with Plan.Gpus.
          Opened: IComputeBackend list }

    /// Resolves and opens the requested GPUs, then plans the model over them. On failure every
    /// opened backend is released.
    let private prepare (options: ModelOptions) (model: HostModel) (devices: DeviceManager) : Prepared =
        let warnings = ResizeArray<string>()
        let mutable opened: (GpuDevice * IComputeBackend) list = []
        try
            let cpuOnlyRequest = not options.Devices.IsEmpty && options.Devices |> List.forall ((=) CPU)
            let options = if cpuOnlyRequest then { options with ExecutionMode = CpuOnly } else options
            let requested =
                match options.ExecutionMode with
                | CpuOnly -> []
                | _ ->
                    match devices.ResolveGpus options.Devices with
                    | Ok [] ->
                        devices.CudaUnavailableReason |> Option.iter warnings.Add
                        []
                    | Ok gpus -> gpus
                    | Error e when options.UseCpuFallback ->
                        warnings.Add e
                        []
                    | Error e -> invalidOp e
            opened <-
                requested
                |> List.choose (fun d ->
                    try Some(d, devices.OpenGpu d)
                    with ex when options.UseCpuFallback ->
                        warnings.Add $"could not open %s{d.Name}: %s{ex.Message}"
                        None)
            let capacity = Scheduler.capacityFor options model.Config.ContextLength
            let estimate = MemoryManager.estimate Llama3.Backend.Cuda.GpuLayout.deviceBytes model capacity
            let gpuFree =
                opened |> List.map (fun (d, b) -> d, b.MemoryInfo |> Option.map (fun m -> m.Free) |> Option.defaultValue 0L)
            let stagingBytes (layers: int[]) =
                match opened with
                | [ (_, (:? IWeightStreaming as s)) ] -> s.StagingBytes(layers |> Array.map (fun i -> model.Layers.[i]))
                | _ -> Int64.MaxValue / 4L
            match Scheduler.plan options estimate capacity gpuFree stagingBytes with
            | Ok p -> { Plan = { p with Notes = List.ofSeq warnings @ p.Notes }; Opened = opened |> List.map snd }
            | Error e -> invalidOp e
        with _ ->
            for (_, b) in opened do b.Dispose()
            reraise ()

    /// Plans and uploads an already loaded model (custom loaders, synthetic models in tests).
    /// Takes ownership of <c>loaded.Resources</c>: they are released with the model, or right
    /// away if this call fails.
    let fromLoaded (options: ModelOptions) (loaded: LoadedModel) : Llama3Model =
        let releaseMappings () = for r in loaded.Resources do r.Dispose()
        let model = loaded.Model
        let devices = new DeviceManager()
        let mutable owned: IComputeBackend list = []
        try
            let prepared = prepare options model devices
            let plan = prepared.Plan
            // Release GPUs the plan does not use; keep indices aligned with plan.Gpus.
            let used = plan.UsedGpus
            let gpus =
                prepared.Opened
                |> List.mapi (fun i b ->
                    if List.contains i used then
                        owned <- b :: owned
                        Some b
                    else
                        b.Dispose()
                        None)
                |> Array.ofList
            let cpu = devices.OpenCpu options :> IComputeBackend
            owned <- cpu :: owned
            let engine = ExecutionEngine.Create(model, plan, cpu, gpus)
            // Upload staging is only needed while loading.
            for g in gpus do
                match g with
                | Some(:? Llama3.Backend.Cuda.CudaBackend as c) -> c.ReleaseStaging()
                | _ -> ()
            let tokenizer = loaded.Tokenizer
            let info =
                { Name = loaded.Name
                  SourcePath = loaded.SourcePath
                  Config = model.Config
                  Tokenizer = tokenizer
                  ChatTemplate = loaded.ChatTemplate
                  AddBos = loaded.AddBos
                  StopTokens = loaded.StopTokens
                  ThinkOpen = tokenizer.TryGetSpecial "<think>"
                  ThinkClose = tokenizer.TryGetSpecial "</think>" }
            new Llama3Model(info, engine, devices, cpu :: (gpus |> Array.choose id |> List.ofArray), loaded.Resources)
        with _ ->
            for b in owned do b.Dispose()
            (devices :> IDisposable).Dispose()
            releaseMappings ()
            reraise ()

    /// Loads a model, plans its placement over the available devices and uploads the weights.
    let load (options: ModelOptions) (path: string) : Llama3Model =
        fromLoaded options (ModelLoader.load path options.TokenizerPath (loadOptions options))

    /// Plans the placement without uploading anything: the file is mapped (tensor data is never
    /// read for natively supported formats) and the GPUs are opened only to query free memory.
    let plan (options: ModelOptions) (path: string) : ExecutionPlan =
        let loaded = ModelLoader.load path options.TokenizerPath { loadOptions options with UseMmap = true; Populate = false }
        try
            use devices = new DeviceManager()
            let prepared = prepare options loaded.Model devices
            for b in prepared.Opened do b.Dispose()
            prepared.Plan
        finally
            for r in loaded.Resources do r.Dispose()

    /// GPUs visible to the runtime.
    let devices () : GpuDevice list =
        use dm = new DeviceManager()
        dm.Gpus
