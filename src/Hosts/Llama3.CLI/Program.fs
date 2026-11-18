module Llama3.Cli.Program

open System
open System.IO
open System.Text
open System.Threading
open Llama3
open Llama3.Core
open Llama3.Runtime
open Llama3.Cli.Arguments
open Llama3.Cli.Output

let settingsFor (model: Llama3Model) (a: CliArgs) =
    let baseSettings = model.DefaultSettings
    let s = baseSettings.Sampling
    let sampling =
        { s with
            Temperature = defaultArg a.Temperature s.Temperature
            TopK = defaultArg a.TopK s.TopK
            TopP = defaultArg a.TopP s.TopP
            MinP = defaultArg a.MinP s.MinP
            RepetitionPenalty = defaultArg a.RepeatPenalty s.RepetitionPenalty
            Seed = (match a.Seed with Some _ -> a.Seed | None -> s.Seed) }
    { baseSettings with
        Sampling = sampling
        MaxNewTokens = defaultArg a.MaxTokens baseSettings.MaxNewTokens
        MaxTotalTokens = (match a.Steps with Some _ -> a.Steps | None -> baseSettings.MaxTotalTokens)
        SystemPrompt = a.System
        EnableThinking = a.Think
        UseChatTemplate = (if a.Raw then Some false else baseSettings.UseChatTemplate) }

/// Interactive chat; <paramref name="firstMessage"/> (the command-line prompt) opens the conversation.
let runChat (model: Llama3Model) (a: CliArgs) (firstMessage: string option) (stdout: TextWriter) =
    use chat = model.CreateChat(?systemPrompt = a.System, enableThinking = a.Think)
    let settings = settingsFor model a
    eprintfn "Interactive chat: empty line or /exit to quit, /reset to clear the conversation."
    let turn (message: string) =
        let printer = TokenPrinter(stdout)
        use cts = new CancellationTokenSource()
        let handler = ConsoleCancelEventHandler(fun _ e -> e.Cancel <- true; cts.Cancel())
        Console.CancelKeyPress.AddHandler handler
        try
            model.Profile |> Option.iter (fun p -> p.Reset())
            let r = chat.Send(message, settings, printer.Print, cts.Token)
            printer.Finish()
            if r.StopReason = Cancelled then eprintfn "\n(cancelled)"
            reportStats model r
            model.Profile |> Option.iter reportProfile
        finally
            Console.CancelKeyPress.RemoveHandler handler
    firstMessage |> Option.iter (fun m ->
        eprintfn $"\n> %s{m}"
        turn m)
    let mutable running = true
    while running do
        eprintf "\n> "
        let line = Console.ReadLine()
        if isNull line || line.Trim() = "" || line.Trim() = "/exit" then running <- false
        elif line.Trim() = "/reset" then
            chat.Reset()
            eprintfn "(conversation cleared)"
        else turn line

let private describeModel (a: CliArgs) =
    eprintf "%s" ((Llama3.plan a.Options a.Model).Describe())

let private run (a: CliArgs) (stdout: TextWriter) =
    if a.PlanOnly then
        describeModel a
        0
    else
        let sw = Diagnostics.Stopwatch.StartNew()
        use model = Llama3.load a.Options a.Model
        let c = model.Config
        eprintfn $"%s{model.Name}: %A{c.Architecture}, %d{c.NumLayers} layers, dim %d{c.Dim}, heads %d{c.NumHeads}/%d{c.NumKvHeads}, vocab %d{c.VocabSize}, loaded in %.1f{sw.Elapsed.TotalSeconds}s"
        eprintf "%s" (model.Plan.Describe())
        for backend in model.Backends do eprintfn $"  backend: %s{backend}"
        if a.Profile then model.Profile <- Some(ForwardProfile())
        if a.Chat || (a.Prompt.IsNone && model.Info.ChatTemplate.IsSome && not a.Raw) then
            runChat model a a.Prompt stdout
        else
            let prompt = defaultArg a.Prompt "I have a dream"
            let printer = TokenPrinter(stdout)
            let r = model.GenerateStream(prompt, settingsFor model a, printer.Print)
            printer.Finish()
            reportStats model r
            model.Profile |> Option.iter reportProfile
        0

[<EntryPoint>]
let main argv =
    match parse argv with
    | Ok ShowHelp ->
        printf "%s" usage
        0
    | Error message ->
        eprintfn $"error: %s{message}\n"
        eprintf "%s" usage
        2
    | Ok(Run a) when a.ListDevices ->
        match Llama3.devices () with
        | [] -> printfn "No GPUs found (CPU only)."
        | gpus ->
            for g in gpus do
                printfn $"%A{g.Kind}:%d{g.Index}  %s{g.Name}  %s{g.Architecture}  %s{MemoryManager.formatBytes g.TotalMemory}"
        0
    | Ok(Run a) ->
        let stdout = new StreamWriter(Console.OpenStandardOutput(), UTF8Encoding(false))
        try
            try run a stdout
            with e ->
                // Library errors carry user-facing messages; the stack trace only on request.
                let root = match e with :? AggregateException as ag -> ag.GetBaseException() | _ -> e
                eprintfn $"error: %s{root.Message}"
                if Environment.GetEnvironmentVariable "LLAMA3_DEBUG" = "1" then eprintfn $"%O{e}"
                1
        finally
            stdout.Flush()
