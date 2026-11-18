/// End-to-end decode loop and chat semantics on a synthetic GGUF model (CPU, exact float path,
/// greedy sampling: every run is deterministic, so sessions can be compared token for token).
module Llama3.Tests.SessionTests

open System
open System.IO
open System.Threading
open Xunit
open Llama3
open Llama3.Core
open Llama3.Tests.GgufWriter

let private options =
    { ModelOptions.Default with
        ExecutionMode = CpuOnly
        ContextLength = Some 256
        QuantizedCpuActivations = false
        CpuThreads = 2 }

let private greedy n =
    { GenerationSettings.Default with MaxNewTokens = n; Sampling = SamplingSettings.Greedy }

/// Loads the synthetic model through the real GGUF path; <paramref name="adjust"/> may change the
/// loaded description (e.g. stop tokens) before the runtime takes it over.
let private withModel (adjust: LoadedModel -> LoadedModel) (test: Llama3Model -> unit) =
    let dir = Directory.CreateTempSubdirectory("llama3-session-").FullName
    try
        let path = writeModel dir "chat" (ReferenceModel.synthetic 41 GgmlType.Q8_0 false) 1
        use model = Llama3.fromLoaded options (adjust (ModelLoader.load path None LoadOptions.Default))
        test model
    finally
        Directory.Delete(dir, true)

let private history (chat: ChatSession) = chat.History |> Seq.toArray

/// The uninterrupted two-turn conversation every scenario must reproduce.
let private reference (model: Llama3Model) =
    use chat = model.CreateChat()
    let first = chat.Send("hello there", greedy 4)
    let second = chat.Send("and again", greedy 4)
    first.Text, second.Text, history chat

[<Fact>]
let ``a turn cancelled before it starts leaves the session unchanged`` () =
    withModel id (fun model ->
        let refFirst, refSecond, refHistory = reference model
        use chat = model.CreateChat()
        Assert.Equal(refFirst, (chat.Send("hello there", greedy 4)).Text)
        let before = history chat
        use cts = new CancellationTokenSource()
        cts.Cancel()
        let cancelled = chat.Send("interrupted", greedy 4, ignore, cts.Token)
        Assert.Equal(Cancelled, cancelled.StopReason)
        Assert.Equal<int[]>(before, history chat)
        Assert.Equal(refSecond, (chat.Send("and again", greedy 4)).Text)
        Assert.Equal<int[]>(refHistory, history chat))

[<Fact>]
let ``a turn cancelled while its prompt is being fed is rolled back`` () =
    withModel id (fun model ->
        let refFirst, refSecond, refHistory = reference model
        use chat = model.CreateChat()
        Assert.Equal(refFirst, (chat.Send("hello there", greedy 4)).Text)
        let before = history chat
        use cts = new CancellationTokenSource()
        // Echoed prompt tokens arrive while feeding: cancel after the third one.
        let mutable seen = 0
        let onToken (t: GeneratedToken) =
            if t.IsPrompt then
                seen <- seen + 1
                if seen = 3 then cts.Cancel()
        let cancelled = chat.Send("a long interrupted message", { greedy 4 with EchoPrompt = true }, onToken, cts.Token)
        Assert.Equal(Cancelled, cancelled.StopReason)
        Assert.True(seen >= 3, "the prompt was not being fed")
        Assert.Equal<int[]>(before, history chat)
        Assert.Equal(refSecond, (chat.Send("and again", greedy 4)).Text)
        Assert.Equal<int[]>(refHistory, history chat))

[<Fact>]
let ``interleaved sessions replay their history and stay exact`` () =
    withModel id (fun model ->
        let refFirst, refSecond, refHistory = reference model
        use a = model.CreateChat()
        use b = model.CreateChat()
        Assert.Equal(refFirst, (a.Send("hello there", greedy 4)).Text)
        b.Send("something unrelated", greedy 6) |> ignore
        model.Generate("raw text", { greedy 3 with UseChatTemplate = Some false }) |> ignore
        Assert.Equal(refSecond, (a.Send("and again", greedy 4)).Text)
        Assert.Equal<int[]>(refHistory, history a))

[<Fact>]
let ``a sampled stop token is replaced by the template's end of turn`` () =
    // Find the first token the model answers with, then declare it a stop token.
    let mutable firstAnswer = -1
    withModel id (fun model ->
        use chat = model.CreateChat()
        chat.Send("hello there", greedy 1, (fun t -> if not t.IsPrompt && firstAnswer < 0 then firstAnswer <- t.Token)) |> ignore)
    Assert.True(firstAnswer >= 0)
    withModel (fun l -> { l with StopTokens = l.StopTokens.Add firstAnswer }) (fun model ->
        use chat = model.CreateChat()
        let first = chat.Send("hello there", greedy 4)
        Assert.Equal(EndOfTurn, first.StopReason)
        let turnLength = chat.Position
        chat.Send("and again", greedy 2) |> ignore
        let endOfTurn = model.Tokenizer.TryGetSpecial "<|im_end|>"
        Assert.Equal(endOfTurn, Some (history chat).[turnLength]))

[<Fact>]
let ``a prompt longer than the context raises ContextOverflowException without side effects`` () =
    withModel id (fun model ->
        use chat = model.CreateChat()
        let long = String.replicate 200 "word "
        Assert.Throws<ContextOverflowException>(fun () -> chat.Send(long, greedy 4) |> ignore) |> ignore
        Assert.Equal(0, chat.Position)
        Assert.NotEqual(Cancelled, (chat.Send("short", greedy 2)).StopReason))

[<Fact>]
let ``streamed pieces reassemble the result and limits are validated`` () =
    withModel id (fun model ->
        let settings = { greedy 6 with UseChatTemplate = Some false }
        let answer = Text.StringBuilder()
        let thinking = Text.StringBuilder()
        let r =
            model.GenerateStream("the cat", settings, fun t ->
                (if t.Channel = Thinking then thinking else answer).Append(t.Text) |> ignore)
        Assert.Equal(r.Text, answer.ToString())
        Assert.Equal(r.Thinking, thinking.ToString())
        Assert.True(r.GeneratedTokens >= 1 && r.GeneratedTokens <= 6)
        Assert.Contains(r.StopReason, [ TokenLimit; EndOfTurn ])
        Assert.Throws<ArgumentOutOfRangeException>(fun () -> model.Generate("x", { settings with MaxNewTokens = 0 }) |> ignore) |> ignore)
