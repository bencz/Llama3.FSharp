namespace Llama3

open System
open System.Collections.Generic
open System.Diagnostics
open System.Text
open System.Threading
open Llama3.Core
open Llama3.Runtime

/// Output stream a token belongs to. Reasoning models wrap their chain of thought in
/// &lt;think&gt;...&lt;/think&gt;; those tokens are routed to <c>Thinking</c>.
type TokenChannel =
    | Answer
    | Thinking

/// One streamed token.
type GeneratedToken =
    { Token: int
      /// Decoded text (complete UTF-8 characters only; may be empty while a sequence is pending).
      Text: string
      Channel: TokenChannel
      /// Position the token will occupy in the sequence.
      Position: int
      /// True for echoed prompt tokens (EchoPrompt).
      IsPrompt: bool }

type StopReason =
    /// The model emitted an end-of-turn / end-of-sequence token.
    | EndOfTurn
    | TokenLimit
    | ContextFull
    | Cancelled

type GenerationSettings =
    { MaxNewTokens: int
      /// Cap on the total sequence length including the prompt (llama3.cu "steps").
      MaxTotalTokens: int option
      Sampling: SamplingSettings
      SystemPrompt: string option
      /// Let hybrid reasoning models think before answering (Qwen3 enable_thinking).
      EnableThinking: bool
      /// None = use the model's chat template when it has one.
      UseChatTemplate: bool option
      /// Stream the prompt tokens too (llama3.cu prints the prompt before the continuation).
      EchoPrompt: bool
      /// llama3.cu's "I have a dream" monkey patch (prompt token 306 -> 76). Only meaningful for
      /// the llama2.c tokenizer; kept to reproduce the reference output bit for bit.
      LegacyPromptPatch: bool }

    static member Default =
        { MaxNewTokens = 512
          MaxTotalTokens = None
          Sampling = SamplingSettings.Chat
          SystemPrompt = None
          EnableThinking = false
          UseChatTemplate = None
          EchoPrompt = false
          LegacyPromptPatch = false }

    /// Settings that reproduce llama3.cu: greedy, 50 total steps, prompt echoed.
    static member Legacy =
        { GenerationSettings.Default with
            MaxNewTokens = Int32.MaxValue
            MaxTotalTokens = Some 50
            Sampling = SamplingSettings.Greedy
            UseChatTemplate = Some false
            EchoPrompt = true
            LegacyPromptPatch = true }

type GenerationResult =
    { Text: string
      Thinking: string
      PromptTokens: int
      GeneratedTokens: int
      PromptTime: TimeSpan
      GenerationTime: TimeSpan
      StopReason: StopReason
      /// Sequence length after generation (KV cache fill).
      FinalPosition: int }

    member r.TokensPerSecond =
        if r.GenerationTime.TotalSeconds > 0.0 then float r.GeneratedTokens / r.GenerationTime.TotalSeconds else 0.0

    member r.PromptTokensPerSecond =
        if r.PromptTime.TotalSeconds > 0.0 then float (r.PromptTokens - 1) / r.PromptTime.TotalSeconds else 0.0

/// The tokens to feed do not fit in the KV cache (prompt longer than the context).
type ContextOverflowException(message: string) =
    inherit InvalidOperationException(message)

/// What the runtime keeps about a model after its weights have been placed on devices.
type ModelInfo =
    { Name: string
      SourcePath: string
      Config: ModelConfig
      Tokenizer: ITokenizer
      ChatTemplate: ChatTemplate option
      AddBos: bool
      StopTokens: Set<int>
      ThinkOpen: int option
      ThinkClose: int option }

/// Decode loop shared by single-shot generation and chat sessions.
module internal Decoding =

    type Outcome =
        { Result: GenerationResult
          /// Last sampled token, not yet fed to the model (fed lazily by the next turn).
          Pending: int option
          /// How many of the input tokens reached the model (all of them unless the run stopped
          /// while still feeding the prompt: cancellation or a full context).
          Fed: int }

        member o.PromptComplete(input: int[]) = o.Fed = input.Length

    /// Feeds <paramref name="tokens"/> starting at <paramref name="startPosition"/> and then samples
    /// until a stop condition. <paramref name="history"/> receives every token fed to the model.
    let run
        (engine: ExecutionEngine)
        (info: ModelInfo)
        (startPosition: int)
        (tokens: int[])
        (settings: GenerationSettings)
        (onToken: GeneratedToken -> unit)
        (cancellation: CancellationToken)
        (history: List<int>)
        : Outcome =
        if tokens.Length = 0 then invalidArg (nameof tokens) "At least one token must be fed"
        if settings.MaxNewTokens <= 0 then
            raise (ArgumentOutOfRangeException("MaxNewTokens", settings.MaxNewTokens, "must be positive"))
        match settings.MaxTotalTokens with
        | Some m when m <= 0 -> raise (ArgumentOutOfRangeException("MaxTotalTokens", m, "must be positive"))
        | _ -> ()
        let capacity = engine.Plan.Capacity
        // One slot stays free for the token sampled last (fed by the next turn).
        if startPosition + tokens.Length >= capacity then
            raise (
                ContextOverflowException
                    $"%d{tokens.Length} tokens starting at position %d{startPosition} do not fit in a context of %d{capacity} tokens"
            )
        let limit =
            match settings.MaxTotalTokens with
            | Some m -> min m capacity
            | None -> capacity
        let sampler = Sampler(settings.Sampling)
        let utf8 = Utf8StreamDecoder()
        let answer = StringBuilder()
        let thinking = StringBuilder()
        let mutable channel = Answer
        let emit (previous: int) (next: int) (position: int) (isPrompt: bool) =
            if Some next = info.ThinkOpen then channel <- Thinking
            elif Some next = info.ThinkClose then channel <- Answer
            else
                let text = utf8.Push(info.Tokenizer.Decode(previous, next))
                (if channel = Thinking then thinking else answer).Append(text) |> ignore
                onToken { Token = next; Text = text; Channel = channel; Position = position; IsPrompt = isPrompt }

        let stopwatch = Stopwatch.StartNew()
        let mutable promptTime = TimeSpan.Zero
        let mutable position = startPosition
        let mutable token = tokens.[0]
        let mutable index = 0
        let mutable generated = 0
        let mutable stop = None
        let mutable pending = None
        let mutable fed = 0
        let mutable lastToken = token
        while stop.IsNone do
            if cancellation.IsCancellationRequested then
                stop <- Some Cancelled
                pending <- Some token
            elif position >= limit - 1 then
                stop <- Some(if limit < capacity then TokenLimit else ContextFull)
                pending <- Some token
            else
                let inPrompt = index < tokens.Length - 1
                let logits = engine.Forward(token, position, not inPrompt)
                history.Add token
                if index < tokens.Length then fed <- index + 1
                position <- position + 1
                if inPrompt then
                    index <- index + 1
                    let next = tokens.[index]
                    if settings.EchoPrompt then emit token next position true
                    token <- next
                    if index = tokens.Length - 1 then
                        promptTime <- stopwatch.Elapsed
                        stopwatch.Restart()
                else
                    let next =
                        match logits with
                        | ValueSome l -> sampler.Sample(l, history)
                        | ValueNone -> invalidOp "The forward pass returned no logits"
                    // From here on every input token has been fed.
                    index <- tokens.Length
                    generated <- generated + 1
                    if info.StopTokens.Contains next then
                        stop <- Some EndOfTurn
                        pending <- Some next
                    else
                        emit token next position false
                        lastToken <- next
                        token <- next
                        if generated >= settings.MaxNewTokens then
                            stop <- Some TokenLimit
                            pending <- Some token
        // Bytes of an incomplete UTF-8 sequence left at the end reach the stream as well.
        let tail = utf8.Flush()
        if tail.Length > 0 then
            (if channel = Thinking then thinking else answer).Append(tail) |> ignore
            onToken { Token = lastToken; Text = tail; Channel = channel; Position = position; IsPrompt = false }
        { Result =
            { Text = answer.ToString()
              Thinking = thinking.ToString()
              PromptTokens = tokens.Length
              GeneratedTokens = generated
              PromptTime = promptTime
              GenerationTime = stopwatch.Elapsed
              StopReason = stop.Value
              FinalPosition = position }
          Pending = pending
          Fed = fed }
