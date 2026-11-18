namespace Llama3.Core

open System

type ChatRole =
    | System
    | User
    | Assistant

/// A prompt fragment. Special parts are emitted as a single control token; text parts go through
/// the regular tokenizer and are never scanned for special-token markup (no prompt injection of
/// "&lt;|im_end|&gt;" through user content).
type PromptPart =
    | Special of string
    | Text of string

/// Conversation wire formats.
type ChatFormat =
    /// Qwen / ChatML: &lt;|im_start|&gt;role\n...&lt;|im_end|&gt;\n
    | ChatML
    /// Llama 3: &lt;|start_header_id|&gt;role&lt;|end_header_id|&gt;\n\n...&lt;|eot_id|&gt;
    | Llama3Headers

/// Incremental chat formatting: a session renders only the parts it has not fed to the model yet,
/// so the KV cache of earlier turns is reused verbatim.
type ChatTemplate =
    { Format: ChatFormat
      /// The template understands <c>enable_thinking</c> (Qwen3 hybrid reasoning models).
      SupportsThinkingToggle: bool }

    /// Parts that open a conversation (BOS, optional system message).
    member t.Begin(systemPrompt: string option) : PromptPart list =
        match t.Format with
        | ChatML ->
            match systemPrompt with
            | Some s -> [ Special "<|im_start|>"; Text("system\n" + s); Special "<|im_end|>"; Text "\n" ]
            | None -> []
        | Llama3Headers ->
            [ yield Special "<|begin_of_text|>"
              match systemPrompt with
              | Some s ->
                  yield! [ Special "<|start_header_id|>"; Text "system"; Special "<|end_header_id|>"; Text("\n\n" + s); Special "<|eot_id|>" ]
              | None -> () ]

    /// A user message followed by the assistant generation prompt.
    member t.UserTurn(content: string, enableThinking: bool) : PromptPart list =
        match t.Format with
        | ChatML ->
            [ yield Special "<|im_start|>"
              yield Text("user\n" + content)
              yield Special "<|im_end|>"
              yield Text "\n"
              yield Special "<|im_start|>"
              yield Text "assistant\n"
              // Qwen3 template with enable_thinking=false pre-fills an empty reasoning block.
              if t.SupportsThinkingToggle && not enableThinking then
                  yield! [ Special "<think>"; Text "\n\n"; Special "</think>"; Text "\n\n" ] ]
        | Llama3Headers ->
            [ Special "<|start_header_id|>"; Text "user"; Special "<|end_header_id|>"; Text("\n\n" + content); Special "<|eot_id|>"
              Special "<|start_header_id|>"; Text "assistant"; Special "<|end_header_id|>"; Text "\n\n" ]

    /// Parts that close an assistant reply (fed lazily before the next user turn).
    member t.EndAssistantTurn: PromptPart list =
        match t.Format with
        | ChatML -> [ Special "<|im_end|>"; Text "\n" ]
        | Llama3Headers -> [ Special "<|eot_id|>" ]

    /// Literal text of the token that terminates an assistant turn.
    member t.EndOfTurnMarker =
        match t.Format with
        | ChatML -> "<|im_end|>"
        | Llama3Headers -> "<|eot_id|>"

    /// Infers the format from the Jinja source stored in <c>tokenizer.chat_template</c>.
    static member TryDetect(jinja: string) : ChatTemplate option =
        let has (s: string) = jinja.Contains(s, StringComparison.Ordinal)
        let thinking = has "enable_thinking"
        if has "<|im_start|>" then Some { Format = ChatML; SupportsThinkingToggle = thinking }
        elif has "<|start_header_id|>" then Some { Format = Llama3Headers; SupportsThinkingToggle = false }
        else None

module PromptEncoding =

    /// Turns prompt parts into token ids. Adjacent text parts are tokenized together so the BPE
    /// boundaries match a tokenizer that sees the rendered string split only at special tokens.
    let encode (tokenizer: ITokenizer) (parts: PromptPart list) : int[] =
        let output = ResizeArray<int>()
        let pending = Text.StringBuilder()
        let flush () =
            if pending.Length > 0 then
                output.AddRange(tokenizer.Encode(pending.ToString()))
                pending.Clear() |> ignore
        for part in parts do
            match part with
            | Text s -> pending.Append s |> ignore
            | Special s ->
                flush ()
                match tokenizer.TryGetSpecial s with
                | Some id -> output.Add id
                | None -> raise (InvalidOperationException $"Special token '%s{s}' is not in the vocabulary")
        flush ()
        output.ToArray()
