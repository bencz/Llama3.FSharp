/// Console rendering of streamed tokens and statistics.
module Llama3.Cli.Output

open System
open System.IO
open Llama3
open Llama3.Core

/// Writes streamed tokens; reasoning is rendered dimmed.
type TokenPrinter(output: TextWriter) =
    let mutable thinking = false
    let colors = not Console.IsOutputRedirected

    member _.Print(t: GeneratedToken) =
        let isThinking = t.Channel = Thinking
        if isThinking <> thinking && colors then output.Write(if isThinking then "\u001b[2m" else "\u001b[0m")
        thinking <- isThinking
        output.Write t.Text
        output.Flush()

    member _.Finish() =
        if thinking && colors then output.Write "\u001b[0m"
        thinking <- false
        output.WriteLine()
        output.Flush()

let reportStats (model: Llama3Model) (r: GenerationResult) =
    match model.Config.Architecture with
    | Llama2C ->
        // Same line as llama3.cu (token count includes BOS).
        let elapsed = r.PromptTime + r.GenerationTime
        let total = r.FinalPosition + 1
        let rate = if elapsed.TotalSeconds > 0.0 then float (total - 2) / elapsed.TotalSeconds else 0.0
        eprintfn $"Token count: %d{total}, elapsed: %f{elapsed.TotalSeconds}s, %d{int rate} tokens/s"
    | _ ->
        eprintfn
            $"[prompt %d{r.PromptTokens} tok, %.1f{r.PromptTokensPerSecond} tok/s | generated %d{r.GeneratedTokens} tok, %.2f{r.TokensPerSecond} tok/s | %A{r.StopReason}]"

let reportProfile (profile: ForwardProfile) =
    let rows = profile.PerToken()
    let total = rows |> List.sumBy (fun (_, _, ms) -> ms)
    eprintfn $"profile over %d{profile.Tokens} forward passes (ms per token, %.2f{total} total):"
    for phase, device, ms in rows do
        eprintfn $"  %-10s{string phase} %-55s{device} %8.3f{ms}  %5.1f{100.0 * ms / total}%%"
