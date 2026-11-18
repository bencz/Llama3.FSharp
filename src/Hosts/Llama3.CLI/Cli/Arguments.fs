/// Command-line parsing.
module Llama3.Cli.Arguments

open System
open System.IO
open Llama3
open Llama3.Backend.Cpu

type CliArgs =
    { Prompt: string option
      Model: string
      Options: ModelOptions
      MaxTokens: int option
      Steps: int option
      Temperature: float32 option
      TopK: int option
      TopP: float32 option
      MinP: float32 option
      RepeatPenalty: float32 option
      Seed: int option
      System: string option
      Think: bool
      Raw: bool
      Chat: bool
      PlanOnly: bool
      Profile: bool
      ListDevices: bool }

/// Outcome of parsing the command line.
type Command =
    | ShowHelp
    | Run of CliArgs

let usage =
    """llama3 — Llama/Qwen inference runtime in F# (CPU · CUDA · hybrid)

usage: llama3 [prompt] [options]

model
  -m, --model <path>        .gguf (llama, qwen2, qwen3) or llama2.c .bin   [models/stories15M.bin]
  -z, --tokenizer <path>    tokenizer.bin for llama2.c checkpoints
      --no-mmap             read the file into memory instead of mapping it
      --load-concurrency <n> tensor reads in flight with --no-mmap          [logical CPUs]
      --ctx <n>             KV-cache capacity in tokens                    [min(model, 4096)]

execution
      --mode cpu|gpu|hybrid execution mode                                 [hybrid]
      --gpus <list>         GPUs to pipeline across: all | none | 0,1,...  [all]
      --device <n>          single CUDA device (same as --gpus n)
      --gpu-mem <MB>        VRAM budget limit per GPU
      --gpu-reserve <MB>    VRAM left free per GPU             [max(256 MiB, 1/64 of the GPU)]
      --gpu-layers <n>      hybrid: number of layers placed on the GPU
      --offload cpu|stream  hybrid: what to do with layers that do not fit [cpu]
      --kv-cache gpu|host   KV cache placement for GPU layers              [gpu]
  -t, --threads <n>         CPU threads                                    [physical cores]
      --exact-cpu           float CPU dot products (no int8 activation quantization)
      --cpu-isa <isa>       force CPU kernels: avx512 | avx2 | neon | portable         [autotuned]
      --no-cpu-fallback     fail instead of running on the CPU when the GPU is unusable

generation
  -n, --max-tokens <n>      new tokens to generate
      --steps <n>           total sequence length incl. prompt (llama3.cu semantics)
      --temp <f>  --top-k <n>  --top-p <f>  --min-p <f>  --repeat-penalty <f>  --seed <n>
  -s, --system <text>       system prompt (chat models)
      --think               enable reasoning (Qwen3 hybrid models)
      --raw                 do not apply the chat template
      --chat                interactive multi-turn chat (the prompt, if any, is the first message)

info
      --plan                print the execution plan and exit
      --profile             time every forward phase per device (serialises GPU work)
      --devices             list GPUs and exit
  -h, --help
"""

/// Typed option values: every failure names the option and the offending text.
module private Value =
    let private invariant = Globalization.CultureInfo.InvariantCulture

    let int (name: string) (minimum: int) (text: string) : Result<int, string> =
        match Int32.TryParse(text, Globalization.NumberStyles.Integer, invariant) with
        | true, v when v >= minimum -> Ok v
        | true, v -> Error $"%s{name}: %d{v} is below the minimum of %d{minimum}"
        | _ -> Error $"%s{name}: '%s{text}' is not an integer"

    let float (name: string) (minimum: float32) (text: string) : Result<float32, string> =
        match Single.TryParse(text, Globalization.NumberStyles.Float, invariant) with
        | true, v when Single.IsFinite v && v >= minimum -> Ok v
        | true, _ -> Error $"%s{name}: '%s{text}' must be a finite number ≥ %g{minimum}"
        | _ -> Error $"%s{name}: '%s{text}' is not a number"

    let ints (name: string) (text: string) : Result<int list, string> =
        text.Split(',', StringSplitOptions.RemoveEmptyEntries ||| StringSplitOptions.TrimEntries)
        |> Array.fold
            (fun acc part ->
                match acc, int name 0 part with
                | Ok xs, Ok x -> Ok(x :: xs)
                | Error e, _ | _, Error e -> Error e)
            (Ok [])
        |> Result.map List.rev

let parse (argv: string[]) : Result<Command, string> =
    let defaults =
        { Prompt = None
          Model = Path.Combine("models", "stories15M.bin")
          Options = ModelOptions.Default
          MaxTokens = None
          Steps = None
          Temperature = None
          TopK = None
          TopP = None
          MinP = None
          RepeatPenalty = None
          Seed = None
          System = None
          Think = false
          Raw = false
          Chat = false
          PlanOnly = false
          Profile = false
          ListDevices = false }
    let mutable deviceFlags = []
    let rec go (args: string list) (a: CliArgs) : Result<Command, string> =
        let o = a.Options
        let next rest parsed (apply: 'T -> CliArgs) =
            match parsed with
            | Ok v -> go rest (apply v)
            | Error e -> Error e
        match args with
        | [] -> Ok(Run a)
        | ("-h" | "--help") :: _ -> Ok ShowHelp
        | ("-m" | "--model") :: v :: rest -> go rest { a with Model = v }
        | ("-z" | "--tokenizer") :: v :: rest -> go rest { a with Options = { o with TokenizerPath = Some v } }
        | "--ctx" :: v :: rest -> next rest (Value.int "--ctx" 1 v) (fun n -> { a with Options = { o with ContextLength = Some n } })
        | "--mode" :: v :: rest ->
            match v.ToLowerInvariant() with
            | "cpu" -> go rest { a with Options = { o with ExecutionMode = CpuOnly } }
            | "gpu" -> go rest { a with Options = { o with ExecutionMode = GpuOnly } }
            | "hybrid" -> go rest { a with Options = { o with ExecutionMode = Hybrid } }
            | other -> Error $"--mode: unknown mode '%s{other}' (cpu | gpu | hybrid)"
        | ("--device" as flag) :: v :: rest ->
            deviceFlags <- flag :: deviceFlags
            next rest (Value.int flag 0 v) (fun n -> { a with Options = { o with Devices = [ CUDA n ] } })
        | ("--gpus" as flag) :: v :: rest ->
            deviceFlags <- flag :: deviceFlags
            match v.ToLowerInvariant() with
            | "all" -> go rest { a with Options = { o with Devices = [] } }
            | "none" -> go rest { a with Options = { o with Devices = [ CPU ] } }
            | _ -> next rest (Value.ints flag v) (fun ids -> { a with Options = { o with Devices = ids |> List.map CUDA } })
        | "--no-mmap" :: rest -> go rest { a with Options = { o with UseMmap = false } }
        | "--load-concurrency" :: v :: rest ->
            next rest (Value.int "--load-concurrency" 1 v) (fun n -> { a with Options = { o with LoadConcurrency = Some n } })
        | "--gpu-mem" :: v :: rest -> next rest (Value.int "--gpu-mem" 1 v) (fun n -> { a with Options = { o with GpuMemoryLimitMB = Some n } })
        | "--gpu-reserve" :: v :: rest -> next rest (Value.int "--gpu-reserve" 0 v) (fun n -> { a with Options = { o with GpuReserveMB = Some n } })
        | "--gpu-layers" :: v :: rest -> next rest (Value.int "--gpu-layers" 0 v) (fun n -> { a with Options = { o with GpuLayers = Some n } })
        | "--offload" :: v :: rest ->
            match v with
            | "cpu" -> go rest { a with Options = { o with Offload = CpuCompute } }
            | "stream" -> go rest { a with Options = { o with Offload = StreamWeights } }
            | other -> Error $"--offload: unknown strategy '%s{other}' (cpu | stream)"
        | "--kv-cache" :: v :: rest ->
            match v with
            | "gpu" -> go rest { a with Options = { o with KvCache = WithCompute } }
            | "host" -> go rest { a with Options = { o with KvCache = HostMemory } }
            | other -> Error $"--kv-cache: unknown placement '%s{other}' (gpu | host)"
        | ("-t" | "--threads") :: v :: rest -> next rest (Value.int "--threads" 0 v) (fun n -> { a with Options = { o with CpuThreads = n } })
        | "--exact-cpu" :: rest -> go rest { a with Options = { o with QuantizedCpuActivations = false } }
        | "--cpu-isa" :: v :: rest ->
            match v.ToLowerInvariant() with
            | "avx2" -> go rest { a with Options = { o with CpuIsa = Some X86Avx2 } }
            | "avx512" | "avx512vnni" -> go rest { a with Options = { o with CpuIsa = Some X86Avx512Vnni } }
            | "neon" -> go rest { a with Options = { o with CpuIsa = Some ArmNeon } }
            | "portable" -> go rest { a with Options = { o with CpuIsa = Some Portable } }
            | other -> Error $"--cpu-isa: unknown ISA '%s{other}' (avx512 | avx2 | neon | portable)"
        | "--no-cpu-fallback" :: rest -> go rest { a with Options = { o with UseCpuFallback = false } }
        | ("-n" | "--max-tokens") :: v :: rest -> next rest (Value.int "--max-tokens" 1 v) (fun n -> { a with MaxTokens = Some n })
        | "--steps" :: v :: rest -> next rest (Value.int "--steps" 1 v) (fun n -> { a with Steps = Some n })
        | "--temp" :: v :: rest -> next rest (Value.float "--temp" 0.0f v) (fun f -> { a with Temperature = Some f })
        | "--top-k" :: v :: rest -> next rest (Value.int "--top-k" 0 v) (fun n -> { a with TopK = Some n })
        | "--top-p" :: v :: rest -> next rest (Value.float "--top-p" 0.0f v) (fun f -> { a with TopP = Some(min f 1.0f) })
        | "--min-p" :: v :: rest -> next rest (Value.float "--min-p" 0.0f v) (fun f -> { a with MinP = Some(min f 1.0f) })
        | "--repeat-penalty" :: v :: rest -> next rest (Value.float "--repeat-penalty" 0.0f v) (fun f -> { a with RepeatPenalty = Some f })
        | "--seed" :: v :: rest -> next rest (Value.int "--seed" Int32.MinValue v) (fun n -> { a with Seed = Some n })
        | ("-s" | "--system") :: v :: rest -> go rest { a with System = Some v }
        | "--think" :: rest -> go rest { a with Think = true }
        | "--raw" :: rest -> go rest { a with Raw = true }
        | "--chat" :: rest -> go rest { a with Chat = true }
        | "--plan" :: rest -> go rest { a with PlanOnly = true }
        | "--profile" :: rest -> go rest { a with Profile = true }
        | "--devices" :: rest -> go rest { a with ListDevices = true }
        | [ flag ] when flag.StartsWith "-" && flag.Length > 1 -> Error $"option '%s{flag}' is unknown or needs a value"
        | flag :: _ when flag.StartsWith "-" && flag.Length > 1 -> Error $"unknown option '%s{flag}'"
        | prompt :: rest when a.Prompt.IsNone -> go rest { a with Prompt = Some prompt }
        | extra :: _ -> Error $"unexpected argument '%s{extra}'"
    /// Combinations that cannot mean what the user intended.
    let conflicts (a: CliArgs) =
        let o = a.Options
        [ if List.length deviceFlags > 1 then "--device and --gpus select devices: give only one"
          if o.ExecutionMode = CpuOnly && not o.Devices.IsEmpty && o.Devices |> List.exists ((<>) CPU) then
              "--mode cpu contradicts the GPUs selected with --device/--gpus"
          if o.ExecutionMode = GpuOnly && o.Devices = [ CPU ] then "--mode gpu contradicts --gpus none"
          if a.Raw && a.Chat then "--raw disables the chat template that --chat needs"
          if a.PlanOnly && (a.Chat || a.Profile) then "--plan exits before generating: drop --chat/--profile"
          if o.GpuLayers.IsSome && o.ExecutionMode <> Hybrid then "--gpu-layers only applies to --mode hybrid"
          if o.LoadConcurrency.IsSome && o.UseMmap then "--load-concurrency only applies with --no-mmap" ]
    match go (List.ofArray argv) defaults with
    | Ok(Run a) ->
        match conflicts a with
        | [] -> Ok(Run a)
        | problems -> Error(String.Join("; ", problems))
    | other -> other
