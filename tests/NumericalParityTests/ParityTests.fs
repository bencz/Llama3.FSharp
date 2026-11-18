module Llama3.Tests.ParityTests

open System
open System.IO
open System.Text.Json
open Xunit
open Llama3
open Llama3.Backend.Cpu
open Llama3.Core
open Llama3.Tests.TestSupport

// Models are large and the GPU has limited memory: run these tests one at a time.
[<assembly: CollectionBehavior(DisableTestParallelization = true)>]
do ()

let private gpuAvailable = lazy (not (Llama3.devices ()).IsEmpty)

let private requireGpu () =
    if not gpuAvailable.Value then skipUnavailable "No CUDA device available"

let private requireModel (path: string option) =
    match path with
    | Some p -> p
    | None ->
        skipUnavailable "Model not found (set LLAMA3_MODELS_DIR)"
        ""

let private cpuExact = { ModelOptions.Default with ExecutionMode = CpuOnly; QuantizedCpuActivations = false }

/// Execution configurations exercised against llama3.cu's output.
let storiesConfigurations () =
    let data = TheoryData<string>()
    for name in [ "cpu"; "cpu-portable"; "gpu"; "hybrid-3-layers"; "gpu-kv-host"; "gpu-streamed"; "hybrid-streamed" ] do
        data.Add name
    data

let private storiesOptions (name: string) =
    let o = ModelOptions.Default
    match name with
    | "cpu" -> { o with ExecutionMode = CpuOnly }
    | "cpu-portable" -> { o with ExecutionMode = CpuOnly; CpuIsa = Some Portable }
    | "gpu" -> { o with ExecutionMode = GpuOnly; UseCpuFallback = false }
    | "hybrid-3-layers" -> { o with ExecutionMode = Hybrid; GpuLayers = Some 3; UseCpuFallback = false }
    | "gpu-kv-host" -> { o with ExecutionMode = GpuOnly; KvCache = HostMemory; UseCpuFallback = false }
    // 49 MiB: classifier + KV fit, layers do not -> all six layers stream through two slots.
    | "gpu-streamed" -> { o with ExecutionMode = GpuOnly; GpuMemoryLimitMB = Some 49; UseCpuFallback = false }
    | "hybrid-streamed" -> { o with ExecutionMode = Hybrid; Offload = StreamWeights; GpuMemoryLimitMB = Some 52; UseCpuFallback = false }
    | other -> failwith other

[<Theory; MemberData(nameof storiesConfigurations)>]
let ``stories15M reproduces llama3.cu output`` (configuration: string) =
    if configuration <> "cpu" && configuration <> "cpu-portable" then requireGpu ()
    use model = Llama3.load (storiesOptions configuration) storiesModel
    let text = model.Generate("I have a dream", GenerationSettings.Legacy)
    Assert.Equal(storiesGolden, text)
    if configuration.EndsWith "streamed" then
        Assert.Contains(model.Plan.Layers, fun l -> l.Weights = Llama3.Runtime.Streamed)

[<Fact>]
let ``stories15M logits agree between CPU and GPU at every position`` () =
    requireGpu ()
    use cpu = Llama3.load cpuExact storiesModel
    use gpu = Llama3.load { ModelOptions.Default with ExecutionMode = GpuOnly; UseCpuFallback = false } storiesModel
    let tokens = [| 1; 306; 505; 263; 12561; 29889; 940; 12561 |]
    for pos in 0 .. tokens.Length - 1 do
        let a = cpu.Forward(tokens.[pos], pos)
        let b = gpu.Forward(tokens.[pos], pos)
        Assert.True(relativeError b a < 1e-4f, $"position %d{pos}: %g{relativeError b a}")
        Assert.Equal(argmax a, argmax b)

// ------------------------------------------------------------------------------------------------
// Qwen3 against the independent NumPy float64 reference (tools/reference/qwen3_reference.py)
// ------------------------------------------------------------------------------------------------

type private Reference = { Tokens: int[]; TopIds: int[][]; TopLogits: float[][] }

let private reference =
    lazy
        (let path = Path.Combine(repoRoot, "tests", "NumericalParityTests", "data", "qwen3-0.6b-f32.reference.json")
         use doc = JsonDocument.Parse(File.ReadAllText path)
         let root = doc.RootElement
         let positions = [| for p in root.GetProperty("positions").EnumerateArray() -> p |]
         { Tokens = [| for t in root.GetProperty("tokens").EnumerateArray() -> t.GetInt32() |]
           TopIds = positions |> Array.map (fun p -> [| for i in p.GetProperty("top_ids").EnumerateArray() -> i.GetInt32() |])
           TopLogits = positions |> Array.map (fun p -> [| for v in p.GetProperty("top_logits").EnumerateArray() -> v.GetDouble() |]) })

let qwenConfigurations () =
    let data = TheoryData<string>()
    for name in [ "cpu"; "gpu"; "hybrid" ] do data.Add name
    data

[<Theory; MemberData(nameof qwenConfigurations)>]
let ``Qwen3-0.6B F32 matches the NumPy float64 reference`` (configuration: string) =
    let path = requireModel (qwen3F32 ())
    let options =
        match configuration with
        | "cpu" -> cpuExact
        | "gpu" ->
            requireGpu ()
            { ModelOptions.Default with ExecutionMode = GpuOnly; UseCpuFallback = false; ContextLength = Some 64 }
        | _ ->
            requireGpu ()
            { ModelOptions.Default with GpuLayers = Some 14; UseCpuFallback = false; ContextLength = Some 64; QuantizedCpuActivations = false }
    use model = Llama3.load options path
    let r = reference.Value
    for pos in 0 .. r.Tokens.Length - 1 do
        let logits = model.Forward(r.Tokens.[pos], pos)
        Assert.Equal(r.TopIds.[pos].[0], argmax logits)
        for k in 0 .. r.TopIds.[pos].Length - 1 do
            let expected = r.TopLogits.[pos].[k]
            let actual = float logits.[r.TopIds.[pos].[k]]
            Assert.True(abs (actual - expected) < 2e-3, $"%s{configuration} pos %d{pos} top%d{k}: %g{actual} vs %g{expected}")

[<Fact>]
let ``Qwen3 Q8_0: integer CPU path, exact CPU path and GPU agree`` () =
    let path = requireModel (qwen3Q8 ())
    requireGpu ()
    let r = reference.Value
    let run options =
        use model = Llama3.load { options with ContextLength = Some 64 } path
        [| for pos in 0 .. r.Tokens.Length - 1 -> model.Forward(r.Tokens.[pos], pos) |]
    let exact = run cpuExact
    let quantized = run { ModelOptions.Default with ExecutionMode = CpuOnly }
    let gpu = run { ModelOptions.Default with ExecutionMode = GpuOnly; UseCpuFallback = false }
    for pos in 0 .. r.Tokens.Length - 1 do
        // Same weights, float activations: only summation order differs.
        Assert.True(relativeError gpu.[pos] exact.[pos] < 1e-3f, $"gpu vs exact @%d{pos}: %g{relativeError gpu.[pos] exact.[pos]}")
        // int8 activations are approximate by design (as llama.cpp's Q8 activation path).
        // Position 0 is excluded from the bound: Qwen3's attention sink drives ~8e3 "massive
        // activations" into a few residual dims there, inflating the per-32 int8 scales right where
        // the last FFN cancels them (measured: 23% max-logit deviation, ranking intact). Later
        // positions attend to that sink, so they inherit a few percent (measured <= 6%).
        if pos > 0 then
            Assert.True(relativeError quantized.[pos] exact.[pos] < 1e-1f, $"int8 vs exact @%d{pos}: %g{relativeError quantized.[pos] exact.[pos]}")
    Assert.Equal(argmax exact.[0], argmax quantized.[0])
    let agreement = Array.map2 (fun a b -> if argmax a = argmax b then 1 else 0) quantized exact |> Array.sum
    Assert.True(agreement >= r.Tokens.Length - 2, $"int8 top-1 agreement %d{agreement}/%d{r.Tokens.Length}")

// ------------------------------------------------------------------------------------------------
// Chat sessions: incremental KV reuse must equal a full replay
// ------------------------------------------------------------------------------------------------

[<Fact>]
let ``chat continuation equals replaying the conversation from scratch`` () =
    let path = requireModel (qwen3Q8 ())
    let options = { ModelOptions.Default with ContextLength = Some 512 }
    use model = Llama3.load options path
    let settings = { GenerationSettings.Default with Sampling = SamplingSettings.Greedy; MaxNewTokens = 24 }
    // Session A runs both turns back to back (second turn reuses the KV cache).
    use a = model.CreateChat()
    let a1 = a.Send("Diga um número entre 1 e 10.", settings)
    let a2 = a.Send("Agora some 5 a ele.", settings)
    // Session B is interrupted by another generation, which forces a full replay on turn 2.
    use b = model.CreateChat()
    let b1 = b.Send("Diga um número entre 1 e 10.", settings)
    model.Generate("Unrelated prompt", settings) |> ignore
    let b2 = b.Send("Agora some 5 a ele.", settings)
    Assert.Equal(a1.Text, b1.Text)
    Assert.Equal(a2.Text, b2.Text)
    Assert.Equal<int seq>(a.History, b.History)

// ------------------------------------------------------------------------------------------------
// Large models (opt-in: LLAMA3_SLOW_TESTS=1, loads ~19 GB)
// ------------------------------------------------------------------------------------------------

[<Fact>]
let ``Qwen3-30B-A3B MoE answers a factual question with op-level hybrid placement`` () =
    if Environment.GetEnvironmentVariable "LLAMA3_SLOW_TESTS" <> "1" then Assert.Skip "Set LLAMA3_SLOW_TESTS=1 to run"
    let path = requireModel (tryModel [ "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf" ])
    use model = Llama3.load { ModelOptions.Default with ContextLength = Some 512 } path
    let settings = { GenerationSettings.Default with Sampling = SamplingSettings.Greedy; MaxNewTokens = 40 }
    let answer = model.Generate("Qual é a capital do Brasil? Responda com uma palavra.", settings)
    Assert.Contains("Bras", answer)
    if gpuAvailable.Value then
        // Attention of every layer on a GPU; at least some experts on the CPU (4 GB card).
        Assert.All(model.Plan.Layers, fun l -> Assert.NotEqual(Llama3.Runtime.OnCpu, l.Attention))
