/// Every placement the scheduler can produce must compute the same function: synthetic dense and
/// MoE models are run under CPU, GPU and mixed (op-level) placements and compared with the
/// independent float64 reference forward pass.
module Llama3.Tests.PlacementTests

open System
open Xunit
open Llama3
open Llama3.Core
open Llama3.Backend.Cpu
open Llama3.Backend.Cuda
open Llama3.Runtime
open Llama3.Tests.TestSupport

let private gpuPlatform = lazy (CudaPlatform.TryCreate() |> Result.toOption)

let private tokens = [| 5; 17; 200; 3; 99; 42 |]

let private references = Collections.Concurrent.ConcurrentDictionary<struct (GgmlType * bool), HostModel * float[][]>()

let private modelAndReference (dtype: GgmlType) (moe: bool) =
    references.GetOrAdd(struct (dtype, moe), fun _ ->
        let model = ReferenceModel.synthetic (int dtype * 7 + (if moe then 1 else 0)) dtype moe
        model, ReferenceModel.forward model tokens)

let private layer i attention ffn weights kv =
    { Layer = i; Attention = attention; Ffn = ffn; Weights = weights; KvCache = kv }

/// Named placements over the 3-layer synthetic model.
let private g0 = OnGpu 0
let private g1 = OnGpu 1

let private placements: (string * (bool -> LayerPlan[] * Placement)) list =
    [ "cpu", fun _ -> Array.init 3 (fun i -> layer i OnCpu OnCpu Resident OnCpu), OnCpu
      "gpu", fun _ -> Array.init 3 (fun i -> layer i g0 g0 Resident g0), g0
      "attention-gpu-ffn-cpu", fun _ -> Array.init 3 (fun i -> layer i g0 OnCpu Resident g0), g0
      "attention-cpu-ffn-gpu", fun _ -> Array.init 3 (fun i -> layer i OnCpu g0 Resident OnCpu), OnCpu
      "alternating-layers", fun _ -> Array.init 3 (fun i -> let p = (if i % 2 = 0 then g0 else OnCpu) in layer i p p Resident p), OnCpu
      "gpu-kv-host", fun _ -> Array.init 3 (fun i -> layer i g0 g0 Resident OnCpu), g0
      // Streaming applies to dense layers only (the scheduler never streams experts).
      "gpu-streamed", fun moe -> Array.init 3 (fun i -> layer i g0 g0 (if i = 0 || moe then Resident else Streamed) g0), g0
      // Two independent CUDA contexts (two GPUs, or two contexts on one GPU when only one exists):
      // CPU -> GPU0 -> GPU1 pipeline, activations crossing every device boundary.
      "two-gpus-pipeline", fun _ -> [| layer 0 OnCpu OnCpu Resident OnCpu; layer 1 g0 g0 Resident g0; layer 2 g1 g1 Resident g1 |], g1
      // Attention and FFN of one layer on different GPUs, KV on the attention GPU.
      "two-gpus-split-parts", fun _ -> Array.init 3 (fun i -> layer i g0 g1 Resident g0), g1 ]

let cases () =
    let data = TheoryData<string, GgmlType, bool>()
    for (name, _) in placements do
        for dtype in [ GgmlType.Q8_0; GgmlType.Q4_K ] do
            for moe in [ false; true ] do
                data.Add(name, dtype, moe)
    data

[<Theory; MemberData(nameof cases)>]
let ``every placement matches the float64 reference`` (placement: string, dtype: GgmlType, moe: bool) =
    let model, expected = modelAndReference dtype moe
    let layers, output = (placements |> List.find (fun (n, _) -> n = placement) |> snd) moe
    let plan =
        { Mode = Hybrid
          Gpus = []
          Capacity = 64
          Layers = layers
          Output = output
          GpuBudgets = [| 0L; 0L |]
          GpuBytes = [| 0L; 0L |]
          StagingBytes = 0L
          Notes = [] }
    let gpus =
        if plan.UsesGpu then
            match gpuPlatform.Value with
            | Some p ->
                // Real GPU i when present, otherwise an extra context on device 0.
                [| for i in 0 .. 1 -> if List.contains i plan.UsedGpus then Some(p.CreateBackend(min i (p.Devices.Length - 1)) :> IComputeBackend) else None |]
            | None ->
                skipUnavailable "No CUDA device available"
                [||]
        else [| None; None |]
    use cpu = new CpuBackend({ Threads = 4; QuantizedActivations = false; Isa = None })
    try
        use engine = ExecutionEngine.Create(model, plan, cpu, gpus)
        for pos in 0 .. tokens.Length - 1 do
            let actual = (engine.Forward(tokens.[pos], pos, true)).Value
            let reference = expected.[pos] |> Array.map float32
            let err = relativeError actual reference
            Assert.True(err < 1e-3f, $"%s{placement} %A{dtype} moe=%b{moe} @%d{pos}: relative error %g{err}")
            Assert.Equal(argmax reference, argmax actual)
    finally
        for g in gpus do g |> Option.iter (fun b -> b.Dispose())

/// Architecture variants x placements that stress them: interleaved RoPE with rope_freqs (Llama
/// 3.1), QKV biases with a tied F32 classifier (Qwen2), and 5-layer models whose streamed layers
/// wrap around the two slots (odd count, >= 3 streamed, prefetch across tokens).
let variantCases () =
    let data = TheoryData<string, string>()
    for variant in [ "llama31"; "qwen2-tied"; "five-layers" ] do
        for placement in [ "cpu"; "gpu"; "split"; "stream-all"; "stream-tail" ] do
            data.Add(variant, placement)
    data

let private variantSpec (variant: string) =
    let baseSpec = { ReferenceModel.SyntheticSpec.Create(91, GgmlType.Q8_0, false) with Layers = 5 }
    match variant with
    | "llama31" -> { baseSpec with RopeMode = Interleaved; FreqFactors = true; QkNorm = false }
    | "qwen2-tied" -> { baseSpec with QkvBias = true; QkNorm = false; TiedClassifier = true; DType = GgmlType.Q4_K }
    | _ -> { baseSpec with DType = GgmlType.Q6_K }

let private variantModels = Collections.Concurrent.ConcurrentDictionary<string, HostModel * float[][]>()

[<Theory; MemberData(nameof variantCases)>]
let ``architecture variants match the float64 reference on every placement`` (variant: string, placement: string) =
    let model, expected =
        variantModels.GetOrAdd(variant, fun v ->
            let m = ReferenceModel.syntheticWith (variantSpec v)
            m, ReferenceModel.forward m tokens)
    let n = model.Config.NumLayers
    let layers, output =
        match placement with
        | "cpu" -> Array.init n (fun i -> layer i OnCpu OnCpu Resident OnCpu), OnCpu
        | "gpu" -> Array.init n (fun i -> layer i g0 g0 Resident g0), g0
        | "split" -> Array.init n (fun i -> if i < 2 then layer i OnCpu OnCpu Resident OnCpu else layer i g0 g0 Resident g0), g0
        | "stream-all" -> Array.init n (fun i -> layer i g0 g0 Streamed g0), g0
        | _ -> Array.init n (fun i -> layer i g0 g0 (if i < 2 then Resident else Streamed) g0), g0
    let plan =
        { Mode = Hybrid
          Gpus = []
          Capacity = 64
          Layers = layers
          Output = output
          GpuBudgets = [| 0L |]
          GpuBytes = [| 0L |]
          StagingBytes = 0L
          Notes = [] }
    let gpus =
        if plan.UsesGpu then
            match gpuPlatform.Value with
            | Some p -> [| Some(p.CreateBackend 0 :> IComputeBackend) |]
            | None ->
                skipUnavailable "No CUDA device available"
                [||]
        else [| None |]
    use cpu = new CpuBackend({ Threads = 4; QuantizedActivations = false; Isa = None })
    try
        use engine = ExecutionEngine.Create(model, plan, cpu, gpus)
        for pos in 0 .. tokens.Length - 1 do
            let actual = (engine.Forward(tokens.[pos], pos, true)).Value
            let reference = expected.[pos] |> Array.map float32
            let err = relativeError actual reference
            Assert.True(err < 1e-3f, $"%s{variant}/%s{placement} @%d{pos}: relative error %g{err}")
            Assert.Equal(argmax reference, argmax actual)
    finally
        for g in gpus do g |> Option.iter (fun b -> b.Dispose())

[<Fact>]
let ``routing observer reports the experts each MoE layer selected`` () =
    let model, _ = modelAndReference GgmlType.Q8_0 true
    let plan =
        { Mode = CpuOnly; Gpus = []; Capacity = 64; Layers = Array.init 3 (fun i -> layer i OnCpu OnCpu Resident OnCpu)
          Output = OnCpu; GpuBudgets = [||]; GpuBytes = [||]; StagingBytes = 0L; Notes = [] }
    use cpu = new CpuBackend({ Threads = 2; QuantizedActivations = false; Isa = None })
    use engine = ExecutionEngine.Create(model, plan, cpu, [||])
    let seen = ResizeArray<int * int * int[] * float32[]>()
    engine.Transformer.AddObserver
        { new IForwardObserver with
            member _.CaptureHiddenStates = false
            member _.OnHiddenState(_, _, _) = ()
            member _.OnExpertRouting(layer, position, experts, weights) = seen.Add((layer, position, experts, weights))
            member _.OnLogits(_, _) = () }
    engine.Forward(7, 0, true) |> ignore
    Assert.Equal(3, seen.Count)
    for (_, position, experts, weights) in seen do
        Assert.Equal(0, position)
        Assert.Equal(2, experts.Length)
        Assert.Equal(1.0f, Array.sum weights, 5)
        Assert.NotEqual(experts.[0], experts.[1])
