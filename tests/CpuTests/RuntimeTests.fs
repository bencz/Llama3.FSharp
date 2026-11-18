module Llama3.Tests.RuntimeTests

open System
open System.Collections.Generic
open Xunit
open Llama3
open Llama3.Core
open Llama3.Runtime

// ------------------------------------------------------------------------------------------------
// Sampler
// ------------------------------------------------------------------------------------------------

[<Fact>]
let ``argmax resolves ties to the lowest index like llama3.cu`` () =
    Assert.Equal(1, Sampler.Argmax(ReadOnlySpan [| 0.0f; 3.0f; 3.0f; -1.0f |]))

[<Fact>]
let ``top-k 1 and top-p near 0 degenerate to argmax`` () =
    let rng = Random 9
    let empty = List<int>() :> IReadOnlyList<int>
    for _ in 1 .. 50 do
        let logits = Array.init 1000 (fun _ -> float32 (rng.NextDouble() * 10.0))
        let best = Sampler.Argmax(ReadOnlySpan logits)
        let s1 = Sampler({ SamplingSettings.Greedy with Temperature = 1.0f; TopK = 1; Seed = Some 1 })
        let s2 = Sampler({ SamplingSettings.Greedy with Temperature = 1.0f; TopP = 1e-6f; Seed = Some 1 })
        Assert.Equal(best, s1.Sample(Array.copy logits, empty))
        Assert.Equal(best, s2.Sample(Array.copy logits, empty))

[<Fact>]
let ``seeded sampling is reproducible`` () =
    let logits = Array.init 500 (fun i -> float32 (i % 17))
    let run () =
        let s = Sampler({ SamplingSettings.Chat with Seed = Some 42 })
        [ for _ in 1 .. 20 -> s.Sample(Array.copy logits, List<int>()) ]
    Assert.Equal<int list>(run (), run ())

[<Fact>]
let ``repetition penalty demotes recent tokens`` () =
    let logits = [| 2.0f; 1.9f; 0.0f |]
    let s = Sampler({ SamplingSettings.Greedy with RepetitionPenalty = 1.5f })
    Assert.Equal(1, s.Sample(Array.copy logits, List<int>([ 0 ])))

// ------------------------------------------------------------------------------------------------
// Scheduler
// ------------------------------------------------------------------------------------------------

let private mib (n: int) = int64 n <<< 20

let private estimate =
    { AttentionWeights = Array.create 10 (mib 30)
      FfnWeights = Array.create 10 (mib 70)
      IsMoe = false
      KvPerLayer = mib 10
      Output = mib 200
      Workspace = mib 10
      Embedding = mib 200 }

let private fakeGpu (index: int) =
    { Kind = Cuda; Index = index; Name = $"Fake GPU %d{index}"; TotalMemory = mib 8192; Architecture = "SM_00" }

/// Free memory such that the budget (free − reserve) is exactly <paramref name="budget"/>.
let private free (budget: int64) = budget + MemoryManager.gpuReserveBytes (mib 8192) None

let private staging (layers: int[]) = if layers.Length = 0 then 0L else mib 200

let private planWith options est (budgetsMiB: int list) =
    let gpus = budgetsMiB |> List.mapi (fun i b -> fakeGpu i, free (mib b))
    match Scheduler.plan options est 1024 gpus staging with
    | Ok p -> p
    | Error e -> failwith e

let private plan options gpuFree =
    match Scheduler.plan options estimate 1024 [ fakeGpu 0, gpuFree ] staging with
    | Ok p -> p
    | Error e -> failwith e

[<Fact>]
let ``CpuOnly never touches the GPU`` () =
    let p = plan { ModelOptions.Default with ExecutionMode = CpuOnly } (free (mib 100000))
    Assert.False p.UsesGpu

[<Fact>]
let ``hybrid places everything on a large GPU`` () =
    let p = plan ModelOptions.Default (free (mib 100000))
    Assert.All(p.Layers, fun l -> Assert.Equal((OnGpu 0), l.Attention); Assert.Equal((OnGpu 0), l.Ffn))
    Assert.Equal((OnGpu 0), p.Output)

[<Fact>]
let ``hybrid puts the tail layers and the classifier on a small GPU`` () =
    // workspace 10 + output 200 + 5 × (100 + 10) = 760 MiB
    let p = plan ModelOptions.Default (free (mib 760))
    let onGpu = p.Layers |> Array.filter (fun l -> l.Attention = OnGpu 0 && l.Ffn = OnGpu 0) |> Array.map (fun l -> l.Layer)
    Assert.Equal<int[]>([| 5; 6; 7; 8; 9 |], onGpu)
    Assert.Equal((OnGpu 0), p.Output)
    Assert.True(p.GpuBytes.[0] <= p.GpuBudgets.[0])

[<Fact>]
let ``GpuOnly streams the layers that do not fit`` () =
    // fixed = 10 + 200 + 10×10 = 310; staging 200; 3 resident layers = 300 → 810 MiB
    let p = plan { ModelOptions.Default with ExecutionMode = GpuOnly } (free (mib 810))
    Assert.All(p.Layers, fun l -> Assert.Equal((OnGpu 0), l.Attention); Assert.Equal((OnGpu 0), l.Ffn))
    Assert.Equal(3, p.Layers |> Array.filter (fun l -> l.Weights = Resident) |> Array.length)
    Assert.Equal(mib 200, p.StagingBytes)

[<Fact>]
let ``MoE keeps all attention on the GPU and places experts of the tail layers`` () =
    // attention: workspace 10 + output 200 + 10 × (30 + 10 KV) = 610; + 3 expert blocks × 70 = 820 MiB
    let moe = { estimate with IsMoe = true }
    let p =
        match Scheduler.plan ModelOptions.Default moe 1024 [ fakeGpu 0, free (mib 820) ] staging with
        | Ok p -> p
        | Error e -> failwith e
    Assert.All(p.Layers, fun l -> Assert.Equal((OnGpu 0), l.Attention); Assert.Equal((OnGpu 0), l.KvCache); Assert.Equal(Resident, l.Weights))
    let gpuExperts = p.Layers |> Array.filter (fun l -> l.Ffn = OnGpu 0) |> Array.map (fun l -> l.Layer)
    Assert.Equal<int[]>([| 7; 8; 9 |], gpuExperts)
    Assert.Equal((OnGpu 0), p.Output)

[<Fact>]
let ``MoE never streams experts, even in GpuOnly`` () =
    let moe = { estimate with IsMoe = true }
    let p =
        match Scheduler.plan { ModelOptions.Default with ExecutionMode = GpuOnly } moe 1024 [ fakeGpu 0, free (mib 820) ] staging with
        | Ok p -> p
        | Error e -> failwith e
    Assert.All(p.Layers, fun l -> Assert.Equal(Resident, l.Weights))
    Assert.Contains(p.Layers, fun l -> l.Ffn = OnCpu)

[<Fact>]
let ``several GPUs take contiguous segments, the last one holding the tail and the classifier`` () =
    // Per GPU usable = budget − workspace 10. GPU1: 10 + output 200 + 3 layers × 110 = 540.
    // GPU0: 10 + 4 layers × 110 = 450. Remaining 3 layers on the CPU.
    let p = planWith ModelOptions.Default estimate [ 450; 540 ]
    let placement = p.Layers |> Array.map (fun l -> l.Attention)
    Assert.Equal<Placement[]>([| OnCpu; OnCpu; OnCpu; OnGpu 0; OnGpu 0; OnGpu 0; OnGpu 0; OnGpu 1; OnGpu 1; OnGpu 1 |], placement)
    Assert.Equal(OnGpu 1, p.Output)
    Assert.Equal<int list>([ 0; 1 ], p.UsedGpus)
    for g in 0 .. 1 do Assert.True(p.GpuBytes.[g] <= p.GpuBudgets.[g])

[<Fact>]
let ``GpuOnly spreads a model that fits across GPUs without streaming`` () =
    let p = planWith { ModelOptions.Default with ExecutionMode = GpuOnly } estimate [ 700; 700 ]
    Assert.All(p.Layers, fun l -> Assert.Equal(Resident, l.Weights); Assert.NotEqual(OnCpu, l.Attention))
    Assert.Equal(0L, p.StagingBytes)

[<Fact>]
let ``MoE spreads attention over GPUs and keeps experts next to their attention`` () =
    let moe = { estimate with IsMoe = true }
    // Attention units are 40 MiB (30 + 10 KV); expert blocks 70 MiB.
    // Usable 290/490 MiB, classifier 200 MiB on GPU1 → 290/290: attention splits 5/5 (H1: no
    // idle GPU), leaving 90 MiB per GPU = one expert block each, next to its attention.
    let p = planWith ModelOptions.Default moe [ 300; 500 ]
    Assert.All(p.Layers, fun l -> Assert.NotEqual(OnCpu, l.Attention); Assert.Equal(l.Attention, l.KvCache))
    Assert.Equal<Placement[]>(Array.init 10 (fun i -> OnGpu(i / 5)), p.Layers |> Array.map (fun l -> l.Attention))
    Assert.All(p.Layers |> Array.filter (fun l -> l.Ffn <> OnCpu), fun l -> Assert.Equal(l.Attention, l.Ffn))
    Assert.Equal<int list>([ 4; 9 ], p.Layers |> Array.filter (fun l -> l.Ffn <> OnCpu) |> Array.map (fun l -> l.Layer) |> List.ofArray)
    Assert.Equal(OnGpu 1, p.Output)
    for g in 0 .. 1 do Assert.True(p.GpuBytes.[g] <= p.GpuBudgets.[g], $"GPU%d{g} over budget")

[<Fact>]
let ``MoE experts that do not fit next to their attention use another GPU before the CPU`` () =
    let moe = { estimate with IsMoe = true; FfnWeights = Array.init 10 (fun i -> if i = 9 then mib 150 else mib 70) }
    // 290/290 after the classifier; attention 200/200 → 90/90 left. Layer 9 (150 MiB) fits nowhere;
    // layer 8 goes to GPU1 (own), layer 4 to GPU0 (own); no GPU has 70 MiB left after that.
    let p = planWith ModelOptions.Default moe [ 300; 500 ]
    Assert.Equal(OnCpu, p.Layers.[9].Ffn)
    Assert.Equal(OnGpu 1, p.Layers.[8].Ffn)
    Assert.Equal(OnGpu 0, p.Layers.[4].Ffn)
    for g in 0 .. 1 do Assert.True(p.GpuBytes.[g] <= p.GpuBudgets.[g], $"GPU%d{g} over budget")

[<Fact>]
let ``MoE expert crosses to the other GPU when its own is full`` () =
    // Unequal budgets: GPU0 usable 190 (attention 5×40 does not fit → proportional cut moves on).
    let moe = { estimate with IsMoe = true }
    let p = planWith ModelOptions.Default moe [ 400; 800 ]
    let onGpu = p.Layers |> Array.filter (fun l -> l.Ffn <> OnCpu)
    Assert.NotEmpty onGpu
    Assert.Contains(OnGpu 0, p.Layers |> Array.map (fun l -> l.Attention))
    Assert.Contains(OnGpu 1, p.Layers |> Array.map (fun l -> l.Attention))
    for g in 0 .. 1 do Assert.True(p.GpuBytes.[g] <= p.GpuBudgets.[g], $"GPU%d{g} over budget")

[<Fact>]
let ``forced zero GPU layers keeps every layer on the CPU`` () =
    let p = planWith { ModelOptions.Default with GpuLayers = Some 0 } estimate [ 4000 ]
    Assert.All(p.Layers, fun l -> Assert.Equal(OnCpu, l.Attention); Assert.Equal(OnCpu, l.Ffn))

[<Fact>]
let ``forced GPU layers beyond the budget are streamed with StreamWeights`` () =
    // 300 MiB usable − 200 classifier = 100: one 110 MiB layer does not fit; force 3.
    let p = planWith { ModelOptions.Default with GpuLayers = Some 3; Offload = StreamWeights } estimate [ 310 ]
    let gpuLayers = p.Layers |> Array.filter (fun l -> l.Attention <> OnCpu)
    Assert.Equal(3, gpuLayers.Length)
    Assert.All(gpuLayers, fun l -> Assert.Equal(Streamed, l.Weights))
    Assert.Equal(mib 200, p.StagingBytes)

[<Fact>]
let ``forced GPU layers beyond the budget are charged and reported`` () =
    let p = planWith { ModelOptions.Default with GpuLayers = Some 3 } estimate [ 310 ]
    Assert.Equal(3, p.Layers |> Array.filter (fun l -> l.Attention <> OnCpu) |> Array.length)
    Assert.True(p.GpuBytes.[0] > p.GpuBudgets.[0], "over-commit must be visible in GpuBytes")
    Assert.Contains(p.Notes, fun n -> n.Contains "exceeds")

[<Fact>]
let ``proportional partition gives every GPU a share`` () =
    let items = Array.create 12 10L
    let remaining = [| 100L; 200L; 100L |]
    let a = Scheduler.fillProportional items remaining
    Assert.Equal<int[]>([| 0; 0; 0; 1; 1; 1; 1; 1; 1; 2; 2; 2 |], a)
    Assert.Equal<int64[]>([| 70L; 140L; 70L |], remaining)

// ------------------------------------------------------------------------------------------------
// MoE routing
// ------------------------------------------------------------------------------------------------

[<Fact>]
let ``routing selects the top-k experts with renormalised softmax weights`` () =
    let logits = [| 0.5f; 2.0f; -1.0f; 3.0f; 2.0f; 0.0f |]
    let experts = Array.zeroCreate<int> 3
    let weights = Array.zeroCreate<float32> 3
    MoeRouting.select (ReadOnlySpan logits) true (Span experts) (Span weights)
    // 3.0 first, then the 2.0 tie resolved to the lower index (1 before 4).
    Assert.Equal<int[]>([| 3; 1; 4 |], experts)
    let e = logits |> Array.map (fun l -> exp (float l))
    let picked = [| e.[3]; e.[1]; e.[4] |]
    for i in 0 .. 2 do
        Assert.Equal(picked.[i] / Array.sum picked, float weights.[i], 6)

[<Fact>]
let ``routing without renormalisation keeps the full-softmax probabilities`` () =
    let rng = Random 4
    let logits = Array.init 128 (fun _ -> float32 (rng.NextDouble() * 6.0 - 3.0))
    let experts = Array.zeroCreate<int> 8
    let weights = Array.zeroCreate<float32> 8
    MoeRouting.select (ReadOnlySpan logits) false (Span experts) (Span weights)
    let z = logits |> Array.sumBy (fun l -> exp (float l))
    let expected = logits |> Array.mapi (fun i l -> i, exp (float l) / z) |> Array.sortBy (fun (i, p) -> -p, i) |> Array.take 8
    Assert.Equal<int[]>(expected |> Array.map fst, experts)
    for i in 0 .. 7 do Assert.Equal(snd expected.[i], float weights.[i], 6)

[<Fact>]
let ``host KV placement keeps caches in RAM`` () =
    let p = plan { ModelOptions.Default with KvCache = HostMemory } (free (mib 100000))
    Assert.All(p.Layers, fun l -> Assert.Equal(OnCpu, l.KvCache))

[<Fact>]
let ``missing GPU without fallback is an error`` () =
    let options = { ModelOptions.Default with ExecutionMode = GpuOnly; UseCpuFallback = false }
    match Scheduler.plan options estimate 1024 [] staging with
    | Error _ -> ()
    | Ok p -> failwith $"expected an error, got %A{p}"
