module Llama3.Tests.KernelTests

open System
open Xunit
open Llama3.Core
open Llama3.Backend.Cpu
open Llama3.Tests.TestSupport

/// Every ISA implementation runnable on this machine (Portable always; AVX2 / NEON when present).
let isas () =
    let data = TheoryData<string>()
    for k in CpuKernels.available () do data.Add(sprintf "%A" k.Isa)
    data

let private kernelsFor (name: string) =
    CpuKernels.available () |> List.find (fun k -> sprintf "%A" k.Isa = name)

let quantCases () =
    let data = TheoryData<string, GgmlType, int>()
    for k in CpuKernels.available () do
        for t in [ GgmlType.Q8_0; GgmlType.Q4_K; GgmlType.Q6_K ] do
            for cols in [ 256; 768; 2560 ] do
                data.Add(sprintf "%A" k.Isa, t, cols)
    data

[<Fact>]
let ``binary16 decoding is bit-exact for every encoding`` () =
    for bits in 0 .. 65535 do
        let expected = float32 (BitConverter.UInt16BitsToHalf(uint16 bits))
        let actual = Numerics.halfToSingle (uint16 bits)
        if Single.IsNaN expected then Assert.True(Single.IsNaN actual)
        else Assert.Equal(BitConverter.SingleToInt32Bits expected, BitConverter.SingleToInt32Bits actual)

[<Fact>]
let ``Q8_0 block decoding follows the ggml definition`` () =
    let block = Array.zeroCreate<byte> 34
    let d = Numerics.singleToHalfBits 0.5f
    block.[0] <- byte d
    block.[1] <- byte (d >>> 8)
    for i in 0 .. 31 do block.[2 + i] <- byte (sbyte (i - 16))
    let out = Array.zeroCreate<float32> 32
    Dequantize.blockQ8_0 (ReadOnlySpan block) (Span out)
    Assert.Equal<float32>(Array.init 32 (fun i -> 0.5f * float32 (i - 16)), out)

[<Fact>]
let ``activation quantization error is bounded by half a step`` () =
    let rng = Random 5
    let x = randomVector rng 4096
    let q = QuantizedActivations(4096)
    q.Quantize(ReadOnlySpan x)
    for i in 0 .. x.Length - 1 do
        let d = q.Scales.[i / 32]
        Assert.True(abs (float32 q.Quants.[i] * d - x.[i]) <= d * 0.5f + 1e-7f)
    for g in 0 .. 4096 / 16 - 1 do
        let expected = Seq.sum (seq { for i in g * 16 .. g * 16 + 15 -> int q.Quants.[i] })
        Assert.Equal(expected, q.Sums16.[g])

[<Theory; MemberData(nameof isas)>]
let ``float dot product matches float64 for awkward lengths`` (isa: string) =
    let k = kernelsFor isa
    let rng = Random 3
    for n in [ 1; 3; 8; 31; 64; 100; 1023; 4096 ] do
        let a = randomVector rng n
        let b = randomVector rng n
        let expected = Array.fold2 (fun acc x y -> acc + float x * float y) 0.0 a b
        let actual = k.DotF32(ReadOnlySpan a, ReadOnlySpan b)
        Assert.True(abs (float actual - expected) < 1e-4 * (1.0 + abs expected), $"%s{isa} n=%d{n}: %g{actual} vs %g{expected}")

[<Theory; MemberData(nameof quantCases)>]
let ``integer dot products equal the exact dot of the decoded operands`` (isa: string, dtype: GgmlType, cols: int) =
    let k = kernelsFor isa
    let rng = Random(cols + int dtype)
    let rows = 7
    let w = randomTensor rng dtype rows cols
    let x = randomVector rng cols
    let xq = QuantizedActivations(cols)
    xq.Quantize(ReadOnlySpan x)
    // The integer path is exact on (decoded weights) * (dequantized activations).
    let xDecoded = Array.init cols (fun i -> float32 xq.Quants.[i] * xq.Scales.[i / 32])
    let expected = referenceMatVec w xDecoded
    let scratch = Array.zeroCreate<float32> 256
    let actual = Array.init rows (fun r -> CpuKernels.dotRow k w r x (ValueSome xq) (Span scratch))
    let err = relativeError actual expected
    Assert.True(err < 2e-5f, $"%s{isa} %A{dtype} cols=%d{cols}: relative error %g{err}")

[<Theory; MemberData(nameof quantCases)>]
let ``float reference path matches the float64 matvec`` (isa: string, dtype: GgmlType, cols: int) =
    let k = kernelsFor isa
    let rng = Random(cols * 3 + int dtype)
    let w = randomTensor rng dtype 5 cols
    let x = randomVector rng cols
    let expected = referenceMatVec w x
    let scratch = Array.zeroCreate<float32> 256
    let actual = Array.init 5 (fun r -> CpuKernels.dotRow k w r x ValueNone (Span scratch))
    Assert.True(relativeError actual expected < 2e-5f)

[<Theory; InlineData(true); InlineData(false)>]
let ``rope pairs dimensions according to the mode`` (neox: bool) =
    let headDim = 8
    let v = Array.init headDim float32
    let cos = Array.init 4 (fun j -> MathF.Cos(float32 (j + 1)))
    let sin = Array.init 4 (fun j -> MathF.Sin(float32 (j + 1)))
    let rotated = Array.copy v
    CpuKernels.applyRope (Span rotated) 1 headDim (if neox then NeoX else Interleaved) (ReadOnlySpan cos) (ReadOnlySpan sin)
    for j in 0 .. 3 do
        let i0, i1 = if neox then j, j + 4 else 2 * j, 2 * j + 1
        Assert.Equal(v.[i0] * cos.[j] - v.[i1] * sin.[j], rotated.[i0], 5)
        Assert.Equal(v.[i0] * sin.[j] + v.[i1] * cos.[j], rotated.[i1], 5)

// ------------------------------------------------------------------------------------------------
// bfloat16, worker pool, routing robustness
// ------------------------------------------------------------------------------------------------

[<Theory; InlineData(256); InlineData(270); InlineData(4096)>]
let ``bf16 rows match the float64 reference (vector body and scalar tail)`` (cols: int) =
    let rng = Random(cols)
    let w = randomTensor rng GgmlType.BF16 7 cols
    let x = randomVector rng cols
    let expected = referenceMatVec w x
    let k = CpuKernels.available () |> List.head
    let scratch = Array.zeroCreate<float32> 256
    let actual = Array.init 7 (fun r -> CpuKernels.dotRow k w r x ValueNone (Span scratch))
    Assert.True(relativeError actual expected < 2e-5f)

/// Counts executions per chunk; optionally throws on one chunk.
type private CountingJob(chunks: int, failOn: int) =
    let hits = Array.zeroCreate<int> chunks
    member _.Hits = hits
    interface IChunkJob with
        member _.Execute chunk =
            Threading.Interlocked.Increment(&hits.[chunk]) |> ignore
            if chunk = failOn then invalidOp "boom"

[<Fact>]
let ``worker pool runs every chunk exactly once across thousands of back-to-back jobs`` () =
    use pool = new CpuWorkerPool(max 2 (min 8 Environment.ProcessorCount))
    let rng = Random 5
    for _ in 1 .. 3000 do
        let n = rng.Next(1, 97)
        let job = CountingJob(n, -1)
        pool.Run(job, n)
        Assert.All(job.Hits, fun h -> Assert.Equal(1, h))

[<Fact>]
let ``worker pool surfaces chunk failures and stays usable`` () =
    use pool = new CpuWorkerPool(4)
    let failing = CountingJob(64, 17)
    let e = Assert.Throws<AggregateException>(fun () -> pool.Run(failing, 64))
    Assert.IsType<InvalidOperationException>(e.InnerException) |> ignore
    Assert.All(failing.Hits, fun h -> Assert.Equal(1, h))
    let next = CountingJob(64, -1)
    pool.Run(next, 64)
    Assert.All(next.Hits, fun h -> Assert.Equal(1, h))

[<Fact>]
let ``MoE routing ignores NaN logits and never yields NaN weights`` () =
    let experts = Array.zeroCreate<int> 2
    let weights = Array.zeroCreate<float32> 2
    MoeRouting.select (ReadOnlySpan<float32>([| nanf; 1.0f; 2.0f; nanf |])) true (Span experts) (Span weights)
    Assert.Equal<int[]>([| 2; 1 |], experts)
    Assert.True(weights |> Array.forall Single.IsFinite)
    Assert.Equal(1.0f, Array.sum weights, 5)
    MoeRouting.select (ReadOnlySpan<float32>([| nanf; nanf; nanf |])) true (Span experts) (Span weights)
    Assert.Equal<int[]>([| 0; 1 |], experts)
    Assert.True((weights = [| 0.5f; 0.5f |]), $"uniform weights expected, got %A{weights}")

