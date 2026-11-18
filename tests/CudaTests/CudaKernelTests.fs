module Llama3.Tests.CudaKernelTests

open System
open Xunit
open Llama3.Core
open Llama3.Backend.Cpu
open Llama3.Backend.Cuda
open Llama3.Tests.TestSupport

/// One CUDA backend for the whole test class (kernel JIT is expensive).
type GpuFixture() =
    let platform = CudaPlatform.TryCreate() |> Result.toOption
    let backend = platform |> Option.map (fun p -> p.CreateBackend 0 :> IComputeBackend)
    member _.Backend = backend
    interface IDisposable with
        member _.Dispose() =
            backend |> Option.iter (fun b -> b.Dispose())
            platform |> Option.iter (fun p -> (p :> IDisposable).Dispose())

let private requireGpu (f: GpuFixture) =
    match f.Backend with
    | Some b -> b
    | None ->
        skipUnavailable "No CUDA device available"
        Unchecked.defaultof<_>

let private cpu = new CpuBackend({ Threads = 1; QuantizedActivations = false; Isa = None }) :> IComputeBackend

let private upload (b: IComputeBackend) (data: float32[]) =
    let buf = b.Allocate data.Length
    b.Upload(data, buf)
    buf

let private download (b: IComputeBackend) (buf: IBuffer) =
    let result = Array.zeroCreate<float32> buf.Length
    b.Download(buf, result)
    result

type CudaKernelTests(fixture: GpuFixture) =

    static member MatVecCases: TheoryData<GgmlType, int, int> =
        let data = TheoryData<GgmlType, int, int>()
        for t in [ GgmlType.F32; GgmlType.F16; GgmlType.BF16; GgmlType.Q8_0; GgmlType.Q4_K; GgmlType.Q6_K ] do
            // Row counts that are not multiples of the 8 rows per group exercise the tail; 16384
            // columns exceed the shared-memory staging of x (global-memory kernels).
            for rows, cols in [ (37, 768); (130, 2560); (8, 256); (9, 16384) ] do
                data.Add(t, rows, cols)
        // Q8_0 rows of 9 blocks: not a multiple of the 4 blocks a warp covers per step.
        data.Add(GgmlType.Q8_0, 21, 288)
        data.Add(GgmlType.F16, 5, 290)
        data.Add(GgmlType.BF16, 5, 290)
        data

    [<Theory; MemberData(nameof CudaKernelTests.MatVecCases)>]
    member _.``matvec matches the float64 reference`` (dtype: GgmlType, rows: int, cols: int) =
        let gpu = requireGpu fixture
        let rng = Random(rows * 31 + cols + int dtype)
        let w = randomTensor rng dtype rows cols
        let x = randomVector rng cols
        let expected = referenceMatVec w x
        use weight = gpu.LoadWeight w
        use xb = upload gpu x
        use out = gpu.Allocate rows
        gpu.MatVec(out, xb, weight)
        let actual = download gpu out
        let err = relativeError actual expected
        Assert.True(err < 2e-5f, $"%A{dtype} %d{rows}x%d{cols}: relative error %g{err}")

    [<Fact>]
    member _.``rmsnorm, head rmsnorm, swiglu and add match the CPU backend`` () =
        let gpu = requireGpu fixture
        let rng = Random 7
        let n = 1024
        let x = randomVector rng n
        let w = HostTensor.OfSingles("w", [| int64 n |], ReadOnlySpan(randomVector rng n))
        let hw = HostTensor.OfSingles("hw", [| 128L |], ReadOnlySpan(randomVector rng 128))
        let run (b: IComputeBackend) =
            use xb = upload b x
            use o = b.Allocate n
            use ww = b.LoadWeight w
            use hww = b.LoadWeight hw
            b.RmsNorm(o, xb, ww, 1e-6f)
            let norm = download b o
            b.HeadRmsNorm(xb, hww, 8, 128, 1e-6f)
            let heads = download b xb
            use up = upload b (Array.rev x)
            b.SwiGlu(xb, up)
            let glu = download b xb
            b.Add(xb, o)
            let added = download b xb
            norm, heads, glu, added
        let (n1, h1, g1, a1) = run cpu
        let (n2, h2, g2, a2) = run gpu
        for name, c, g in [ "rmsnorm", n1, n2; "headnorm", h1, h2; "swiglu", g1, g2; "add", a1, a2 ] do
            Assert.True(relativeError g c < 1e-5f, $"%s{name}: %g{relativeError g c}")

    [<Theory; InlineData(true); InlineData(false)>]
    member _.``rope matches the CPU backend`` (neox: bool) =
        let gpu = requireGpu fixture
        let rng = Random 11
        let spec b factors =
            { Mode = (if neox then NeoX else Interleaved)
              Theta = 1_000_000.0f
              HeadDim = 128
              NumHeads = 4
              NumKvHeads = 2
              FreqFactors = factors |> Option.map (fun t -> (b: IComputeBackend).LoadWeight t) }
        let q = randomVector rng 512
        let k = randomVector rng 256
        let factors = HostTensor.OfSingles("f", [| 64L |], ReadOnlySpan(Array.init 64 (fun i -> 1.0f + float32 i / 16.0f)))
        let run (b: IComputeBackend) (useFactors: bool) =
            use qb = upload b q
            use kb = upload b k
            b.Rope(qb, kb, 1234, spec b (if useFactors then Some factors else None))
            Array.append (download b qb) (download b kb)
        for useFactors in [ false; true ] do
            let err = relativeError (run gpu useFactors) (run cpu useFactors)
            Assert.True(err < 1e-4f, $"rope (factors=%b{useFactors}): %g{err}")

    [<Theory; InlineData(0); InlineData(1); InlineData(77); InlineData(300)>]
    member _.``grouped-query attention matches the CPU backend`` (position: int) =
        let gpu = requireGpu fixture
        let rng = Random(position + 3)
        let spec = { NumHeads = 8; NumKvHeads = 2; HeadDim = 64 }
        let capacity = 512
        let keys = Array.init (position + 1) (fun _ -> randomVector rng spec.KvDim)
        let values = Array.init (position + 1) (fun _ -> randomVector rng spec.KvDim)
        let q = randomVector rng (spec.NumHeads * spec.HeadDim)
        let run (b: IComputeBackend) =
            use cache = b.CreateKvCache(capacity, spec.KvDim)
            for t in 0 .. position do
                use kb = upload b keys.[t]
                use vb = upload b values.[t]
                b.StoreKv(cache, kb, vb, t)
            use qb = upload b q
            use out = b.Allocate q.Length
            use scratch = b.Allocate(spec.NumHeads * capacity)
            b.Attention(out, qb, cache, position, spec, scratch)
            download b out
        let err = relativeError (run gpu) (run cpu)
        Assert.True(err < 1e-5f, $"attention @%d{position}: %g{err}")

    interface IClassFixture<GpuFixture>
