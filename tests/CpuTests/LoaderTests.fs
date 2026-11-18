module Llama3.Tests.LoaderTests

open System
open System.IO
open System.Text
open Xunit
open Llama3
open Llama3.Core
open Llama3.Backend.Cpu
open Llama3.Runtime
open Llama3.Tests.TestSupport
open Llama3.Tests.GgufWriter

let private tokens = [| 3; 77; 300; 12 |]

/// Fresh scratch directory, removed afterwards (mapped files are released before deletion).
let private inTempDirectory (f: string -> unit) =
    let dir = Directory.CreateTempSubdirectory("llama3-tests-").FullName
    try f dir
    finally Directory.Delete(dir, true)

/// Logits of a host model on the exact (float) CPU path.
let private cpuLogits (model: HostModel) =
    let n = model.Config.NumLayers
    let plan =
        { Mode = CpuOnly
          Gpus = []
          Capacity = 32
          Layers = Array.init n (fun i -> { Layer = i; Attention = OnCpu; Ffn = OnCpu; Weights = Resident; KvCache = OnCpu })
          Output = OnCpu
          GpuBudgets = Array.empty
          GpuBytes = Array.empty
          StagingBytes = 0L
          Notes = [] }
    use cpu = new CpuBackend({ Threads = 2; QuantizedActivations = false; Isa = None })
    use engine = ExecutionEngine.Create(model, plan, cpu, Array.empty)
    tokens |> Array.mapi (fun pos t -> Array.copy (engine.Forward(t, pos, true)).Value)

let loadCases () =
    let data = TheoryData<bool, int, bool>()
    for moe in [ false; true ] do
        for parts in [ 1; 3 ] do
            for mmap in [ true; false ] do
                data.Add(moe, parts, mmap)
    data

[<Theory; MemberData(nameof loadCases)>]
let ``GGUF models load identically through mmap, buffered reads and splits`` (moe: bool, parts: int, mmap: bool) =
    inTempDirectory (fun dir ->
        let spec = { ReferenceModel.SyntheticSpec.Create(23, GgmlType.Q4_K, moe) with QkvBias = not moe; TiedClassifier = moe }
        let model = ReferenceModel.syntheticWith spec
        let path = writeModel dir "synthetic" model parts
        let loaded = ModelLoader.load path None { UseMmap = mmap; Populate = mmap; Concurrency = 3 }
        try
            Assert.Equal(model.Config, loaded.Model.Config)
            Assert.Equal(mmap, not loaded.Resources.IsEmpty)
            Assert.Equal(model.TiedEmbeddings, loaded.Model.TiedEmbeddings)
            // Payloads are byte-identical to what was written.
            let original = model.Layers.[1].Attention.Wv
            let restored = loaded.Model.Layers.[1].Attention.Wv
            Assert.True(original.Bytes.SequenceEqual(restored.Bytes), "tensor payload differs")
            let expected = ReferenceModel.forward model tokens
            let actual = cpuLogits loaded.Model
            for pos in 0 .. tokens.Length - 1 do
                let reference = expected.[pos] |> Array.map float32
                Assert.True(relativeError actual.[pos] reference < 1e-3f, $"position %d{pos}")
            Assert.Equal(Some(model.Config.VocabSize - 3), loaded.Tokenizer.TryGetSpecial "<|im_end|>")
            Assert.True(loaded.ChatTemplate.IsSome, "ChatML template not detected")
        finally
            for r in loaded.Resources do r.Dispose())

[<Fact>]
let ``a split model must be opened through its first part and needs every part`` () =
    inTempDirectory (fun dir ->
        let model = ReferenceModel.synthetic 3 GgmlType.Q8_0 false
        let first = writeModel dir "split" model 3
        let second = first.Replace("-00001-of-", "-00002-of-")
        Assert.Throws<InvalidDataException>(fun () -> ModelLoader.load second None LoadOptions.Default |> ignore) |> ignore
        File.Delete(first.Replace("-00001-of-", "-00003-of-"))
        Assert.Throws<FileNotFoundException>(fun () -> ModelLoader.load first None LoadOptions.Default |> ignore) |> ignore)

/// Raw GGUF prefix (magic, version, counts) for hand-made malformed files.
let private header (w: BinaryWriter) (tensors: uint64) (kvs: uint64) =
    w.Write 0x46554747u
    w.Write 3u
    w.Write tensors
    w.Write kvs

let private malformed (name: string) (build: BinaryWriter -> unit) =
    inTempDirectory (fun dir ->
        let path = Path.Combine(dir, name + ".gguf")
        do
            use w = new BinaryWriter(File.Create path)
            build w
        Assert.Throws<InvalidDataException>(fun () -> GgufFile.Open path |> ignore) |> ignore)

[<Fact>]
let ``malformed GGUF files are rejected with InvalidDataException`` () =
    // A key length far beyond the file.
    malformed "huge-string" (fun w ->
        header w 0UL 1UL
        w.Write(1UL <<< 40))
    // An array count that cannot fit in the remaining bytes.
    malformed "huge-array" (fun w ->
        header w 0UL 1UL
        w.Write 1UL
        w.Write "k"B
        w.Write 9u
        w.Write 5u
        w.Write(1UL <<< 50))
    // Alignment must be a positive power of two.
    malformed "alignment" (fun w ->
        header w 0UL 1UL
        w.Write(uint64 "general.alignment".Length)
        w.Write(Encoding.ASCII.GetBytes "general.alignment")
        w.Write 4u
        w.Write 3u)
    // Unknown tensor type.
    malformed "tensor-type" (fun w ->
        header w 1UL 0UL
        w.Write 1UL
        w.Write "t"B
        w.Write 1u
        w.Write 32UL
        w.Write 99u
        w.Write 0UL)
    // Truncated mid-header.
    malformed "truncated" (fun w ->
        header w 2UL 3UL
        w.Write 5UL)

[<Fact>]
let ``tensor data beyond the end of the file is rejected`` () =
    inTempDirectory (fun dir ->
        let path = writeModel dir "cut" (ReferenceModel.synthetic 9 GgmlType.Q8_0 false) 1
        do
            use f = new FileStream(path, FileMode.Open)
            f.SetLength(f.Length / 2L)
        Assert.Throws<InvalidDataException>(fun () -> GgufFile.Open path |> ignore) |> ignore)
