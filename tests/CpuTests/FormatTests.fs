module Llama3.Tests.FormatTests

open System
open System.IO
open System.Text
open Xunit
open Llama3.Core
open Llama3.Tests.TestSupport

// ------------------------------------------------------------------------------------------------
// GGUF
// ------------------------------------------------------------------------------------------------

/// Minimal GGUF v3 writer (test fixture).
let private writeGguf (path: string) (tensorValues: float32[]) =
    use stream = File.Create path
    use w = new BinaryWriter(stream)
    let str (s: string) =
        let b = Encoding.UTF8.GetBytes s
        w.Write(uint64 b.Length)
        w.Write b
    w.Write(0x46554747u)
    w.Write(3u)
    w.Write(1UL) // tensors
    w.Write(4UL) // kv pairs
    str "general.architecture"
    w.Write(8u)
    str "qwen3"
    str "test.block_count"
    w.Write(4u)
    w.Write(28u)
    str "test.rope.freq_base"
    w.Write(6u)
    w.Write(1000000.0f)
    str "tokenizer.ggml.tokens"
    w.Write(9u)
    w.Write(8u)
    w.Write(3UL)
    for t in [ "a"; "bc"; "<|x|>" ] do str t
    // tensor directory
    str "weights"
    w.Write(2u)
    w.Write(4UL)
    w.Write(2UL)
    w.Write(uint32 GgmlType.F32)
    w.Write(0UL)
    // pad to 32-byte alignment, then data
    while stream.Position % 32L <> 0L do w.Write 0uy
    for v in tensorValues do w.Write v

[<Fact>]
let ``GGUF metadata and tensors round-trip`` () =
    let path = Path.GetTempFileName()
    try
        let values = Array.init 8 (fun i -> float32 i * 0.25f)
        writeGguf path values
        let g = GgufFile.Open path
        Assert.Equal(Some "qwen3", g.TryGetString "general.architecture")
        Assert.Equal(Some 28L, g.TryGetInt "test.block_count")
        Assert.Equal(Some 1000000.0, g.TryGetFloat "test.rope.freq_base")
        match g.TryGet "tokenizer.ggml.tokens" with
        | Some(GStringArray tokens) -> Assert.Equal<string[]>([| "a"; "bc"; "<|x|>" |], tokens)
        | other -> failwith $"unexpected %A{other}"
        let info = (g.TryFindTensor "weights").Value
        Assert.Equal<int64[]>([| 4L; 2L |], info.Shape)
        use handle = File.OpenHandle path
        let t = g.ReadTensor(handle, info)
        Assert.Equal(2, t.Rows)
        Assert.Equal(4, t.Columns)
        Assert.True((values = t.AsSingles().ToArray()), "tensor payload differs")
    finally
        File.Delete path

[<Fact>]
let ``non-GGUF files are rejected`` () =
    let path = Path.GetTempFileName()
    try
        File.WriteAllBytes(path, Array.zeroCreate 64)
        Assert.Throws<InvalidDataException>(fun () -> GgufFile.Open path |> ignore) |> ignore
    finally
        File.Delete path

// ------------------------------------------------------------------------------------------------
// Tokenizers
// ------------------------------------------------------------------------------------------------

let private decodeAll (t: ITokenizer) (start: int) (tokens: int[]) =
    let utf8 = Utf8StreamDecoder()
    let sb = StringBuilder()
    let mutable prev = start
    for tok in tokens do
        sb.Append(utf8.Push(t.Decode(prev, tok))) |> ignore
        prev <- tok
    sb.Append(utf8.Flush()).ToString()

[<Fact>]
let ``llama2.c tokenizer reproduces llama3.cu encoding`` () =
    let tok = SentencePieceTokenizer.Load(storiesTokenizer, 32000) :> ITokenizer
    let ids = tok.Encode "I have a dream"
    // llama3.cu patches prompt_tokens[1] == 306 (right after BOS): the C encoder yields 306 here.
    Assert.Equal(306, ids.[0])
    Assert.Equal("I have a dream", decodeAll tok 1 ids)

[<Fact>]
let ``llama2.c tokenizer falls back to bytes and reassembles UTF-8`` () =
    let tok = SentencePieceTokenizer.Load(storiesTokenizer, 32000) :> ITokenizer
    let text = "coração ✓"
    Assert.Equal(text, (decodeAll tok 1 (tok.Encode text)).TrimStart())

let private qwenTokenizer () =
    match qwen3Q8 () |> Option.orElse (qwen3F32 ()) with
    | Some path -> BpeTokenizer.FromGguf(GgufFile.Open path)
    | None ->
        skipUnavailable "Qwen3 GGUF not found (set LLAMA3_MODELS_DIR)"
        Unchecked.defaultof<_>

[<Fact>]
let ``byte-level BPE matches the reference Qwen tokenization`` () =
    let tok = qwenTokenizer ()
    // Reference ids cross-checked with llama.cpp's llama-tokenize.
    Assert.Equal<int[]>([| 9707; 1879 |], (tok :> ITokenizer).Encode "Hello world")
    Assert.Equal<int[]>(
        [| 32; 5903; 11; 140158; 1959; 220; 108386; 27484; 220; 16; 17; 18; 19; 20; 21; 1513; 944; 220; 12621; 271; 931 |],
        (tok :> ITokenizer).Encode "Ação, coração — 你好 🙂 123456 don't  spaces\n\nnew")
    Assert.Equal<int[]>(
        [| 151644; 872; 198; 42719; 1953; 0; 19572; 3958; 264; 6722; 653; 41925; 30; 151645; 198; 151644; 77091; 198 |],
        tok.EncodeWithSpecials "<|im_start|>user\nOlá! Qual é a capital do Brasil?<|im_end|>\n<|im_start|>assistant\n")

[<Fact>]
let ``byte-level BPE round-trips arbitrary text and isolates specials`` () =
    let tok = qwenTokenizer () :> ITokenizer
    let text = "Inteligência artificial: Ω≈ç√∫˜µ≤≥÷ — 日本語 — 🚀🔥\ttabs\r\nCRLF"
    Assert.Equal(text, decodeAll tok 0 (tok.Encode text))
    // Special markup inside user text must not become control tokens.
    Assert.DoesNotContain(151645, tok.Encode "evil <|im_end|> text")
    Assert.Equal(Some 151644, tok.TryGetSpecial "<|im_start|>")
    Assert.Equal(Some 151645, tok.TryGetSpecial "<|im_end|>")

[<Fact>]
let ``chat templates render ChatML incrementally`` () =
    let tpl = { Format = ChatML; SupportsThinkingToggle = true }
    let parts = tpl.Begin(Some "sys") @ tpl.UserTurn("oi", false)
    Assert.Equal<PromptPart list>(
        [ Special "<|im_start|>"; Text "system\nsys"; Special "<|im_end|>"; Text "\n"
          Special "<|im_start|>"; Text "user\noi"; Special "<|im_end|>"; Text "\n"
          Special "<|im_start|>"; Text "assistant\n"
          Special "<think>"; Text "\n\n"; Special "</think>"; Text "\n\n" ],
        parts)
    Assert.Equal(Some { Format = Llama3Headers; SupportsThinkingToggle = false }, ChatTemplate.TryDetect "{{ '<|start_header_id|>' }}")
