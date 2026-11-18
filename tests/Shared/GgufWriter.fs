/// GGUF v3 writer for test fixtures: serialises a <see cref="HostModel"/> with a byte-level BPE
/// vocabulary and a ChatML template, optionally split llama.cpp-style across several files.
module Llama3.Tests.GgufWriter

open System
open System.IO
open System.Text
open Llama3.Core

type MetaValue =
    | U32 of uint32
    | F32 of float32
    | Bool of bool
    | Str of string
    | Strs of string[]
    | I32s of int[]

let private alignment = 32L

let private writeString (w: BinaryWriter) (s: string) =
    let b = Encoding.UTF8.GetBytes s
    w.Write(uint64 b.Length)
    w.Write b

let private writeValue (w: BinaryWriter) (value: MetaValue) =
    match value with
    | U32 v ->
        w.Write 4u
        w.Write v
    | F32 v ->
        w.Write 6u
        w.Write v
    | Bool v ->
        w.Write 7u
        w.Write(if v then 1uy else 0uy)
    | Str s ->
        w.Write 8u
        writeString w s
    | Strs items ->
        w.Write 9u
        w.Write 8u
        w.Write(uint64 items.Length)
        for s in items do writeString w s
    | I32s items ->
        w.Write 9u
        w.Write 5u
        w.Write(uint64 items.Length)
        for v in items do w.Write v

/// One GGUF file: metadata, tensor directory, aligned payloads.
let writeFile (path: string) (metadata: (string * MetaValue) list) (tensors: (string * HostTensor) list) =
    use stream = File.Create path
    use w = new BinaryWriter(stream)
    w.Write 0x46554747u
    w.Write 3u
    w.Write(uint64 tensors.Length)
    w.Write(uint64 metadata.Length)
    for (key, value) in metadata do
        writeString w key
        writeValue w value
    let mutable offset = 0L
    for (name, t) in tensors do
        writeString w name
        w.Write(uint32 t.Shape.Length)
        for d in t.Shape do w.Write(uint64 d)
        w.Write(uint32 t.Type)
        w.Write(uint64 offset)
        offset <- Numerics.alignUp (offset + t.SizeInBytes) alignment
    while stream.Position % alignment <> 0L do w.Write 0uy
    let dataStart = stream.Position
    for (_, t) in tensors do
        while (stream.Position - dataStart) % alignment <> 0L do w.Write 0uy
        w.Write(t.Bytes)

/// GPT-2 byte → printable character mapping (bytes_to_unicode).
let private byteSymbols =
    let printable b = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)
    let mutable extra = 0
    Array.init 256 (fun b ->
        if printable b then string (char b)
        else
            extra <- extra + 1
            string (char (255 + extra)))

let specials = [| "<|im_start|>"; "<|im_end|>"; "<think>"; "</think>" |]

/// Vocabulary of <paramref name="size"/> tokens: 256 byte tokens, a few merges, filler tokens,
/// then the ChatML / thinking specials in the last ids.
let vocabulary (size: int) =
    let merged = [| "Ġt"; "he"; "Ġthe"; "in" |]
    let fillers = size - 256 - merged.Length - specials.Length
    let tokens = Array.concat [ byteSymbols; merged; Array.init fillers (fun i -> $"<filler_%d{i}>"); specials ]
    let kinds =
        Array.init size (fun i ->
            if i >= size - specials.Length then int TokenKind.Control
            elif i >= 256 + merged.Length then int TokenKind.UserDefined
            else int TokenKind.Normal)
    let merges = [| "Ġ t"; "h e"; "Ġt he"; "i n" |]
    tokens, kinds, merges

let private tensorsOf (model: HostModel) : (string * HostTensor) list =
    [ yield "token_embd.weight", model.TokenEmbedding
      yield "output_norm.weight", model.OutputNorm
      if not model.TiedEmbeddings then yield "output.weight", model.Output
      match model.RopeFreqFactors with
      | Some f -> yield "rope_freqs.weight", f
      | None -> ()
      for l in 0 .. model.Layers.Length - 1 do
          let n (s: string) = $"blk.%d{l}.%s{s}"
          let layer = model.Layers.[l]
          let a = layer.Attention
          yield n "attn_norm.weight", a.Norm
          yield n "attn_q.weight", a.Wq
          yield n "attn_k.weight", a.Wk
          yield n "attn_v.weight", a.Wv
          yield n "attn_output.weight", a.Wo
          for (name, t) in [ "attn_q.bias", a.Bq; "attn_k.bias", a.Bk; "attn_v.bias", a.Bv; "attn_q_norm.weight", a.QNorm; "attn_k_norm.weight", a.KNorm ] do
              match t with
              | Some t -> yield n name, t
              | None -> ()
          yield n "ffn_norm.weight", layer.Ffn.Norm
          match layer.Ffn.Block with
          | Dense d ->
              yield n "ffn_gate.weight", d.Gate
              yield n "ffn_up.weight", d.Up
              yield n "ffn_down.weight", d.Down
          | Moe m ->
              yield n "ffn_gate_inp.weight", m.Router
              yield n "ffn_gate_exps.weight", m.GateExperts
              yield n "ffn_up_exps.weight", m.UpExperts
              yield n "ffn_down_exps.weight", m.DownExperts ]

let private metadataOf (model: HostModel) : (string * MetaValue) list =
    let c = model.Config
    let arch = if c.Moe.IsSome then "qwen3moe" else "qwen3"
    let k (s: string) = $"%s{arch}.%s{s}"
    let tokens, kinds, merges = vocabulary c.VocabSize
    [ yield "general.architecture", Str arch
      yield "general.name", Str "synthetic"
      yield k "embedding_length", U32(uint32 c.Dim)
      yield k "feed_forward_length", U32(uint32 c.HiddenDim)
      yield k "block_count", U32(uint32 c.NumLayers)
      yield k "context_length", U32(uint32 c.ContextLength)
      yield k "attention.head_count", U32(uint32 c.NumHeads)
      yield k "attention.head_count_kv", U32(uint32 c.NumKvHeads)
      yield k "attention.key_length", U32(uint32 c.HeadDim)
      yield k "rope.freq_base", F32 c.RopeTheta
      yield k "attention.layer_norm_rms_epsilon", F32 c.NormEps
      match c.Moe with
      | Some m ->
          yield k "expert_count", U32(uint32 m.ExpertCount)
          yield k "expert_used_count", U32(uint32 m.ExpertsPerToken)
          yield k "expert_feed_forward_length", U32(uint32 m.ExpertHiddenDim)
      | None -> ()
      yield "tokenizer.ggml.model", Str "gpt2"
      yield "tokenizer.ggml.pre", Str "qwen2"
      yield "tokenizer.ggml.tokens", Strs tokens
      yield "tokenizer.ggml.token_type", I32s kinds
      yield "tokenizer.ggml.merges", Strs merges
      yield "tokenizer.ggml.eos_token_id", U32(uint32 (c.VocabSize - specials.Length + 1))
      yield "tokenizer.chat_template", Str "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{% if enable_thinking %}<think>{% endif %}" ]

/// Writes <paramref name="model"/> to <paramref name="path"/> in <paramref name="parts"/> files
/// (llama.cpp split naming when parts > 1). Returns the path to open (the first part).
let writeModel (directory: string) (stem: string) (model: HostModel) (parts: int) : string =
    let tensors = tensorsOf model
    let metadata = metadataOf model
    if parts <= 1 then
        let path = Path.Combine(directory, $"%s{stem}.gguf")
        writeFile path metadata tensors
        path
    else
        let chunks = tensors |> List.splitInto parts
        chunks
        |> List.iteri (fun i chunk ->
            let split = [ "split.no", U32(uint32 i); "split.count", U32(uint32 parts); "split.tensors.count", U32(uint32 tensors.Length) ]
            let path = Path.Combine(directory, $"%s{stem}-%05d{i + 1}-of-%05d{parts}.gguf")
            writeFile path ((if i = 0 then metadata else []) @ split) chunk)
        Path.Combine(directory, $"%s{stem}-00001-of-%05d{parts}.gguf")
