namespace Llama3.Core

open System
open System.IO
open System.Collections.Generic
open System.Threading
open System.Threading.Tasks

/// Everything needed to run a model: weights, tokenizer and conversation conventions.
type LoadedModel =
    { Name: string
      SourcePath: string
      Model: HostModel
      Tokenizer: ITokenizer
      ChatTemplate: ChatTemplate option
      /// Prepend BOS when encoding a raw (non-chat) prompt.
      AddBos: bool
      /// Sampling any of these ends generation.
      StopTokens: Set<int>
      /// Memory mappings backing the tensors; dispose after every consumer of the weights.
      Resources: IDisposable list }

/// How tensor payloads are brought into memory.
type LoadOptions =
    { /// Map the files (llama.cpp default): zero-copy views, no 2 GiB tensor limit.
      UseMmap: bool
      /// With mmap: fault every page in at load time (llama.cpp MAP_POPULATE) instead of on first
      /// use. Disable only when the weights will not be used for inference (planning).
      Populate: bool
      /// Reads in flight (buffered loading, or populating a mapping).
      Concurrency: int }

    static member Default = { UseMmap = true; Populate = true; Concurrency = Environment.ProcessorCount }

/// Structural checks of a host model against its configuration: every mismatch is reported at
/// load time instead of surfacing later as an out-of-range access on some device.
module ModelValidation =

    let private fail (tensor: HostTensor) (expected: string) =
        raise (InvalidDataException $"Tensor '%s{tensor.Name}' has shape %A{tensor.Shape}, expected %s{expected}")

    /// A matrix with <paramref name="rows"/> outputs of <paramref name="cols"/> inputs.
    let matrix (t: HostTensor) (rows: int) (cols: int) =
        if t.Columns <> cols || t.Rows <> rows then fail t $"%d{rows}×%d{cols} (rows × columns)"

    let vector (t: HostTensor) (length: int) =
        if t.ElementCount <> int64 length then fail t $"a vector of %d{length}"

    /// Stacked experts: [cols; rows; experts].
    let experts (t: HostTensor) (rows: int) (cols: int) (count: int) =
        if t.Shape.Length <> 3 || t.Shape.[0] <> int64 cols || t.Shape.[1] <> int64 rows || t.Shape.[2] <> int64 count then
            fail t $"[%d{cols}; %d{rows}; %d{count}]"

    let model (m: HostModel) =
        let c = m.Config
        matrix m.TokenEmbedding c.VocabSize c.Dim
        if not (GgmlTypes.isDecodable m.TokenEmbedding.Type) then
            raise (NotSupportedException $"Token embedding uses %A{m.TokenEmbedding.Type}, which cannot be decoded")
        matrix m.Output c.VocabSize c.Dim
        vector m.OutputNorm c.Dim
        m.RopeFreqFactors |> Option.iter (fun f -> vector f (c.HeadDim / 2))
        if m.Layers.Length <> c.NumLayers then
            raise (InvalidDataException $"Model has %d{m.Layers.Length} layers, metadata says %d{c.NumLayers}")
        for l in m.Layers do
            let a = l.Attention
            vector a.Norm c.Dim
            matrix a.Wq c.QDim c.Dim
            matrix a.Wk c.KvDim c.Dim
            matrix a.Wv c.KvDim c.Dim
            matrix a.Wo c.Dim c.QDim
            a.Bq |> Option.iter (fun b -> vector b c.QDim)
            a.Bk |> Option.iter (fun b -> vector b c.KvDim)
            a.Bv |> Option.iter (fun b -> vector b c.KvDim)
            a.QNorm |> Option.iter (fun n -> vector n c.HeadDim)
            a.KNorm |> Option.iter (fun n -> vector n c.HeadDim)
            vector l.Ffn.Norm c.Dim
            match l.Ffn.Block, c.Moe with
            | Dense d, None ->
                matrix d.Gate c.HiddenDim c.Dim
                matrix d.Up c.HiddenDim c.Dim
                matrix d.Down c.Dim c.HiddenDim
            | Moe e, Some mc ->
                matrix e.Router mc.ExpertCount c.Dim
                experts e.GateExperts mc.ExpertHiddenDim c.Dim mc.ExpertCount
                experts e.UpExperts mc.ExpertHiddenDim c.Dim mc.ExpertCount
                experts e.DownExperts c.Dim mc.ExpertHiddenDim mc.ExpertCount
            | _ -> raise (InvalidDataException "Feed-forward blocks do not match the MoE configuration")
        m

module ModelLoader =

    // ------------------------------------------------------------------------------------------
    // llama2.c checkpoint (stories15M.bin & co.) — the format consumed by llama3.cu.
    // ------------------------------------------------------------------------------------------

    /// Reads a llama2.c checkpoint: 7 int32 header followed by float32 tensors grouped by kind.
    /// The expected file size is derived from the header and checked before any weight is read.
    let loadLlama2C (checkpointPath: string) (tokenizerPath: string) : LoadedModel =
        if not (File.Exists tokenizerPath) then
            raise (FileNotFoundException($"Tokenizer for llama2.c checkpoint not found: %s{tokenizerPath}", tokenizerPath))
        use stream = new FileStream(checkpointPath, FileMode.Open, FileAccess.Read, FileShare.Read, 1 <<< 20)
        use r = new BinaryReader(stream)
        if stream.Length < 28L then raise (InvalidDataException $"'%s{checkpointPath}' is too small to be a llama2.c checkpoint")
        let dim = r.ReadInt32()
        let hidden = r.ReadInt32()
        let nLayers = r.ReadInt32()
        let nHeads = r.ReadInt32()
        let nKvHeads = r.ReadInt32()
        let vocabRaw = r.ReadInt32()
        let seqLen = r.ReadInt32()
        if dim <= 0 || hidden <= 0 || nLayers <= 0 || nHeads <= 0 || nKvHeads <= 0 || vocabRaw = 0 || seqLen <= 0 || dim % nHeads <> 0 then
            raise (InvalidDataException $"'%s{checkpointPath}' is not a llama2.c checkpoint (implausible header)")
        // Negative vocab size signals an untied classifier.
        let shared = vocabRaw > 0
        let vocab = abs vocabRaw
        let headDim = dim / nHeads
        let kvDim = nKvHeads * headDim
        let floats =
            int64 vocab * int64 dim
            + int64 nLayers * (2L * int64 dim + 2L * int64 dim * int64 dim + 2L * int64 dim * int64 kvDim + 3L * int64 dim * int64 hidden)
            + int64 dim
            + int64 seqLen * int64 headDim
            + (if shared then 0L else int64 vocab * int64 dim)
        if 28L + floats * 4L <> stream.Length then
            raise (InvalidDataException $"'%s{checkpointPath}' has %d{stream.Length} bytes; its header implies %d{28L + floats * 4L}")
        let config =
            { Architecture = Llama2C
              Dim = dim
              HiddenDim = hidden
              NumLayers = nLayers
              NumHeads = nHeads
              NumKvHeads = nKvHeads
              HeadDim = headDim
              VocabSize = vocab
              ContextLength = seqLen
              RopeTheta = 10000.0f
              RopeMode = Interleaved
              NormEps = 1e-5f
              Moe = None }.Validate()

        let read (name: string) (cols: int) (rows: int) =
            let bytes = int64 cols * int64 rows * 4L
            if bytes > int64 Array.MaxLength then raise (NotSupportedException $"Tensor '%s{name}' exceeds the 2 GiB host tensor limit")
            let data = r.ReadBytes(int bytes)
            if int64 data.Length <> bytes then raise (EndOfStreamException $"Checkpoint truncated at '%s{name}'")
            let shape = if rows = 1 then [| int64 cols |] else [| int64 cols; int64 rows |]
            HostTensor(name, GgmlType.F32, shape, data)

        let perLayer (kind: string) (cols: int) (rows: int) =
            Array.init nLayers (fun l -> read $"blk.%d{l}.%s{kind}" cols rows)

        let embedding = read "token_embd.weight" dim vocab
        let attnNorm = perLayer "attn_norm.weight" dim 1
        let wq = perLayer "attn_q.weight" dim (nHeads * headDim)
        let wk = perLayer "attn_k.weight" dim kvDim
        let wv = perLayer "attn_v.weight" dim kvDim
        let wo = perLayer "attn_output.weight" (nHeads * headDim) dim
        let ffnNorm = perLayer "ffn_norm.weight" dim 1
        let w1 = perLayer "ffn_gate.weight" dim hidden
        let w2 = perLayer "ffn_down.weight" hidden dim
        let w3 = perLayer "ffn_up.weight" dim hidden
        let outputNorm = read "output_norm.weight" dim 1
        // Legacy freq_cis_real / freq_cis_imag tables (RoPE is computed on the fly).
        stream.Seek(int64 seqLen * int64 headDim * 4L, SeekOrigin.Current) |> ignore
        let output = if shared then embedding else read "output.weight" dim vocab

        let layers =
            Array.init nLayers (fun l ->
                { Attention =
                    { Norm = attnNorm.[l]
                      Wq = wq.[l]
                      Wk = wk.[l]
                      Wv = wv.[l]
                      Wo = wo.[l]
                      Bq = None
                      Bk = None
                      Bv = None
                      QNorm = None
                      KNorm = None }
                  Ffn = { Norm = ffnNorm.[l]; Block = Dense { Gate = w1.[l]; Up = w3.[l]; Down = w2.[l] } } })

        let model =
            ModelValidation.model
                { Config = config
                  TokenEmbedding = embedding
                  Layers = layers
                  OutputNorm = outputNorm
                  Output = output
                  RopeFreqFactors = None }
        { Name = Path.GetFileNameWithoutExtension checkpointPath
          SourcePath = checkpointPath
          Model = model
          Tokenizer = SentencePieceTokenizer.Load(tokenizerPath, vocab)
          ChatTemplate = None
          AddBos = true
          // llama3.cu stops when the model emits BOS (=1), which delimits stories.
          StopTokens = Set.singleton 1
          Resources = [] }

    // ------------------------------------------------------------------------------------------
    // GGUF (llama / qwen2 / qwen3 / qwen3moe)
    // ------------------------------------------------------------------------------------------

    let private architectureOf (name: string) =
        match name with
        | "llama" -> Llama
        | "qwen2" -> Qwen2
        | "qwen3" -> Qwen3
        | "qwen3moe" -> Qwen3Moe
        | other -> raise (NotSupportedException $"GGUF architecture '%s{other}' is not supported (llama, qwen2, qwen3, qwen3moe)")

    /// Hyper-parameters from GGUF metadata (no tensor payload is read).
    let configOf (gguf: GgufFile) : ModelConfig =
        let archName = gguf.GetString "general.architecture"
        let arch = architectureOf archName
        let key (k: string) = $"%s{archName}.%s{k}"
        let getInt k = gguf.GetInt(key k) |> int
        let tryInt k = gguf.TryGetInt(key k) |> Option.map int
        let dim = getInt "embedding_length"
        let nHeads = getInt "attention.head_count"
        if dim <= 0 || nHeads <= 0 then raise (InvalidDataException $"Invalid dimensions in '%s{gguf.Path}'")
        let nKvHeads = tryInt "attention.head_count_kv" |> Option.defaultValue nHeads
        let headDim = tryInt "attention.key_length" |> Option.defaultValue (dim / nHeads)
        let hidden = getInt "feed_forward_length"
        let embeddingInfo =
            gguf.TryFindTensor "token_embd.weight" |> Option.defaultWith (fun () -> raise (InvalidDataException "token_embd.weight missing"))
        if embeddingInfo.Shape.Length <> 2 then raise (InvalidDataException "token_embd.weight must be 2-D")
        { Architecture = arch
          Dim = dim
          HiddenDim = hidden
          NumLayers = getInt "block_count"
          NumHeads = nHeads
          NumKvHeads = nKvHeads
          HeadDim = headDim
          VocabSize = int embeddingInfo.Shape.[1]
          ContextLength = tryInt "context_length" |> Option.defaultValue 4096
          RopeTheta = gguf.TryGetFloat(key "rope.freq_base") |> Option.map float32 |> Option.defaultValue 10000.0f
          RopeMode = (match arch with Llama | Llama2C -> Interleaved | Qwen2 | Qwen3 | Qwen3Moe -> NeoX)
          NormEps = gguf.TryGetFloat(key "attention.layer_norm_rms_epsilon") |> Option.map float32 |> Option.defaultValue 1e-5f
          Moe =
            tryInt "expert_count"
            |> Option.filter (fun n -> n > 0)
            |> Option.map (fun experts ->
                { ExpertCount = experts
                  ExpertsPerToken = getInt "expert_used_count"
                  // Mixtral-style GGUFs omit it: experts then use the dense FFN size.
                  ExpertHiddenDim = tryInt "expert_feed_forward_length" |> Option.defaultValue hidden
                  // Qwen3-MoE / Mixtral renormalise the top-k weights; GGUF exposes it when it differs.
                  NormalizeTopK = gguf.TryGetBool(key "expert_weights_norm") |> Option.defaultValue true }) }
            .Validate()

    /// Names of every tensor a model of this configuration may use, in load order.
    let private tensorNames (config: ModelConfig) =
        [| yield "token_embd.weight"
           yield "output_norm.weight"
           yield "output.weight"
           yield "rope_freqs.weight"
           for l in 0 .. config.NumLayers - 1 do
               for t in [ "attn_norm.weight"; "attn_q.weight"; "attn_k.weight"; "attn_v.weight"; "attn_output.weight"
                          "attn_q.bias"; "attn_k.bias"; "attn_v.bias"; "attn_q_norm.weight"; "attn_k_norm.weight"
                          "ffn_norm.weight"; "ffn_gate.weight"; "ffn_up.weight"; "ffn_down.weight"
                          "ffn_gate_inp.weight"; "ffn_gate_exps.weight"; "ffn_up_exps.weight"; "ffn_down_exps.weight" ] do
                   yield $"blk.%d{l}.%s{t}" |]

    /// All files of a (possibly split) GGUF model, in split order. llama.cpp names splits
    /// "&lt;prefix&gt;-00001-of-0000N.gguf" (llama_split_path) and records split.count in each file.
    let splitFiles (path: string) : GgufFile[] =
        let first = GgufFile.Open path
        match first.TryGetInt "split.count" with
        | Some count when count > 1L ->
            let m = Text.RegularExpressions.Regex.Match(Path.GetFileName path, @"^(.*)-(\d{5})-of-(\d{5})\.gguf$")
            if not m.Success || int m.Groups.[2].Value <> 1 then
                raise (InvalidDataException $"'%s{path}' is split in %d{count} files: open the first one (<prefix>-00001-of-%05d{count}.gguf)")
            let dir = Path.GetDirectoryName(Path.GetFullPath path)
            Array.init (int count) (fun i ->
                if i = 0 then first
                else
                    let part = Path.Combine(dir, $"%s{m.Groups.[1].Value}-%05d{i + 1}-of-%05d{count}.gguf")
                    if not (File.Exists part) then raise (FileNotFoundException($"Missing GGUF split %d{i + 1}/%d{count}: %s{part}", part))
                    GgufFile.Open part)
        | _ -> [| first |]

    /// Tensor directory of a model across its splits (types and shapes only; used for planning).
    let tensorDirectory (files: GgufFile[]) (config: ModelConfig) : (GgufTensorInfo * int)[] =
        tensorNames config
        |> Array.choose (fun name ->
            files |> Array.tryFindIndex (fun f -> (f.TryFindTensor name).IsSome)
            |> Option.map (fun i -> (files.[i].TryFindTensor name).Value, i))

    /// Loads a GGUF model (single or split). With mmap every tensor is a view of its file and the
    /// kernel prefetches the files asynchronously; otherwise payloads are read with bounded
    /// concurrency through asynchronous positional I/O.
    let loadGguf (path: string) (options: LoadOptions) (cancellation: CancellationToken) : LoadedModel =
        let files = splitFiles path
        let gguf = files.[0]
        let config = configOf gguf
        // The tokenizer is validated before any weight is touched.
        let tokenizer = BpeTokenizer.FromGguf gguf
        let itok = tokenizer :> ITokenizer
        if itok.VocabSize < config.VocabSize then
            raise (InvalidDataException $"Tokenizer has %d{itok.VocabSize} tokens but the model emits %d{config.VocabSize} logits")
        let directory = tensorDirectory files config
        let resources = List<IDisposable>()
        try
            let loaded =
                if options.UseMmap then
                    let mapped = files |> Array.map (fun f -> let m = new MappedFile(f.Path) in resources.Add m; m)
                    for m in mapped do
                        if options.Populate then m.Populate options.Concurrency else m.Prefetch(0L, m.Length)
                    directory
                    |> Array.map (fun (info, file) ->
                        KeyValuePair(info.Name, HostTensor(info.Name, info.Type, info.Shape, MappedStorage(mapped.[file], info.FileOffset, info.SizeInBytes))))
                    |> Dictionary
                else
                    let handles =
                        files |> Array.map (fun f ->
                            let h = File.OpenHandle(f.Path, FileMode.Open, FileAccess.Read, FileShare.Read, FileOptions.RandomAccess ||| FileOptions.Asynchronous)
                            resources.Add h
                            h)
                    // FSharp.Core 11 Task.parallelLimit: bounded reads in flight, no blocked threads,
                    // input order preserved, siblings cancelled on the first failure.
                    let tensors =
                        directory
                        |> Seq.map (fun (info, file) -> fun (token: CancellationToken) -> files.[file].ReadTensorAsync(handles.[file], info, token))
                        |> Task.parallelLimit (max 1 options.Concurrency) cancellation
                        |> fun t -> t.GetAwaiter().GetResult()
                    // Handles are only needed while reading.
                    for h in handles do
                        h.Dispose()
                        resources.Remove h |> ignore
                    tensors |> Array.map (fun t -> KeyValuePair(t.Name, t)) |> Dictionary

            /// Takes a tensor out of the loaded set, so converted originals become garbage immediately.
            let take name =
                match loaded.TryGetValue name with
                | true, t ->
                    loaded.Remove name |> ignore
                    Some t
                | _ -> None
            let tensor name = take name |> Option.defaultWith (fun () -> raise (InvalidDataException $"Tensor '%s{name}' missing"))
            let matrix name = TensorConversion.toComputeFormat (tensor name)
            let vector name = TensorConversion.toSingleVector (tensor name)
            let optVector name = take name |> Option.map TensorConversion.toSingleVector

            let layers =
                Array.init config.NumLayers (fun l ->
                    let n (s: string) = $"blk.%d{l}.%s{s}"
                    let block =
                        match config.Moe with
                        | Some _ ->
                            Moe
                                { Router = matrix (n "ffn_gate_inp.weight")
                                  GateExperts = matrix (n "ffn_gate_exps.weight")
                                  UpExperts = matrix (n "ffn_up_exps.weight")
                                  DownExperts = matrix (n "ffn_down_exps.weight") }
                        | None ->
                            Dense
                                { Gate = matrix (n "ffn_gate.weight")
                                  Up = matrix (n "ffn_up.weight")
                                  Down = matrix (n "ffn_down.weight") }
                    { Attention =
                        { Norm = vector (n "attn_norm.weight")
                          Wq = matrix (n "attn_q.weight")
                          Wk = matrix (n "attn_k.weight")
                          Wv = matrix (n "attn_v.weight")
                          Wo = matrix (n "attn_output.weight")
                          Bq = optVector (n "attn_q.bias")
                          Bk = optVector (n "attn_k.bias")
                          Bv = optVector (n "attn_v.bias")
                          QNorm = optVector (n "attn_q_norm.weight")
                          KNorm = optVector (n "attn_k_norm.weight") }
                      Ffn = { Norm = vector (n "ffn_norm.weight"); Block = block } })

            let rawEmbedding = tensor "token_embd.weight"
            let embedding, output =
                match take "output.weight" with
                | Some o -> rawEmbedding, TensorConversion.toComputeFormat o
                | None ->
                    // Tied classifier: the same tensor must be consumable by matvec kernels.
                    let e = TensorConversion.toComputeFormat rawEmbedding
                    e, e
            let model =
                ModelValidation.model
                    { Config = config
                      TokenEmbedding = embedding
                      Layers = layers
                      OutputNorm = vector "output_norm.weight"
                      Output = output
                      RopeFreqFactors = optVector "rope_freqs.weight" }

            let chat = gguf.TryGetString "tokenizer.chat_template" |> Option.bind ChatTemplate.TryDetect
            let stopTokens =
                [ yield! Option.toList itok.EosId
                  yield! gguf.TryGetInt "tokenizer.ggml.eot_token_id" |> Option.map int |> Option.toList
                  for s in [ "<|im_end|>"; "<|endoftext|>"; "<|eot_id|>"; "<|end_of_text|>"; "<|eom_id|>" ] do
                      yield! itok.TryGetSpecial s |> Option.toList ]
                |> List.filter (fun t -> t >= 0 && t < config.VocabSize)
                |> Set.ofList
            let addBos =
                gguf.TryGetBool "tokenizer.ggml.add_bos_token"
                |> Option.defaultValue (config.Architecture = Llama)

            { Name = gguf.TryGetString "general.name" |> Option.defaultValue (Path.GetFileNameWithoutExtension path)
              SourcePath = path
              Model = model
              Tokenizer = tokenizer
              ChatTemplate = chat
              AddBos = addBos && itok.BosId.IsSome
              StopTokens = stopTokens
              Resources = List.ofSeq resources }
        with _ ->
            for r in resources do r.Dispose()
            reraise ()

    /// True when the file starts with the GGUF magic.
    let isGguf (path: string) =
        use s = File.OpenRead path
        let header = Array.zeroCreate<byte> 4
        s.Read(header, 0, 4) = 4 && header = "GGUF"B

    let private tokenizerFor (path: string) (tokenizerPath: string option) =
        tokenizerPath |> Option.defaultWith (fun () -> Path.Combine(Path.GetDirectoryName(Path.GetFullPath path), "tokenizer.bin"))

    /// Picks the loader from the file content (GGUF magic) — anything else must be a llama2.c
    /// checkpoint, whose header and size are validated.
    let load (path: string) (tokenizerPath: string option) (options: LoadOptions) : LoadedModel =
        if not (File.Exists path) then raise (FileNotFoundException($"Model file not found: %s{path}", path))
        if isGguf path then loadGguf path options CancellationToken.None
        else loadLlama2C path (tokenizerFor path tokenizerPath)
