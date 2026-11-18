namespace Llama3.Core

/// Transformer families understood by the runtime. They share one decoder block;
/// the differences are captured by <see cref="ModelConfig"/> flags and optional tensors.
type Architecture =
    /// Karpathy llama2.c checkpoint (the format used by the original llama3.cu).
    | Llama2C
    /// GGUF <c>llama</c> with a byte-level BPE vocabulary: Llama 3.x, SmolLM… (SentencePiece-vocabulary
    /// GGUFs such as Llama 2 / TinyLlama are not supported yet).
    | Llama
    /// GGUF <c>qwen2</c>: Q/K/V projections carry biases.
    | Qwen2
    /// GGUF <c>qwen3</c>: per-head RMSNorm on Q and K, decoupled head_dim.
    | Qwen3
    /// GGUF <c>qwen3moe</c>: Qwen3 attention with a sparse mixture-of-experts FFN.
    | Qwen3Moe

/// How rotary embeddings pair the dimensions of a head.
type RopeMode =
    /// Adjacent pairs (2i, 2i+1) — GPT-J / Meta layout (ggml ROPE_TYPE_NORM).
    | Interleaved
    /// Split halves (i, i + d/2) — GPT-NeoX / HF <c>rotate_half</c> layout.
    | NeoX

/// Sparse mixture-of-experts hyper-parameters.
type MoeConfig =
    { ExpertCount: int
      /// Experts evaluated per token (top-k).
      ExpertsPerToken: int
      /// Hidden size of one expert's SwiGLU.
      ExpertHiddenDim: int
      /// Renormalise the top-k routing weights to sum to 1 (Qwen3 <c>norm_topk_prob</c>).
      NormalizeTopK: bool }

/// Hyper-parameters of a decoder-only transformer.
type ModelConfig =
    { Architecture: Architecture
      Dim: int
      HiddenDim: int
      NumLayers: int
      NumHeads: int
      NumKvHeads: int
      HeadDim: int
      VocabSize: int
      /// Context length the model was trained for.
      ContextLength: int
      RopeTheta: float32
      RopeMode: RopeMode
      NormEps: float32
      /// Present for sparse (MoE) models; HiddenDim then describes a dense FFN size from metadata only.
      Moe: MoeConfig option }

    member c.QDim = c.NumHeads * c.HeadDim
    member c.KvDim = c.NumKvHeads * c.HeadDim
    /// Query heads sharing one KV head (grouped-query attention).
    member c.KvMul = c.NumHeads / c.NumKvHeads

    member c.Validate() =
        let fail msg = invalidOp $"Invalid model config: %s{msg}"
        if c.Dim <= 0 || c.HiddenDim <= 0 || c.NumLayers <= 0 || c.HeadDim <= 0 || c.VocabSize <= 0 || c.ContextLength <= 0 then
            fail "non-positive dimensions"
        if c.NumHeads <= 0 || c.NumKvHeads <= 0 || c.NumHeads % c.NumKvHeads <> 0 then
            fail $"n_heads=%d{c.NumHeads} must be a multiple of n_kv_heads=%d{c.NumKvHeads}"
        if c.HeadDim % 2 <> 0 then fail $"head_dim=%d{c.HeadDim} must be even for RoPE"
        match c.Moe with
        | Some m when m.ExpertsPerToken <= 0 || m.ExpertsPerToken > m.ExpertCount ->
            fail $"experts per token (%d{m.ExpertsPerToken}) must be in 1..%d{m.ExpertCount}"
        | Some m when m.ExpertHiddenDim <= 0 -> fail "non-positive expert hidden size"
        | _ -> ()
        c
