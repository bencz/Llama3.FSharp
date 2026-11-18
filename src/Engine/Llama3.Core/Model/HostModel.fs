namespace Llama3.Core

/// Attention half of a decoder block (pre-norm, projections and optional Qwen-style extras).
type HostAttention =
    { Norm: HostTensor
      Wq: HostTensor
      Wk: HostTensor
      Wv: HostTensor
      Wo: HostTensor
      Bq: HostTensor option
      Bk: HostTensor option
      Bv: HostTensor option
      /// Qwen3 per-head RMSNorm weights (length head_dim).
      QNorm: HostTensor option
      KNorm: HostTensor option }

    member a.Tensors: HostTensor list =
        [ yield a.Norm
          yield a.Wq
          yield a.Wk
          yield a.Wv
          yield a.Wo
          yield! Option.toList a.Bq
          yield! Option.toList a.Bk
          yield! Option.toList a.Bv
          yield! Option.toList a.QNorm
          yield! Option.toList a.KNorm ]

/// SwiGLU feed-forward: down(silu(gate(x)) * up(x)).
type HostDenseFfn =
    { Gate: HostTensor
      Up: HostTensor
      Down: HostTensor }

/// Sparse mixture of experts. Expert tensors are 3-D in ggml order
/// <c>[cols, rows, experts]</c>, i.e. a stack of per-expert matrices whose rows are contiguous:
/// expert <c>e</c> owns rows <c>[e*rows, (e+1)*rows)</c> of the tensor viewed as 2-D.
type HostMoe =
    { /// Router (gate_inp): one logit per expert.
      Router: HostTensor
      GateExperts: HostTensor
      UpExperts: HostTensor
      DownExperts: HostTensor }

    member m.ExpertCount = int m.GateExperts.Shape.[2]
    /// Rows of one expert's gate/up projection (expert hidden size).
    member m.ExpertHidden = int m.GateExperts.Shape.[1]

type HostFfnBlock =
    | Dense of HostDenseFfn
    | Moe of HostMoe

/// Feed-forward half of a decoder block.
type HostFfn =
    { Norm: HostTensor
      Block: HostFfnBlock }

    member f.Tensors: HostTensor list =
        [ yield f.Norm
          match f.Block with
          | Dense d -> yield! [ d.Gate; d.Up; d.Down ]
          | Moe m -> yield! [ m.Router; m.GateExperts; m.UpExperts; m.DownExperts ] ]

/// Weights of one decoder block, in host memory, already in a compute format.
type HostLayer =
    { Attention: HostAttention
      Ffn: HostFfn }

    /// Every tensor of the block, in a stable order (packing and memory accounting).
    member l.Tensors: HostTensor list = l.Attention.Tensors @ l.Ffn.Tensors

    member l.SizeInBytes = l.Tensors |> List.sumBy (fun t -> t.SizeInBytes)

/// A complete model in host memory.
type HostModel =
    { Config: ModelConfig
      /// Token embedding table; rows are looked up on the host every step.
      TokenEmbedding: HostTensor
      Layers: HostLayer[]
      OutputNorm: HostTensor
      /// Classifier; the same object as <c>TokenEmbedding</c> when embeddings are tied.
      Output: HostTensor
      /// Llama 3.1+ RoPE frequency correction factors (length head_dim/2).
      RopeFreqFactors: HostTensor option }

    member m.TiedEmbeddings = obj.ReferenceEquals(m.TokenEmbedding, m.Output)
