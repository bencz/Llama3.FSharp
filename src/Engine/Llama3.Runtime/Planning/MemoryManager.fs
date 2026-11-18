namespace Llama3.Runtime

open System
open Llama3.Core

/// Byte requirements of every placeable component of a model at a given KV capacity.
type MemoryEstimate =
    { /// Attention weights of each decoder block (device layout).
      AttentionWeights: int64[]
      /// Feed-forward weights of each decoder block (dense FFN or router + all experts).
      FfnWeights: int64[]
      /// Sparse model: FFN parts dominate and are placed independently of attention.
      IsMoe: bool
      /// Keys + values of one layer at full capacity (float32).
      KvPerLayer: int64
      /// Final norm + classifier.
      Output: int64
      /// Activation buffers of one device workspace (incl. attention scores and logits).
      Workspace: int64
      /// Token embedding table (always host-resident).
      Embedding: int64 }

    member e.LayerCount = e.AttentionWeights.Length
    member e.LayerWeights(layer: int) = e.AttentionWeights.[layer] + e.FfnWeights.[layer]
    member e.TotalWeights = Array.sum e.AttentionWeights + Array.sum e.FfnWeights + e.Output

module MemoryManager =

    /// VRAM kept free for the CUDA context, kernel images, allocator slack and other processes:
    /// the explicit setting, else max(256 MiB, 1/64 of the device), since the context and allocator
    /// overhead grow with the device (more SMs, larger page tables).
    let gpuReserveBytes (total: int64) (reserveMB: int option) =
        match reserveMB with
        | Some mb when mb >= 0 -> int64 mb <<< 20
        | Some mb -> raise (ArgumentOutOfRangeException("GpuReserveMB", mb, "must not be negative"))
        | None -> max (256L <<< 20) (total / 64L)

    /// Granularity of device allocations (ILGPU/CUDA round small buffers up).
    let private allocationGranularity = 512L

    let private padded (bytes: int64) = Numerics.alignUp bytes allocationGranularity

    /// <paramref name="deviceBytes"/> gives the size of a tensor in the accelerator's storage layout.
    let estimate (deviceBytes: HostTensor -> int64) (model: HostModel) (capacity: int) : MemoryEstimate =
        let c = model.Config
        let floats (n: int) = padded (int64 n * 4L)
        let size (tensors: HostTensor list) = tensors |> List.sumBy (fun t -> padded (deviceBytes t))
        let moeScratch =
            match c.Moe with
            | Some m -> floats m.ExpertCount + int64 m.ExpertsPerToken * (2L * floats m.ExpertHiddenDim + floats c.Dim)
            | None -> 0L
        { AttentionWeights = model.Layers |> Array.map (fun l -> size l.Attention.Tensors)
          FfnWeights = model.Layers |> Array.map (fun l -> size l.Ffn.Tensors)
          IsMoe = c.Moe.IsSome
          KvPerLayer = 2L * padded (int64 capacity * int64 c.KvDim * 4L)
          Output = padded (deviceBytes model.OutputNorm) + padded (deviceBytes model.Output)
          Workspace =
            3L * floats c.Dim
            + 2L * floats c.QDim
            + 2L * floats c.KvDim
            + floats (c.NumHeads * capacity)
            + (if c.Moe.IsSome then 0L else 2L * floats c.HiddenDim)
            + moeScratch
            + floats c.VocabSize
            + (model.RopeFreqFactors |> Option.map (fun t -> padded t.SizeInBytes) |> Option.defaultValue 0L)
          Embedding = model.TokenEmbedding.SizeInBytes }

    /// Usable GPU bytes: free memory minus the reserve, capped by the user limit.
    let gpuBudget (free: int64) (total: int64) (reserveMB: int option) (limitMB: int option) =
        let available = max 0L (free - gpuReserveBytes total reserveMB)
        match limitMB with
        | Some mb -> min available (int64 mb <<< 20)
        | None -> available

    let formatBytes (bytes: int64) =
        let b = float bytes
        if b >= 1073741824.0 then $"%.2f{b / 1073741824.0} GiB"
        elif b >= 1048576.0 then $"%.1f{b / 1048576.0} MiB"
        elif b >= 1024.0 then $"%.1f{b / 1024.0} KiB"
        else $"%d{bytes} B"
