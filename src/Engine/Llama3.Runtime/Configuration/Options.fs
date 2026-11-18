namespace Llama3

open Llama3.Backend.Cpu

/// Where inference runs.
type ExecutionMode =
    /// Everything on the CPU; no GPU is touched.
    | CpuOnly
    /// Every layer computes on the GPU. Weights that do not fit in VRAM are streamed from pinned
    /// host memory (offloading) instead of falling back to the CPU.
    | GpuOnly
    /// Layers are split between GPU and CPU according to the memory budget (default).
    | Hybrid

/// A compute device.
type Device =
    | CPU
    | CUDA of deviceId: int

/// What Hybrid does with layers that do not fit in the GPU budget.
type OffloadStrategy =
    /// Run them on the CPU (activations cross PCIe once per token).
    | CpuCompute
    /// Run them on the GPU, paging their weights in from pinned RAM every token.
    | StreamWeights

/// Where the KV cache of GPU layers lives.
type KvCachePlacement =
    /// Next to the layer's compute (fastest).
    | WithCompute
    /// Always in host RAM; attention of GPU layers runs on the CPU (saves VRAM for weights).
    | HostMemory

type ModelOptions =
    { ExecutionMode: ExecutionMode
      /// Devices to use. Empty = every CUDA device of the machine; listing only <c>CPU</c> is
      /// equivalent to CpuOnly. Layers are pipelined across the listed GPUs in order.
      Devices: Device list
      /// When the requested GPU is missing or too small, run on the CPU instead of failing.
      UseCpuFallback: bool
      /// Upper bound on VRAM used per GPU for weights, KV cache and buffers.
      GpuMemoryLimitMB: int option
      /// VRAM left free per GPU for the CUDA context, kernel images and other processes
      /// (None = max(256 MiB, 1/64 of the device's memory)).
      GpuReserveMB: int option
      /// CPU worker threads (0 = physical cores).
      CpuThreads: int
      /// Hybrid: force this many layers onto the GPU (None = as many as fit).
      GpuLayers: int option
      Offload: OffloadStrategy
      KvCache: KvCachePlacement
      /// KV-cache capacity in tokens (None = min(model context, 4096)).
      ContextLength: int option
      /// CPU integer dot products for quantized weights (faster, slightly less precise).
      QuantizedCpuActivations: bool
      /// CPU kernel implementation; None = autotuned per format on this machine.
      CpuIsa: CpuIsa option
      /// Tokenizer for llama2.c checkpoints (default: tokenizer.bin next to the model).
      TokenizerPath: string option
      /// Map the model file instead of reading it (llama.cpp default): instant start, the page
      /// cache is shared with other processes, no 2 GiB limit per tensor.
      UseMmap: bool
      /// Tensor reads in flight when not mapping (None = one per logical processor).
      LoadConcurrency: int option }

    static member Default =
        { ExecutionMode = Hybrid
          Devices = []
          UseCpuFallback = true
          GpuMemoryLimitMB = None
          GpuReserveMB = None
          CpuThreads = 0
          GpuLayers = None
          Offload = CpuCompute
          KvCache = WithCompute
          ContextLength = None
          QuantizedCpuActivations = true
          CpuIsa = None
          TokenizerPath = None
          UseMmap = true
          LoadConcurrency = None }
