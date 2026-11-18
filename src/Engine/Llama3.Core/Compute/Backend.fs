namespace Llama3.Core

open System

/// Family of a compute device. New vendors (ROCm/HIP, OpenCL, Metal…) extend this union and
/// provide an <see cref="IComputeBackend"/>; the transformer itself never changes.
type DeviceKind =
    | Cpu
    | Cuda
    | OpenCL

/// A float32 activation vector resident on one device.
type IBuffer =
    inherit IDisposable
    abstract Length: int

/// A weight tensor resident on (or visible to) one device, in a compute format.
type IWeight =
    inherit IDisposable
    abstract Name: string
    abstract Type: GgmlType
    abstract Rows: int
    abstract Columns: int
    abstract SizeInBytes: int64

/// Keys and values of one layer for every position of a sequence.
/// Backends may implement any layout (contiguous, paged, reduced precision…) behind this handle.
type IKvCache =
    inherit IDisposable
    abstract Capacity: int
    abstract KvDim: int
    abstract SizeInBytes: int64

type RopeSpec =
    { Mode: RopeMode
      Theta: float32
      HeadDim: int
      NumHeads: int
      NumKvHeads: int
      /// Per-pair divisors (Llama 3.1+ long-context scaling), F32 vector of length HeadDim/2.
      FreqFactors: IWeight option }

type AttentionSpec =
    { NumHeads: int
      NumKvHeads: int
      HeadDim: int }

    member s.KvMul = s.NumHeads / s.NumKvHeads
    member s.KvDim = s.NumKvHeads * s.HeadDim

/// Device memory status in bytes.
type MemoryInfo = { Free: int64; Total: int64 }

/// The operations a device must provide to run the decoder. All calls are enqueued in order on the
/// device's compute stream; only <c>Download</c> and <c>Synchronize</c> block the caller.
type IComputeBackend =
    inherit IDisposable
    abstract Name: string
    abstract Kind: DeviceKind
    abstract DeviceIndex: int
    abstract MemoryInfo: MemoryInfo option

    abstract Allocate: length: int -> IBuffer
    abstract Upload: source: float32[] * destination: IBuffer -> unit
    abstract Download: source: IBuffer * destination: float32[] -> unit
    abstract LoadWeight: tensor: HostTensor -> IWeight
    /// Non-owning view of rows [firstRow, firstRow + rowCount) of a loaded weight (MoE experts are
    /// row ranges of one stacked tensor). Disposing the view does not release the parent.
    abstract SliceRows: weight: IWeight * firstRow: int * rowCount: int -> IWeight
    abstract CreateKvCache: capacity: int * kvDim: int -> IKvCache

    /// out = x / rms(x) * weight
    abstract RmsNorm: output: IBuffer * x: IBuffer * weight: IWeight * eps: float32 -> unit
    /// In-place RMSNorm applied independently to each head of x (Qwen3 QK-norm).
    abstract HeadRmsNorm: x: IBuffer * weight: IWeight * numHeads: int * headDim: int * eps: float32 -> unit
    /// output = W · x (W has Rows outputs of Columns inputs).
    abstract MatVec: output: IBuffer * x: IBuffer * weight: IWeight -> unit
    /// outputs[i] = weights[i] · inputs[i] for every i, as one job. Inputs may repeat (Q/K/V and
    /// gate/up share theirs): a backend may quantize each distinct input once and schedule all rows
    /// of all products together (MoE experts are many small products with distinct inputs).
    abstract MatVecBatch: inputs: IBuffer[] * outputs: IBuffer[] * weights: IWeight[] -> unit
    /// target += other
    abstract Add: target: IBuffer * other: IBuffer -> unit
    /// target += bias
    abstract AddBias: target: IBuffer * bias: IWeight -> unit
    /// target += scale · other
    abstract AddScaled: target: IBuffer * other: IBuffer * scale: float32 -> unit
    /// Rotary embedding of q (NumHeads) and k (NumKvHeads) at position pos.
    abstract Rope: q: IBuffer * k: IBuffer * position: int * spec: RopeSpec -> unit
    /// Writes k and v into slot <paramref name="position"/> of the cache.
    abstract StoreKv: cache: IKvCache * k: IBuffer * v: IBuffer * position: int -> unit
    /// Causal multi-head attention of q over cache positions 0..position (inclusive).
    abstract Attention: output: IBuffer * q: IBuffer * cache: IKvCache * position: int * spec: AttentionSpec * scratch: IBuffer -> unit
    /// gate = silu(gate) * up
    abstract SwiGlu: gate: IBuffer * up: IBuffer -> unit
    abstract Synchronize: unit -> unit

/// Attention weights of one block on a device.
type DeviceAttention =
    { Norm: IWeight
      Wq: IWeight
      Wk: IWeight
      Wv: IWeight
      Wo: IWeight
      Bq: IWeight option
      Bk: IWeight option
      Bv: IWeight option
      QNorm: IWeight option
      KNorm: IWeight option }

/// Mixture of experts on a device. Per-expert views are created once at load time, so routing a
/// token allocates nothing.
type DeviceMoe =
    { Router: IWeight
      /// Owning stacked tensors (disposed with the block).
      Stacks: IWeight[]
      Gate: IWeight[]
      Up: IWeight[]
      Down: IWeight[] }

type DeviceFfnBlock =
    | DenseBlock of gate: IWeight * up: IWeight * down: IWeight
    | MoeBlock of DeviceMoe

/// Feed-forward weights of one block on a device.
type DeviceFfn = { Norm: IWeight; Block: DeviceFfnBlock }

/// The two halves of a decoder block. They can live on different devices (op-level placement).
type LayerPart =
    | AttentionPart
    | FfnPart

module DeviceWeights =

    let attention (h: HostAttention) (load: HostTensor -> IWeight) : DeviceAttention =
        { Norm = load h.Norm
          Wq = load h.Wq
          Wk = load h.Wk
          Wv = load h.Wv
          Wo = load h.Wo
          Bq = h.Bq |> Option.map load
          Bk = h.Bk |> Option.map load
          Bv = h.Bv |> Option.map load
          QNorm = h.QNorm |> Option.map load
          KNorm = h.KNorm |> Option.map load }

    /// <paramref name="slice"/> creates a row view (see <see cref="IComputeBackend.SliceRows"/>).
    let ffn (h: HostFfn) (load: HostTensor -> IWeight) (slice: IWeight * int * int -> IWeight) : DeviceFfn =
        let block =
            match h.Block with
            | Dense d -> DenseBlock(load d.Gate, load d.Up, load d.Down)
            | Moe m ->
                let gate = load m.GateExperts
                let up = load m.UpExperts
                let down = load m.DownExperts
                let experts = m.ExpertCount
                let split (w: IWeight) =
                    let rows = w.Rows / experts
                    Array.init experts (fun e -> slice (w, e * rows, rows))
                MoeBlock
                    { Router = load m.Router
                      Stacks = [| gate; up; down |]
                      Gate = split gate
                      Up = split up
                      Down = split down }
        { Norm = load h.Norm; Block = block }

    let disposeAttention (a: DeviceAttention) =
        for w in [ a.Norm; a.Wq; a.Wk; a.Wv; a.Wo ] do w.Dispose()
        for w in [ a.Bq; a.Bk; a.Bv; a.QNorm; a.KNorm ] do w |> Option.iter (fun w -> w.Dispose())

    let disposeFfn (f: DeviceFfn) =
        f.Norm.Dispose()
        match f.Block with
        | DenseBlock(g, u, d) -> for w in [ g; u; d ] do w.Dispose()
        | MoeBlock m ->
            m.Router.Dispose()
            for w in m.Stacks do w.Dispose()

/// Supplies the weights of layer parts right before they execute. Resident sources return
/// preloaded tensors; streaming sources page weights in from host memory (offloading).
/// For a given layer the runtime acquires the attention part before the FFN part.
type ILayerSource =
    inherit IDisposable
    abstract Parts: (int * LayerPart)[]
    abstract AcquireAttention: layer: int -> DeviceAttention
    abstract AcquireFfn: layer: int -> DeviceFfn

/// Optional backend capability: execute whole layers whose weights live in host memory.
type IWeightStreaming =
    /// Bytes of device memory the streaming machinery needs for the given layers.
    abstract StagingBytes: layers: HostLayer[] -> int64
    abstract CreateStreamedSource: layers: (int * HostLayer)[] -> ILayerSource

/// Keeps the assigned layer parts permanently on one backend. If an upload fails midway (e.g.
/// out of device memory), every weight already uploaded is released before the error propagates.
[<Sealed>]
type ResidentLayerSource(backend: IComputeBackend, parts: (int * HostLayer * LayerPart)[]) =
    let uploaded = Collections.Generic.List<IWeight>()
    let load (t: HostTensor) =
        let w = backend.LoadWeight t
        uploaded.Add w
        w
    let attention, ffn =
        try
            let attention =
                parts
                |> Array.choose (fun (i, h, p) -> if p = AttentionPart then Some(i, DeviceWeights.attention h.Attention load) else None)
                |> dict
            let ffn =
                parts
                |> Array.choose (fun (i, h, p) -> if p = FfnPart then Some(i, DeviceWeights.ffn h.Ffn load backend.SliceRows) else None)
                |> dict
            attention, ffn
        with _ ->
            for w in uploaded do w.Dispose()
            reraise ()
    let partList = parts |> Array.map (fun (i, _, p) -> i, p)
    let mutable disposed = false

    interface ILayerSource with
        member _.Parts = partList
        member _.AcquireAttention(layer) = attention.[layer]
        member _.AcquireFfn(layer) = ffn.[layer]
        member _.Dispose() =
            if not disposed then
                disposed <- true
                attention.Values |> Seq.iter DeviceWeights.disposeAttention
                ffn.Values |> Seq.iter DeviceWeights.disposeFfn
