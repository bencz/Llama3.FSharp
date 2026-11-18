namespace Llama3.Backend.Cuda

open System
open System.Collections.Generic
open ILGPU.Runtime
open Llama3.Core

/// Byte layout of a layer packed into one contiguous region (256-byte aligned tensors), shared by
/// the pinned host copy and the device slots so one DMA moves a whole layer.
type internal PackedLayout =
    { Offsets: Dictionary<HostTensor, int64>
      Size: int64 }

module internal Packing =
    let alignment = 256L

    let layout (layer: HostLayer) : PackedLayout =
        let offsets = Dictionary<HostTensor, int64>(HashIdentity.Reference)
        let mutable cursor = 0L
        for t in layer.Tensors do
            if not (offsets.ContainsKey t) then
                offsets.[t] <- cursor
                cursor <- Numerics.alignUp (cursor + GpuLayout.deviceBytes t) alignment
        { Offsets = offsets; Size = max cursor alignment }

/// Weight offloading: layers live in page-locked host memory and are copied into one of two device
/// slots right before they run. While layer i computes on the compute stream, layer i+1 is
/// transferred on a dedicated copy stream (double buffering), so PCIe time overlaps kernels.
[<Sealed>]
type CudaStreamedLayerSource(accelerator: Accelerator, computeStream: AcceleratorStream, layers: (int * HostLayer)[]) =
    let order = layers |> Array.map fst
    let indexOf = order |> Array.mapi (fun i l -> KeyValuePair(l, i)) |> Dictionary
    let layouts = layers |> Array.map (fun (_, h) -> Packing.layout h)
    let slotBytes = layouts |> Array.map (fun l -> l.Size) |> Array.max
    let slotCount = if order.Length = 1 then 1 else 2
    let bind () =
        if not (obj.ReferenceEquals(Accelerator.Current, accelerator)) then accelerator.Bind()

    // Everything is acquired in order; on failure (pinned or device memory exhausted) the parts
    // already obtained are released before the exception propagates.
    let owned = Stack<IDisposable>()
    let own (d: 'T :> IDisposable) =
        owned.Push d
        d
    let releaseAll () =
        while owned.Count > 0 do owned.Pop().Dispose()

    let copyStream, pinned, slots =
        try
            bind ()
            let copyStream = own (accelerator.CreateStream())
            let pinned =
                layers
                |> Array.mapi (fun i _ ->
                    let layout = layouts.[i]
                    let pl = own (accelerator.AllocatePageLocked1D<byte>(layout.Size))
                    let span = pl.Span
                    for KeyValue(t, off) in layout.Offsets do
                        GpuLayout.write t (span.Slice(int off, int (GpuLayout.deviceBytes t)))
                    pl)
            let slots = Array.init slotCount (fun _ -> own (accelerator.Allocate1D<byte>(slotBytes)))
            copyStream, pinned, slots
        with _ ->
            releaseAll ()
            reraise ()
    /// Layer index (position in `order`) currently held or being loaded by each slot.
    let slotContent = Array.create slotCount -1
    let mutable disposed = false
    let views = Dictionary<struct (int * int), struct (DeviceAttention * DeviceFfn)>()

    let slotOf (i: int) = if order.Length = 1 then 0 else i % 2

    let beginCopy (i: int) =
        let s = slotOf i
        let size = layouts.[i].Size
        slots.[s].View.SubView(0L, size).CopyFromPageLockedAsync(copyStream, pinned.[i])
        slotContent.[s] <- i

    /// Views of layer i's tensors inside slot s (built once per pair, then reused).
    let deviceLayer (i: int) (s: int) =
        match views.TryGetValue(struct (i, s)) with
        | true, l -> l
        | _ ->
            let (_, host) = layers.[i]
            let layout = layouts.[i]
            let baseView = slots.[s].View.AsContiguous()
            let load (t: HostTensor) =
                let view = baseView.SubView(layout.Offsets.[t], GpuLayout.deviceBytes t)
                new CudaWeight(t.Name, t.Type, t.Rows, t.Columns, view, None) :> IWeight
            let slice (w: IWeight, first: int, count: int) = (w :?> CudaWeight).Slice(first, count) :> IWeight
            let l = struct (DeviceWeights.attention host.Attention load, DeviceWeights.ffn host.Ffn load slice)
            views.[struct (i, s)] <- l
            l

    /// Slot currently holding layer i (it must have been acquired through its attention part).
    let resident (i: int) =
        let s = slotOf i
        if slotContent.[s] <> i then invalidOp $"Streamed layer %d{order.[i]}: FFN acquired before attention"
        s


    /// Device bytes needed for the two slots.
    static member RequiredDeviceBytes(layers: HostLayer[]) =
        if layers.Length = 0 then 0L
        else
            let slot = layers |> Array.map (fun h -> (Packing.layout h).Size) |> Array.max
            (if layers.Length = 1 then 1L else 2L) * slot

    interface ILayerSource with
        member _.Parts = [| for l in order do yield l, AttentionPart; yield l, FfnPart |]

        member _.AcquireAttention(layer) =
            bind ()
            let i = indexOf.[layer]
            let s = slotOf i
            if slotContent.[s] <> i then
                // Not prefetched (first token, or odd layer count wrap-around): the slot may still
                // be read by queued kernels, so drain them before overwriting.
                computeStream.Synchronize()
                beginCopy i
            copyStream.Synchronize()
            // Prefetch the next streamed layer into the other slot once its previous user is done.
            if order.Length > 2 then
                let next = (i + 1) % order.Length
                let ns = slotOf next
                if ns <> s && slotContent.[ns] <> next then
                    computeStream.Synchronize()
                    beginCopy next
            let struct (attention, _) = deviceLayer i s
            attention

        member _.AcquireFfn(layer) =
            let i = indexOf.[layer]
            let struct (_, ffn) = deviceLayer i (resident i)
            ffn

        member _.Dispose() =
            if not disposed then
                disposed <- true
                bind ()
                try
                    // Kernels of the compute stream may still read the slots.
                    copyStream.Synchronize()
                    computeStream.Synchronize()
                finally
                    releaseAll ()
