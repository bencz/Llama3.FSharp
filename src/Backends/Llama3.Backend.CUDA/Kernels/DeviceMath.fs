namespace Llama3.Backend.Cuda

open ILGPU

/// Device-side helpers shared by every kernel: binary16 decoding and warp/group reductions.
///
/// Never convert an unsigned integer straight to float in device code: F# emits <c>conv.r.un</c>,
/// which goes through float64 — 1/32 throughput on consumer GPUs. Convert via <c>int</c> first.
/// Helpers are curried module functions (plain static methods) that ILGPU inlines into kernels.
[<RequireQualifiedAccess>]
module DeviceMath =

    /// IEEE binary16 bits (low 16 bits of <paramref name="h"/>) → float32 through the hardware
    /// converter (cvt.f32.f16: one instruction, exact for every encoding incl. subnormals/inf/NaN).
    let halfToFloat (h: uint32) : float32 =
        let half = Interop.IntAsFloat(uint16 h)
        let value: float32 = Half.op_Implicit half
        value

    /// Butterfly reduction: every lane ends with the warp-wide sum.
    let warpAllSum (value: float32) : float32 =
        let mutable s = value
        let mutable mask = 16
        while mask > 0 do
            s <- s + Warp.ShuffleXor(s, mask)
            mask <- mask >>> 1
        s

    let warpAllMax (value: float32) : float32 =
        let mutable s = value
        let mutable mask = 16
        while mask > 0 do
            let other = Warp.ShuffleXor(s, mask)
            if other > s then s <- other
            mask <- mask >>> 1
        s

    /// Group-wide sum; every thread receives the result. <paramref name="shared"/> needs 32 slots.
    let groupSum (value: float32) (shared: ArrayView<float32>) : float32 =
        let partial = warpAllSum value
        if Warp.LaneIdx = 0 then shared.[Warp.WarpIdx] <- partial
        Group.Barrier()
        let warps = (Group.DimX + 31) / 32
        let v = if Warp.LaneIdx < warps then shared.[Warp.LaneIdx] else 0.0f
        let total = warpAllSum v
        Group.Barrier()
        total

    let groupMax (value: float32) (shared: ArrayView<float32>) : float32 =
        let partial = warpAllMax value
        if Warp.LaneIdx = 0 then shared.[Warp.WarpIdx] <- partial
        Group.Barrier()
        let warps = (Group.DimX + 31) / 32
        let v = if Warp.LaneIdx < warps then shared.[Warp.LaneIdx] else -3.402823466e38f
        let total = warpAllMax v
        Group.Barrier()
        total

    /// Sign-extends byte <paramref name="k"/> of a packed word.
    let sByteOf (v: uint32) (k: int) : float32 = float32 ((int (v <<< (24 - 8 * k))) >>> 24)
