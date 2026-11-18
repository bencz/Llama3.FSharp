namespace Llama3.Backend.Cuda

open ILGPU
open ILGPU.Algorithms

/// Normalisation and attention (explicitly grouped kernels with group-wide reductions).
/// Every kernel is a curried module function: it compiles to a plain static method, which F# 11
/// binds directly to a delegate (no closure): the form ILGPU compiles to PTX. The project disables
/// F# tail calls, whose IL prefix ILGPU rejects.
[<RequireQualifiedAccess>]
module AttentionKernels =

    /// Group size of the reduction kernels (RMSNorm, attention).
    [<Literal>]
    let ReductionGroupSize = 256

    /// Single group: output = weight * (x * rsqrt(mean(x^2) + eps)).
    let rmsNorm (output: ArrayView<float32>) (x: ArrayView<float32>) (weight: ArrayView<float32>) (size: int) (eps: float32) =
        let shared = SharedMemory.Allocate<float32>(32)
        let tid = Group.IdxX
        let mutable ss = 0.0f
        let mutable i = tid
        while i < size do
            let v = x.[i]
            ss <- ss + v * v
            i <- i + Group.DimX
        let total = DeviceMath.groupSum ss shared
        let scale = 1.0f / XMath.Sqrt(total / float32 size + eps)
        i <- tid
        while i < size do
            output.[i] <- weight.[i] * (scale * x.[i])
            i <- i + Group.DimX

    /// One warp per head (grid = heads, group = 32), in place.
    let headRmsNorm (x: ArrayView<float32>) (weight: ArrayView<float32>) (headDim: int) (eps: float32) =
        let offset = Grid.IdxX * headDim
        let lane = Warp.LaneIdx
        let mutable ss = 0.0f
        let mutable i = lane
        while i < headDim do
            let v = x.[offset + i]
            ss <- ss + v * v
            i <- i + 32
        let total = DeviceMath.warpAllSum ss
        let scale = 1.0f / XMath.Sqrt(total / float32 headDim + eps)
        i <- lane
        while i < headDim do
            x.[offset + i] <- weight.[i] * (scale * x.[offset + i])
            i <- i + 32

    /// One group per query head (grid = heads): scores into <paramref name="att"/> (row h of a
    /// heads x capacity scratch), softmax over 0..position, then the value mix. GQA: head h reads
    /// KV head h / KvMul.
    let attention (output: ArrayView<float32>) (q: ArrayView<float32>) (keys: ArrayView<float32>) (values: ArrayView<float32>) (att: ArrayView<float32>) (args: AttentionArgs) =
        let shared = SharedMemory.Allocate<float32>(32)
        let h = Grid.IdxX
        let tid = Group.IdxX
        let lane = Warp.LaneIdx
        let warps = Group.DimX / 32
        let headDim = args.HeadDim
        let qOff = h * headDim
        let kvOff = (h / args.KvMul) * headDim
        let attOff = int64 h * int64 args.Capacity
        let length = args.Position + 1
        // 1. scores (one warp per timestep, lanes split head_dim)
        let mutable t = Warp.WarpIdx
        while t < length do
            let kBase = int64 t * int64 args.KvDim + int64 kvOff
            let mutable s = 0.0f
            let mutable i = lane
            while i < headDim do
                s <- s + q.[qOff + i] * keys.[kBase + int64 i]
                i <- i + 32
            let score = DeviceMath.warpAllSum s
            if lane = 0 then att.[attOff + int64 t] <- score / args.SqrtHeadDim
            t <- t + warps
        Group.Barrier()
        // 2. softmax over 0..position
        let mutable localMax = -3.402823466e38f
        t <- tid
        while t < length do
            let v = att.[attOff + int64 t]
            if v > localMax then localMax <- v
            t <- t + Group.DimX
        let maxVal = DeviceMath.groupMax localMax shared
        let mutable localSum = 0.0f
        t <- tid
        while t < length do
            let e = XMath.Exp(att.[attOff + int64 t] - maxVal)
            att.[attOff + int64 t] <- e
            localSum <- localSum + e
            t <- t + Group.DimX
        let sum = DeviceMath.groupSum localSum shared
        t <- tid
        while t < length do
            att.[attOff + int64 t] <- att.[attOff + int64 t] / sum
            t <- t + Group.DimX
        Group.Barrier()
        // 3. weighted sum of values
        let mutable i = tid
        while i < headDim do
            let mutable acc = 0.0f
            for tt in 0 .. length - 1 do
                acc <- acc + att.[attOff + int64 tt] * values.[int64 tt * int64 args.KvDim + int64 (kvOff + i)]
            output.[qOff + i] <- acc
            i <- i + Group.DimX
