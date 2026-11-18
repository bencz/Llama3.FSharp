namespace Llama3.Backend.Cuda

open ILGPU
open ILGPU.Algorithms

/// Element-wise kernels (auto-grouped, one thread per element or rotation pair).
/// Every kernel is a curried module function: it compiles to a plain static method, which F# 11
/// binds directly to a delegate (no closure): the form ILGPU compiles to PTX. The project disables
/// F# tail calls, whose IL prefix ILGPU rejects.
[<RequireQualifiedAccess>]
module ElementwiseKernels =

    let rope (index: Index1D) (q: ArrayView<float32>) (k: ArrayView<float32>) (factors: ArrayView<float32>) (args: RopeArgs) =
        let half = args.HeadDim / 2
        let idx = index.X
        let pair = idx % half
        let head = idx / half
        let mutable freq = 1.0f / XMath.Pow(args.Theta, float32 (2 * pair) / float32 args.HeadDim)
        if args.HasFactors <> 0 then freq <- freq / factors.[pair]
        let angle = float32 args.Position * freq
        let c = XMath.Cos angle
        let s = XMath.Sin angle
        let i0 = if args.NeoX <> 0 then pair else 2 * pair
        let i1 = if args.NeoX <> 0 then pair + half else 2 * pair + 1
        if head < args.NumHeads then
            let o = head * args.HeadDim
            let a = q.[o + i0]
            let b = q.[o + i1]
            q.[o + i0] <- a * c - b * s
            q.[o + i1] <- a * s + b * c
        elif head < args.NumHeads + args.NumKvHeads then
            let o = (head - args.NumHeads) * args.HeadDim
            let a = k.[o + i0]
            let b = k.[o + i1]
            k.[o + i0] <- a * c - b * s
            k.[o + i1] <- a * s + b * c

    let swiGlu (index: Index1D) (gate: ArrayView<float32>) (up: ArrayView<float32>) =
        let i = index.X
        let v = gate.[i]
        gate.[i] <- v * (1.0f / (1.0f + XMath.Exp(-v))) * up.[i]

    let add (index: Index1D) (target: ArrayView<float32>) (other: ArrayView<float32>) =
        let i = index.X
        target.[i] <- target.[i] + other.[i]

    let addScaled (index: Index1D) (target: ArrayView<float32>) (other: ArrayView<float32>) (scale: float32) =
        let i = index.X
        target.[i] <- target.[i] + scale * other.[i]
