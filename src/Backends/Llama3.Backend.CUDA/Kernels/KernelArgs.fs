namespace Llama3.Backend.Cuda

open System.Runtime.InteropServices

/// Scalar arguments of the RoPE kernel (packed: ILGPU passes structs by value).
[<Struct; StructLayout(LayoutKind.Sequential)>]
type RopeArgs =
    val Position: int
    val NumHeads: int
    val NumKvHeads: int
    val HeadDim: int
    val Theta: float32
    /// 1 = NeoX (split halves), 0 = interleaved pairs.
    val NeoX: int
    val HasFactors: int

    new(position, numHeads, numKvHeads, headDim, theta, neox, hasFactors) =
        { Position = position
          NumHeads = numHeads
          NumKvHeads = numKvHeads
          HeadDim = headDim
          Theta = theta
          NeoX = neox
          HasFactors = hasFactors }

/// Scalar arguments of the attention kernel.
[<Struct; StructLayout(LayoutKind.Sequential)>]
type AttentionArgs =
    val Position: int
    val Capacity: int
    val KvDim: int
    val HeadDim: int
    val KvMul: int
    val SqrtHeadDim: float32

    new(position, capacity, kvDim, headDim, kvMul, sqrtHeadDim) =
        { Position = position
          Capacity = capacity
          KvDim = kvDim
          HeadDim = headDim
          KvMul = kvMul
          SqrtHeadDim = sqrtHeadDim }
