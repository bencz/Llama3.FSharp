namespace Llama3.Core

open System
open System.Buffers.Binary
open System.Runtime.CompilerServices

/// Scalar conversions between the storage formats found in checkpoints and float32.
/// Every routine is branch-light and bit-exact with respect to IEEE 754.
module Numerics =

    /// Converts IEEE 754 binary16 bits to float32 without going through System.Half
    /// (keeps the conversion identical to the GPU kernels, which use the same bit manipulation).
    let halfBitsToSingle (h: uint16) : float32 =
        let h = uint32 h
        let sign = (h &&& 0x8000u) <<< 16
        let exponent = (h >>> 10) &&& 0x1Fu
        let mantissa = h &&& 0x3FFu
        if exponent = 0u then
            if mantissa = 0u then
                BitConverter.UInt32BitsToSingle sign
            else
                // Subnormal: value = mantissa * 2^-24, exactly representable in float32.
                let magnitude = float32 mantissa * 5.96046448e-8f
                if sign <> 0u then -magnitude else magnitude
        elif exponent = 31u then
            BitConverter.UInt32BitsToSingle(sign ||| 0x7F800000u ||| (mantissa <<< 13))
        else
            BitConverter.UInt32BitsToSingle(sign ||| ((exponent + 112u) <<< 23) ||| (mantissa <<< 13))

    /// Lookup table covering all 65536 binary16 encodings (256 KiB), the hot path for F16/K-quant scales.
    let private halfTable: float32[] =
        Array.init 65536 (fun i -> halfBitsToSingle (uint16 i))

    /// Table-driven binary16 -> float32.
    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    let halfToSingle (bits: uint16) : float32 = halfTable.[int bits]

    /// Reads a little-endian binary16 at <paramref name="offset"/> and widens it.
    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    let readHalf (data: ReadOnlySpan<byte>) (offset: int) : float32 =
        halfTable.[int (BinaryPrimitives.ReadUInt16LittleEndian(data.Slice(offset, 2)))]

    /// bfloat16 is the upper half of a float32.
    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    let bfloat16ToSingle (bits: uint16) : float32 =
        BitConverter.UInt32BitsToSingle(uint32 bits <<< 16)

    /// Round-to-nearest-even float32 -> binary16 (used by tests and by the quantizers).
    let singleToHalfBits (value: float32) : uint16 = BitConverter.HalfToUInt16Bits(Half.op_Explicit value)

    /// Integer ceiling division for non-negative operands.
    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    let divUp (a: int) (b: int) = (a + b - 1) / b

    /// Rounds <paramref name="value"/> up to a multiple of <paramref name="alignment"/> (power of two).
    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    let alignUp (value: int64) (alignment: int64) = (value + alignment - 1L) &&& ~~~(alignment - 1L)
