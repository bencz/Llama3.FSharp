namespace Llama3.Backend.Cpu

open System
open System.Buffers.Binary
open System.Runtime.CompilerServices

/// Q4_K 6-bit scale/min unpacking, all eight at once with 32-bit lane arithmetic (the ggml
/// "utmp" trick): byte j of the returned scales/mins equals get_scale_min_k4(j).
module Q4KScales =
    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    let unpack (packed: ReadOnlySpan<byte>) : struct (uint64 * uint64) =
        let u0 = BinaryPrimitives.ReadUInt32LittleEndian packed
        let u1 = BinaryPrimitives.ReadUInt32LittleEndian(packed.Slice 4)
        let u2 = BinaryPrimitives.ReadUInt32LittleEndian(packed.Slice 8)
        let scLow = u0 &&& 0x3f3f3f3fu
        let mnLow = u1 &&& 0x3f3f3f3fu
        let scHigh = (u2 &&& 0x0f0f0f0fu) ||| (((u0 >>> 6) &&& 0x03030303u) <<< 4)
        let mnHigh = ((u2 >>> 4) &&& 0x0f0f0f0fu) ||| (((u1 >>> 6) &&& 0x03030303u) <<< 4)
        struct (uint64 scLow ||| (uint64 scHigh <<< 32), uint64 mnLow ||| (uint64 mnHigh <<< 32))

    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    let byteAt (packed: uint64) (j: int) = int ((packed >>> (8 * j)) &&& 0xFFUL)
