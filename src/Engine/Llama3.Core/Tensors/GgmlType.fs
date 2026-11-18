namespace Llama3.Core

open System
open System.Runtime.InteropServices

/// Storage formats, numbered exactly like <c>enum ggml_type</c> so GGUF values map 1:1.
type GgmlType =
    | F32 = 0
    | F16 = 1
    | Q4_0 = 2
    | Q4_1 = 3
    | Q5_0 = 6
    | Q5_1 = 7
    | Q8_0 = 8
    | Q8_1 = 9
    | Q2_K = 10
    | Q3_K = 11
    | Q4_K = 12
    | Q5_K = 13
    | Q6_K = 14
    | Q8_K = 15
    | I8 = 24
    | I16 = 25
    | I32 = 26
    | I64 = 27
    | F64 = 28
    | BF16 = 30

/// Block geometry of every storage format (elements per block, bytes per block).
/// Byte layouts follow ggml-common.h:
///   Q8_0 : { f16 d; i8 qs[32] }                                   = 34 B / 32
///   Q4_0 : { f16 d; u8 qs[16] }                                   = 18 B / 32
///   Q4_K : { f16 d; f16 dmin; u8 scales[12]; u8 qs[128] }         = 144 B / 256
///   Q6_K : { u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d }        = 210 B / 256
module GgmlTypes =

    /// Super-block size of the K-quants.
    [<Literal>]
    let QK_K = 256

    /// Block size of the legacy 32-element quants.
    [<Literal>]
    let QK8_0 = 32

    let blockSize (t: GgmlType) =
        match t with
        | GgmlType.F32 | GgmlType.F16 | GgmlType.BF16 | GgmlType.F64
        | GgmlType.I8 | GgmlType.I16 | GgmlType.I32 | GgmlType.I64 -> 1
        | GgmlType.Q4_0 | GgmlType.Q4_1 | GgmlType.Q5_0 | GgmlType.Q5_1
        | GgmlType.Q8_0 | GgmlType.Q8_1 -> 32
        | GgmlType.Q2_K | GgmlType.Q3_K | GgmlType.Q4_K | GgmlType.Q5_K
        | GgmlType.Q6_K | GgmlType.Q8_K -> QK_K
        | other -> raise (NotSupportedException $"Unknown ggml type %A{other}")

    let typeSize (t: GgmlType) =
        match t with
        | GgmlType.F32 | GgmlType.I32 -> 4
        | GgmlType.F16 | GgmlType.BF16 | GgmlType.I16 -> 2
        | GgmlType.I8 -> 1
        | GgmlType.F64 | GgmlType.I64 -> 8
        | GgmlType.Q4_0 -> 18
        | GgmlType.Q4_1 -> 20
        | GgmlType.Q5_0 -> 22
        | GgmlType.Q5_1 -> 24
        | GgmlType.Q8_0 -> 34
        | GgmlType.Q8_1 -> 36
        | GgmlType.Q2_K -> 84
        | GgmlType.Q3_K -> 110
        | GgmlType.Q4_K -> 144
        | GgmlType.Q5_K -> 176
        | GgmlType.Q6_K -> 210
        | GgmlType.Q8_K -> 292
        | other -> raise (NotSupportedException $"Unknown ggml type %A{other}")

    /// Bytes needed to store <paramref name="elements"/> values (must be a whole number of blocks).
    let rowBytes (t: GgmlType) (elements: int64) =
        let bs = int64 (blockSize t)
        if elements % bs <> 0L then
            invalidArg (nameof elements) $"%d{elements} elements is not a multiple of the %A{t} block size %d{bs}"
        elements / bs * int64 (typeSize t)

    /// Formats that the compute backends consume directly (no conversion at load time).
    let isNativeComputeType (t: GgmlType) =
        match t with
        | GgmlType.F32 | GgmlType.F16 | GgmlType.BF16 | GgmlType.Q8_0 | GgmlType.Q4_K | GgmlType.Q6_K -> true
        | _ -> false

    /// Types this runtime knows the geometry of (anything else in a file is rejected at parse time).
    let isKnown (t: GgmlType) =
        match t with
        | GgmlType.F32 | GgmlType.F16 | GgmlType.BF16 | GgmlType.F64 | GgmlType.I8 | GgmlType.I16 | GgmlType.I32 | GgmlType.I64
        | GgmlType.Q4_0 | GgmlType.Q4_1 | GgmlType.Q5_0 | GgmlType.Q5_1 | GgmlType.Q8_0 | GgmlType.Q8_1
        | GgmlType.Q2_K | GgmlType.Q3_K | GgmlType.Q4_K | GgmlType.Q5_K | GgmlType.Q6_K | GgmlType.Q8_K -> true
        | _ -> false

    /// Formats that can be decoded to float32 (and therefore converted at load time).
    let isDecodable (t: GgmlType) =
        match t with
        | GgmlType.F32 | GgmlType.F16 | GgmlType.BF16 | GgmlType.Q4_0 | GgmlType.Q8_0
        | GgmlType.Q4_K | GgmlType.Q6_K -> true
        | _ -> false
