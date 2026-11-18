/// Helpers shared by every test project (linked, not referenced).
module Llama3.Tests.TestSupport

open System
open System.IO
open Llama3.Core

/// Repository root (directory containing Llama3.FSharp.slnx).
let repoRoot =
    let rec up (dir: DirectoryInfo) =
        if isNull dir then None
        elif File.Exists(Path.Combine(dir.FullName, "Llama3.FSharp.slnx")) then Some dir.FullName
        else up dir.Parent
    [ AppContext.BaseDirectory; Environment.CurrentDirectory ]
    |> List.tryPick (DirectoryInfo >> up)
    |> Option.defaultWith (fun () -> failwith "Repository root (Llama3.FSharp.slnx) not found")

/// Skips a test whose prerequisite (GPU, model file) is missing. With LLAMA3_REQUIRE_ALL=1 (CI on
/// a fully equipped machine) a missing prerequisite fails the test instead of hiding it.
let skipUnavailable (reason: string) : unit =
    if Environment.GetEnvironmentVariable "LLAMA3_REQUIRE_ALL" = "1" then
        Xunit.Assert.Fail $"%s{reason} (LLAMA3_REQUIRE_ALL=1 forbids skipping)"
    else Xunit.Assert.Skip reason

let storiesModel = Path.Combine(repoRoot, "models", "stories15M.bin")
let storiesTokenizer = Path.Combine(repoRoot, "models", "tokenizer.bin")

/// Output of the original llama3.cu for "I have a dream" (README of likejazz/llama3.cuda).
let storiesGolden =
    "I have a dream. He dreams of a big, beautiful garden full of flowers and trees. He dreams of playing with his friends and eating yummy snacks.\nOne day, he was walking in the garden when he saw"

/// Directory with GGUF models: $LLAMA3_MODELS_DIR or ~/models.
let modelsDir =
    match Environment.GetEnvironmentVariable "LLAMA3_MODELS_DIR" with
    | null | "" -> Path.Combine(Environment.GetFolderPath Environment.SpecialFolder.UserProfile, "models")
    | d -> d

/// First existing file among candidate names (relative to modelsDir).
let tryModel (candidates: string list) =
    candidates |> List.map (fun c -> Path.Combine(modelsDir, c)) |> List.tryFind File.Exists

let qwen3F32 () = tryModel [ "matrix/Qwen3-0.6B-F32.gguf"; "Qwen3-0.6B-F32.gguf" ]
let qwen3Q8 () = tryModel [ "Qwen3-0.6B-Q8_0.gguf"; "matrix/Qwen3-0.6B-Q8_0.gguf" ]

let private half (v: float32) = Numerics.singleToHalfBits v

let private writeHalf (bytes: byte[]) (offset: int) (v: float32) =
    let h = half v
    bytes.[offset] <- byte h
    bytes.[offset + 1] <- byte (h >>> 8)

/// Random tensor of the given format whose decoded values are well-scaled (|w| <= ~1).
let randomTensor (rng: Random) (dtype: GgmlType) (rows: int) (cols: int) : HostTensor =
    let bytes = Array.zeroCreate<byte> (int (GgmlTypes.rowBytes dtype (int64 rows * int64 cols)))
    let uniform (lo: float) (hi: float) = float32 (lo + (hi - lo) * rng.NextDouble())
    match dtype with
    | GgmlType.F32 ->
        let values = Array.init (rows * cols) (fun _ -> uniform -1.0 1.0)
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length)
    | GgmlType.F16 ->
        for i in 0 .. rows * cols - 1 do writeHalf bytes (2 * i) (uniform -1.0 1.0)
    | GgmlType.Q8_0 ->
        for b in 0 .. bytes.Length / 34 - 1 do
            let o = b * 34
            writeHalf bytes o (uniform 0.002 0.01)
            for i in 0 .. 31 do bytes.[o + 2 + i] <- byte (sbyte (rng.Next(-127, 128)))
    | GgmlType.Q4_K ->
        rng.NextBytes bytes
        for b in 0 .. bytes.Length / 144 - 1 do
            let o = b * 144
            writeHalf bytes o (uniform 0.0005 0.003)
            writeHalf bytes (o + 2) (uniform 0.0 0.003)
    | GgmlType.Q6_K ->
        rng.NextBytes bytes
        for b in 0 .. bytes.Length / 210 - 1 do
            writeHalf bytes (b * 210 + 208) (uniform 0.0002 0.001)
    | GgmlType.BF16 ->
        // Round-to-nearest-even of a uniform float32 (what converters do).
        for i in 0 .. rows * cols - 1 do
            let bits = BitConverter.SingleToUInt32Bits(uniform -1.0 1.0)
            let rounded = (bits + 0x7FFFu + ((bits >>> 16) &&& 1u)) >>> 16
            bytes.[2 * i] <- byte rounded
            bytes.[2 * i + 1] <- byte (rounded >>> 8)
    | other -> failwith $"randomTensor: %A{other} not supported"
    HostTensor($"random.%A{dtype}", dtype, [| int64 cols; int64 rows |], bytes)

let randomVector (rng: Random) (n: int) = Array.init n (fun _ -> float32 (rng.NextDouble() * 2.0 - 1.0))

/// W*x in float64 over exactly decoded weights.
let referenceMatVec (w: HostTensor) (x: float32[]) : float32[] =
    let decoded = Dequantize.toSingles w
    Array.init w.Rows (fun r ->
        let mutable acc = 0.0
        for c in 0 .. w.Columns - 1 do
            acc <- acc + float decoded.[r * w.Columns + c] * float x.[c]
        float32 acc)

/// max_i |a_i - b_i| / max(1, max_i |b_i|)
let relativeError (actual: float32[]) (expected: float32[]) =
    let scale = expected |> Array.fold (fun m v -> max m (abs v)) 1.0f
    Array.fold2 (fun m a e -> max m (abs (a - e))) 0.0f actual expected / scale

/// Index of the largest element.
let argmax (v: float32[]) = Sampler.Argmax(ReadOnlySpan v)
