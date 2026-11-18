namespace Llama3.Core

open System
open System.Collections.Generic
open System.IO
open System.Text
open System.Text.RegularExpressions

/// Token categories (GGUF <c>tokenizer.ggml.token_type</c> values).
type TokenKind =
    | Normal = 1
    | Unknown = 2
    | Control = 3
    | UserDefined = 4
    | Unused = 5
    | Byte = 6

/// Text <-> token ids.
type ITokenizer =
    abstract VocabSize: int
    abstract BosId: int option
    abstract EosId: int option
    /// Encodes plain text. Special-token markup inside the text is NOT interpreted.
    abstract Encode: text: string -> int[]
    /// Raw bytes produced by <paramref name="token"/> when it follows <paramref name="previous"/>.
    /// Pieces may split UTF-8 sequences; use <see cref="Utf8StreamDecoder"/> to reassemble text.
    abstract Decode: previous: int * token: int -> byte[]
    /// Id of a special token given its literal text (e.g. "&lt;|im_start|&gt;").
    abstract TryGetSpecial: text: string -> int option
    abstract Kind: token: int -> TokenKind
    /// Human-readable token text (diagnostics).
    abstract TokenText: token: int -> string

/// Incremental UTF-8 -> string decoding for streamed token pieces.
[<Sealed>]
type Utf8StreamDecoder() =
    let decoder = UTF8Encoding(false, false).GetDecoder()
    let chars = Array.zeroCreate<char> 64

    member _.Push(bytes: byte[]) : string =
        if bytes.Length = 0 then ""
        else
            let needed = decoder.GetCharCount(bytes, 0, bytes.Length, false)
            let buffer = if needed <= chars.Length then chars else Array.zeroCreate needed
            let n = decoder.GetChars(bytes, 0, bytes.Length, buffer, 0, false)
            String(buffer, 0, n)

    member _.Flush() : string =
        let n = decoder.GetChars(Array.empty<byte>, 0, 0, chars, 0, true)
        String(chars, 0, n)
