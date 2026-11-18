namespace Llama3.Core

open System
open System.Collections.Generic
open System.IO
open System.Text
open System.Text.RegularExpressions

/// The llama2.c tokenizer. Encoding reproduces the C implementation byte for byte, including its
/// dummy-prefix and byte-fallback behaviour, so token ids match llama3.cu exactly.
[<Sealed>]
type SentencePieceTokenizer private (pieces: byte[][], scores: float32[], maxTokenLength: int) =
    /// Byte strings are keyed through Latin-1, which maps every byte to exactly one char.
    static let latin1 = Encoding.Latin1
    let lookup =
        let d = Dictionary<string, int>(pieces.Length, StringComparer.Ordinal)
        pieces |> Array.iteri (fun i p -> d.TryAdd(latin1.GetString p, i) |> ignore)
        d

    let strLookup (bytes: ReadOnlySpan<byte>) =
        match lookup.TryGetValue(latin1.GetString bytes) with
        | true, id -> id
        | _ -> -1

    static member Load(path: string, vocabSize: int) =
        use stream = File.OpenRead path
        use r = new BinaryReader(stream)
        let maxLen = r.ReadInt32()
        let scores = Array.zeroCreate<float32> vocabSize
        let pieces =
            Array.init vocabSize (fun i ->
                scores.[i] <- r.ReadSingle()
                let len = r.ReadInt32()
                r.ReadBytes len)
        SentencePieceTokenizer(pieces, scores, maxLen)

    member _.MaxTokenLength = maxTokenLength
    member _.Scores = scores

    interface ITokenizer with
        member _.VocabSize = pieces.Length
        member _.BosId = Some 1
        member _.EosId = Some 2

        member _.Encode(text: string) =
            let bytes = Encoding.UTF8.GetBytes text
            let tokens = ResizeArray<int>(bytes.Length + 3)
            let buffer = Array.zeroCreate<byte> (maxTokenLength * 2 + 1 + 2)
            if bytes.Length > 0 then tokens.Add(strLookup (ReadOnlySpan<byte>(" "B)))
            let mutable strLen = 0
            for i in 0 .. bytes.Length - 1 do
                let c = bytes.[i]
                if (c &&& 0xC0uy) <> 0x80uy then strLen <- 0
                buffer.[strLen] <- c
                strLen <- strLen + 1
                let next = if i + 1 < bytes.Length then bytes.[i + 1] else 0uy
                if not ((next &&& 0xC0uy) = 0x80uy && strLen < 4) then
                    let id = strLookup (ReadOnlySpan<byte>(buffer, 0, strLen))
                    if id <> -1 then tokens.Add id
                    else
                        // byte_fallback: the first 3 vocab entries are <unk>, <s>, </s>.
                        for b in 0 .. strLen - 1 do tokens.Add(int buffer.[b] + 3)
                    strLen <- 0
            // Greedy merge of the best-scoring adjacent pair until no pair is in the vocabulary.
            let mutable merging = true
            let pair = Array.zeroCreate<byte> (maxTokenLength * 2 + 2)
            while merging do
                let mutable bestScore = -1e10f
                let mutable bestId = -1
                let mutable bestIdx = -1
                for i in 0 .. tokens.Count - 2 do
                    let a = pieces.[tokens.[i]]
                    let b = pieces.[tokens.[i + 1]]
                    let len = a.Length + b.Length
                    let span = if len <= pair.Length then Span<byte>(pair, 0, len) else Span<byte>(Array.zeroCreate len)
                    a.CopyTo(span)
                    b.CopyTo(span.Slice a.Length)
                    let id = strLookup (Span<byte>.op_Implicit span)
                    if id <> -1 && scores.[id] > bestScore then
                        bestScore <- scores.[id]
                        bestId <- id
                        bestIdx <- i
                if bestIdx = -1 then merging <- false
                else
                    tokens.[bestIdx] <- bestId
                    tokens.RemoveAt(bestIdx + 1)
            tokens.ToArray()

        member _.Decode(previous: int, token: int) =
            let piece = pieces.[token]
            let piece =
                if previous = 1 && piece.Length > 0 && piece.[0] = ' 'B then piece.AsSpan(1).ToArray() else piece
            // Raw byte tokens look like "<0x0A>".
            let isHex (b: byte) = (b >= '0'B && b <= '9'B) || (b >= 'A'B && b <= 'F'B) || (b >= 'a'B && b <= 'f'B)
            let hexValue (b: byte) =
                if b <= '9'B then int (b - '0'B) elif b >= 'a'B then int (b - 'a'B) + 10 else int (b - 'A'B) + 10
            let decoded =
                if piece.Length >= 5 && piece.[0] = '<'B && piece.[1] = '0'B && piece.[2] = 'x'B && isHex piece.[3] then
                    let v = if isHex piece.[4] then (hexValue piece.[3] <<< 4) ||| hexValue piece.[4] else hexValue piece.[3]
                    [| byte v |]
                else piece
            // safe_printf: drop lone ASCII control bytes (bytes >= 0x80 are kept so UTF-8 can be reassembled).
            if decoded.Length = 1 then
                let b = decoded.[0]
                let printable = (b >= 0x20uy && b < 0x7Fuy) || (b >= 0x09uy && b <= 0x0Duy) || b >= 0x80uy
                if printable then decoded else Array.empty
            else decoded

        member _.TryGetSpecial(text: string) =
            match text with
            | "<s>" -> Some 1
            | "</s>" -> Some 2
            | _ -> None

        member _.Kind(token: int) =
            if token = 0 then TokenKind.Unknown
            elif token <= 2 then TokenKind.Control
            elif token < 259 then TokenKind.Byte
            else TokenKind.Normal

        member _.TokenText(token: int) = Encoding.UTF8.GetString pieces.[token]
