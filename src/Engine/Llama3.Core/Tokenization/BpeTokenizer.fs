namespace Llama3.Core

open System
open System.Collections.Generic
open System.IO
open System.Text
open System.Text.RegularExpressions

/// Pre-tokenizers as in llama.cpp: a cascade of regular expressions. Each regex splits every
/// piece produced by the previous one into its matches and the gaps between them.
module PreTokenizers =

    let private gpt2Core = @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"

    let private qwen2 =
        @"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

    let private llama3 =
        @"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

    /// Regex cascade and whether a pre-token already in the vocabulary skips BPE (ignore_merges).
    type Spec = { Patterns: string list; IgnoreMerges: bool }

    /// Maps <c>tokenizer.ggml.pre</c> to its definition; unknown values are rejected rather than
    /// silently tokenized with a different pre-tokenizer.
    let forName (pre: string) : Spec =
        match pre with
        | "qwen2" | "deepseek-r1-qwen" -> { Patterns = [ qwen2 ]; IgnoreMerges = false }
        | "llama3" | "llama-v3" | "llama-bpe" | "falcon3" -> { Patterns = [ llama3 ]; IgnoreMerges = true }
        | "gpt-2" | "gpt2" | "phi-2" -> { Patterns = [ gpt2Core ]; IgnoreMerges = false }
        | "smollm" | "starcoder" | "refact" | "command-r" | "codeshell" | "exaone" | "minerva" ->
            { Patterns = [ @"\p{N}"; gpt2Core ]; IgnoreMerges = false }
        | "default" ->
            { Patterns = [ @"[\p{P}\$\+<=>\^~\|]+"; gpt2Core; @"\p{N}+"; "[0-9][0-9][0-9]" ]; IgnoreMerges = false }
        | other ->
            raise (NotSupportedException
                    $"Pre-tokenizer '%s{other}' is not supported (qwen2, llama3/llama-bpe, gpt-2, smollm/starcoder, default)")

/// Applies a regex cascade with code-point semantics. .NET regexes see UTF-16 code units, so a
/// supplementary-plane character (emoji, CJK extension B...) would be two surrogates that match
/// neither \p{L} nor \p{N}. Each such character is replaced by a one-char representative of the
/// same category (letter -> 'a', number -> '0', punctuation -> '!', other -> U+2603) and piece
/// boundaries are mapped back, giving exactly the code-point behaviour of llama.cpp / tokenizers.
[<Sealed>]
type internal CascadeSplitter(patterns: string list) =
    let regexes =
        patterns |> List.map (fun p -> Regex(p, RegexOptions.Compiled ||| RegexOptions.CultureInvariant)) |> Array.ofList

    static let representative (rune: Rune) =
        match Rune.GetUnicodeCategory rune with
        | Globalization.UnicodeCategory.UppercaseLetter | Globalization.UnicodeCategory.LowercaseLetter
        | Globalization.UnicodeCategory.TitlecaseLetter | Globalization.UnicodeCategory.ModifierLetter
        | Globalization.UnicodeCategory.OtherLetter -> 'a'
        | Globalization.UnicodeCategory.DecimalDigitNumber | Globalization.UnicodeCategory.LetterNumber
        | Globalization.UnicodeCategory.OtherNumber -> '0'
        | Globalization.UnicodeCategory.ConnectorPunctuation | Globalization.UnicodeCategory.DashPunctuation
        | Globalization.UnicodeCategory.OpenPunctuation | Globalization.UnicodeCategory.ClosePunctuation
        | Globalization.UnicodeCategory.InitialQuotePunctuation | Globalization.UnicodeCategory.FinalQuotePunctuation
        | Globalization.UnicodeCategory.OtherPunctuation -> '!'
        | _ -> '\u2603'

    /// Splits <paramref name="text"/> into pre-tokens, appended to <paramref name="output"/>.
    member _.Split(text: string, output: List<string>) =
        // Shadow string (one char per code point) and code-point -> UTF-16 offset map, only when needed.
        let hasSurrogates = text.AsSpan().IndexOfAnyInRange('\uD800', '\uDFFF') >= 0
        let shadow, offsets =
            if not hasSurrogates then text, Array.empty<int>
            else
                let sb = StringBuilder(text.Length)
                let map = List<int>(text.Length + 1)
                let mutable i = 0
                while i < text.Length do
                    map.Add i
                    match Rune.DecodeFromUtf16(text.AsSpan(i)) with
                    | Buffers.OperationStatus.Done, rune, consumed ->
                        sb.Append(if consumed = 1 then text.[i] else representative rune) |> ignore
                        i <- i + consumed
                    | _ ->
                        // Lone surrogate: keep it as an "other" character.
                        sb.Append('\u2603') |> ignore
                        i <- i + 1
                map.Add text.Length
                sb.ToString(), map.ToArray()
        let mutable pieces = List<struct (int * int)>()
        pieces.Add(struct (0, shadow.Length))
        for regex in regexes do
            let next = List<struct (int * int)>(pieces.Count * 4)
            for struct (start, length) in pieces do
                let piece = shadow.Substring(start, length)
                let mutable cursor = 0
                let mutable m = regex.Match piece
                while m.Success do
                    if m.Index > cursor then next.Add(struct (start + cursor, m.Index - cursor))
                    if m.Length > 0 then next.Add(struct (start + m.Index, m.Length))
                    cursor <- m.Index + m.Length
                    m <- m.NextMatch()
                if cursor < length then next.Add(struct (start + cursor, length - cursor))
            pieces <- next
        for struct (start, length) in pieces do
            if offsets.Length = 0 then output.Add(text.Substring(start, length))
            else output.Add(text.Substring(offsets.[start], offsets.[start + length] - offsets.[start]))

/// GPT-2 byte <-> unicode bijection used by byte-level BPE vocabularies.
module ByteLevel =
    let byteToChar: char[] =
        let table = Array.zeroCreate<char> 256
        let mutable n = 0
        for b in 0 .. 255 do
            let direct = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)
            if direct then table.[b] <- char b
            else
                table.[b] <- char (256 + n)
                n <- n + 1
        table

    /// Inverse map indexed by char code (all mapped chars are below 0x144); -1 = not a byte char.
    let charToByte: int[] =
        let table = Array.create 0x200 -1
        byteToChar |> Array.iteri (fun b c -> table.[int c] <- b)
        table

/// Byte-level BPE tokenizer. Merges are resolved on token ids: the rank table is keyed by
/// <c>(leftId &lt;&lt;&lt; 32) ||| rightId</c> and yields the merged id directly. Merging uses a priority
/// queue over a doubly linked list of symbols (lowest rank first, then leftmost), O(n log n) per
/// pre-token like llama.cpp, so pathological inputs (long runs of symbols) stay cheap.
[<Sealed>]
type BpeTokenizer(tokens: string[], kinds: TokenKind[], merges: string[], preTokenizer: string, bos: int option, eos: int option) =
    do
        if kinds.Length <> tokens.Length then
            raise (InvalidDataException $"Tokenizer has %d{tokens.Length} tokens but %d{kinds.Length} token types")

    let spec = PreTokenizers.forName preTokenizer
    let splitter = CascadeSplitter(spec.Patterns)

    let tokenToId =
        let d = Dictionary<string, int>(tokens.Length, StringComparer.Ordinal)
        tokens |> Array.iteri (fun i t -> d.TryAdd(t, i) |> ignore)
        d

    let mergeTable =
        let d = Dictionary<int64, struct (int * int)>(merges.Length)
        merges
        |> Array.iteri (fun rank m ->
            let sep = m.IndexOf ' '
            if sep > 0 then
                let a = m.Substring(0, sep)
                let b = m.Substring(sep + 1)
                match tokenToId.TryGetValue a, tokenToId.TryGetValue b, tokenToId.TryGetValue(a + b) with
                | (true, ia), (true, ib), (true, merged) ->
                    d.TryAdd((int64 ia <<< 32) ||| int64 (uint32 ib), struct (rank, merged)) |> ignore
                | _ -> ())
        d

    /// Specials matched literally inside prompts, longest first.
    let specials =
        tokens
        |> Array.mapi (fun i t -> struct (i, t))
        |> Array.filter (fun struct (i, t) ->
            (kinds.[i] = TokenKind.Control || kinds.[i] = TokenKind.UserDefined) && t.Length > 0)
        |> Array.sortByDescending (fun struct (_, t) -> t.Length)

    let specialLookup =
        let d = Dictionary<string, int>(StringComparer.Ordinal)
        for struct (i, t) in specials do d.TryAdd(t, i) |> ignore
        d

    let byteTokenIds =
        ByteLevel.byteToChar
        |> Array.mapi (fun b c ->
            match tokenToId.TryGetValue(string c) with
            | true, id -> id
            | _ -> raise (InvalidDataException $"Byte-level vocabulary lacks the token for byte 0x%02X{b}"))

    /// Pre-tokens up to this length are memoised (long ones are rare and would bloat the cache).
    static let cacheableLength = 32
    let cache = Collections.Concurrent.ConcurrentDictionary<string, int[]>(StringComparer.Ordinal)

    let mergeKey (left: int) (right: int) = (int64 left <<< 32) ||| int64 (uint32 right)

    let bpeWord (word: string) : int[] =
        let n = word.Length
        let ids = Array.init n (fun i -> byteTokenIds.[ByteLevel.charToByte.[int word.[i]]])
        if n = 1 then ids
        else
            let prev = Array.init n (fun i -> i - 1)
            let next = Array.init n (fun i -> if i + 1 < n then i + 1 else -1)
            let alive = Array.create n true
            // Element: (left position, left id, right id); priority: (rank, left position).
            let queue = PriorityQueue<struct (int * int * int), int64>()
            let push (left: int) =
                let right = next.[left]
                if right >= 0 then
                    match mergeTable.TryGetValue(mergeKey ids.[left] ids.[right]) with
                    | true, struct (rank, _) -> queue.Enqueue(struct (left, ids.[left], ids.[right]), (int64 rank <<< 32) ||| int64 left)
                    | _ -> ()
            for i in 0 .. n - 2 do push i
            let mutable entry = struct (0, 0, 0)
            let mutable priority = 0L
            while queue.TryDequeue(&entry, &priority) do
                let struct (left, leftId, rightId) = entry
                let right = next.[left]
                // Skip entries invalidated by an earlier merge of either symbol.
                if alive.[left] && right >= 0 && ids.[left] = leftId && ids.[right] = rightId then
                    let struct (_, merged) = mergeTable.[mergeKey leftId rightId]
                    ids.[left] <- merged
                    alive.[right] <- false
                    next.[left] <- next.[right]
                    if next.[right] >= 0 then prev.[next.[right]] <- left
                    if prev.[left] >= 0 then push prev.[left]
                    push left
            let result = List<int>(n)
            let mutable i = 0
            while i >= 0 do
                result.Add ids.[i]
                i <- next.[i]
            result.ToArray()

    let encodeWord (word: string) (output: List<int>) =
        match (if spec.IgnoreMerges then tokenToId.TryGetValue word else (false, 0)) with
        | true, id -> output.Add id
        | _ when word.Length <= cacheableLength -> output.AddRange(cache.GetOrAdd(word, bpeWord))
        | _ -> output.AddRange(bpeWord word)

    let encodePlain (text: string) (output: List<int>) =
        let pieces = List<string>()
        splitter.Split(text, pieces)
        let sb = StringBuilder()
        for piece in pieces do
            sb.Clear() |> ignore
            for b in Encoding.UTF8.GetBytes piece do
                sb.Append(ByteLevel.byteToChar.[int b]) |> ignore
            encodeWord (sb.ToString()) output

    /// Creates the tokenizer from the <c>tokenizer.ggml.*</c> keys of a GGUF file.
    static member FromGguf(gguf: GgufFile) =
        let model = gguf.TryGetString "tokenizer.ggml.model" |> Option.defaultValue ""
        if model <> "gpt2" then
            raise (NotSupportedException $"GGUF tokenizer model '%s{model}' is not supported (expected byte-level BPE 'gpt2')")
        let tokens =
            match gguf.TryGet "tokenizer.ggml.tokens" with
            | Some(GStringArray t) -> t
            | _ -> raise (InvalidDataException "tokenizer.ggml.tokens missing")
        let kinds =
            match gguf.TryGet "tokenizer.ggml.token_type" with
            | Some(GIntArray k) -> k |> Array.map (fun v -> enum<TokenKind> (int v))
            | _ -> Array.create tokens.Length TokenKind.Normal
        let merges =
            match gguf.TryGet "tokenizer.ggml.merges" with
            | Some(GStringArray m) -> m
            | _ -> Array.empty
        let pre = gguf.TryGetString "tokenizer.ggml.pre" |> Option.defaultValue "default"
        let bos = gguf.TryGetInt "tokenizer.ggml.bos_token_id" |> Option.map int
        let eos = gguf.TryGetInt "tokenizer.ggml.eos_token_id" |> Option.map int
        BpeTokenizer(tokens, kinds, merges, pre, bos, eos)

    member _.PreTokenizer = preTokenizer

    /// Encodes text, turning literal special-token markup into the corresponding ids.
    member _.EncodeWithSpecials(text: string) : int[] =
        let output = List<int>()
        let mutable start = 0
        let mutable i = 0
        while i < text.Length do
            let mutable matched = -1
            let mutable matchedLen = 0
            if text.[i] = '<' || text.[i] = '[' then
                let mutable k = 0
                while matched < 0 && k < specials.Length do
                    let struct (id, s) = specials.[k]
                    if String.CompareOrdinal(text, i, s, 0, s.Length) = 0 then
                        matched <- id
                        matchedLen <- s.Length
                    k <- k + 1
            if matched >= 0 then
                if i > start then encodePlain (text.Substring(start, i - start)) output
                output.Add matched
                i <- i + matchedLen
                start <- i
            else i <- i + 1
        if start < text.Length then encodePlain (text.Substring start) output
        output.ToArray()

    interface ITokenizer with
        member _.VocabSize = tokens.Length
        member _.BosId = bos
        member _.EosId = eos

        member _.Encode(text: string) =
            let output = List<int>(text.Length / 3 + 4)
            encodePlain text output
            output.ToArray()

        member _.Decode(_previous: int, token: int) =
            if token < 0 || token >= tokens.Length then
                raise (ArgumentOutOfRangeException(nameof token, $"Token %d{token} outside the vocabulary (%d{tokens.Length})"))
            match kinds.[token] with
            | TokenKind.Control | TokenKind.Unused | TokenKind.Unknown -> Array.empty
            | TokenKind.UserDefined -> Encoding.UTF8.GetBytes tokens.[token]
            | _ ->
                let text = tokens.[token]
                let bytes = Array.zeroCreate<byte> (text.Length * 3)
                let mutable n = 0
                for c in text do
                    let code = int c
                    let b = if code < ByteLevel.charToByte.Length then ByteLevel.charToByte.[code] else -1
                    if b >= 0 then
                        bytes.[n] <- byte b
                        n <- n + 1
                    else
                        n <- n + Encoding.UTF8.GetBytes(string c, 0, 1, bytes, n)
                Array.sub bytes 0 n

        member _.TryGetSpecial(text: string) =
            match specialLookup.TryGetValue text with
            | true, id -> Some id
            | _ ->
                match tokenToId.TryGetValue text with
                | true, id -> Some id
                | _ -> None

        member _.Kind(token: int) = kinds.[token]
        member _.TokenText(token: int) = tokens.[token]
