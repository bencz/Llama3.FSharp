namespace Llama3.Backend.Cuda

open ILGPU

/// Matrix-vector products: one warp per output row; lanes walk the row with a 32-element stride
/// so every load instruction of a warp touches a contiguous range (coalesced). Quantized formats
/// read whole 32-bit words of the padded device layouts (see GpuLayout) and stage x in shared memory
/// when the row fits (the limit comes from the device, see CudaBackend).
/// Every kernel is a curried module function: it compiles to a plain static method, which F# 11
/// binds directly to a delegate (no closure) — the form ILGPU compiles to PTX. The project disables
/// F# tail calls, whose IL prefix ILGPU rejects.
[<RequireQualifiedAccess>]
module MatVecKernels =

    /// Storage format codes passed to the word kernels (launch-uniform: no divergence).
    [<Literal>]
    let FormatF16 = 0
    [<Literal>]
    let FormatQ8_0 = 1
    [<Literal>]
    let FormatQ4_K = 2
    [<Literal>]
    let FormatQ6_K = 3
    [<Literal>]
    let FormatBF16 = 4

    /// Group size of the matvec kernels (8 warps).
    [<Literal>]
    let MatVecGroupSize = 256
    /// One row per warp.
    [<Literal>]
    let MatVecRowsPerGroup = MatVecGroupSize / 32

    let matVecF32 (output: ArrayView<float32>) (x: ArrayView<float32>) (w: ArrayView<float32>) (rows: int) (cols: int) =
        let row = Grid.IdxX * (Group.DimX / 32) + Warp.WarpIdx
        if row < rows then
            let lane = Warp.LaneIdx
            let rowBase = int64 row * int64 cols
            let mutable s0 = 0.0f
            let mutable s1 = 0.0f
            let mutable j = lane
            while j + 32 < cols do
                s0 <- s0 + w.[rowBase + int64 j] * x.[j]
                s1 <- s1 + w.[rowBase + int64 (j + 32)] * x.[j + 32]
                j <- j + 64
            if j < cols then s0 <- s0 + w.[rowBase + int64 j] * x.[j]
            let total = DeviceMath.warpAllSum (s0 + s1)
            if lane = 0 then output.[row] <- total

    /// F16: one word (two halves) per lane step.
    let dotF16 (w: ArrayView<uint32>) (x: ArrayView<float32>) (row: int) (cols: int) (lane: int) : float32 =
        let words = cols / 2
        // 64-bit offset math once per row; the inner loop indexes with 32-bit integers.
        let r = w.SubView(int64 row * int64 words, int64 words)
        let mutable s = 0.0f
        let mutable j = lane
        while j < words do
            let v = r.[j]
            s <- s + DeviceMath.halfToFloat (v &&& 0xFFFFu) * x.[2 * j] + DeviceMath.halfToFloat (v >>> 16) * x.[2 * j + 1]
            j <- j + 32
        s

    /// BF16: a bfloat16 is the upper half of a float32, so widening is a shift or a mask.
    let dotBF16 (w: ArrayView<uint32>) (x: ArrayView<float32>) (row: int) (cols: int) (lane: int) : float32 =
        let words = cols / 2
        let r = w.SubView(int64 row * int64 words, int64 words)
        let mutable s = 0.0f
        let mutable j = lane
        while j < words do
            let v = r.[j]
            s <- s + Interop.IntAsFloat(v <<< 16) * x.[2 * j] + Interop.IntAsFloat(v &&& 0xFFFF0000u) * x.[2 * j + 1]
            j <- j + 32
        s

    /// Q8_0, GPU layout [d:f16][pad][qs:32×i8] = 9 words/block. Lanes 8i..8i+7 cover block b+i
    /// (4 quants per word): 4 blocks per warp step, fully coalesced.
    let dotQ8_0 (w: ArrayView<uint32>) (x: ArrayView<float32>) (row: int) (cols: int) (lane: int) : float32 =
        let blocks = cols / 32
        let r = w.SubView(int64 row * int64 (blocks * 9), int64 (blocks * 9))
        let word = lane &&& 7
        let mutable s = 0.0f
        let mutable b = lane >>> 3
        while b < blocks do
            let blockBase = b * 9
            let d = DeviceMath.halfToFloat r.[blockBase]
            let q = r.[blockBase + 1 + word]
            let e = b * 32 + word * 4
            s <- s + d * (DeviceMath.sByteOf q 0 * x.[e] + DeviceMath.sByteOf q 1 * x.[e + 1]
                          + DeviceMath.sByteOf q 2 * x.[e + 2] + DeviceMath.sByteOf q 3 * x.[e + 3])
            b <- b + 4
        s

    /// Q4_K, GPU layout (148 B = 37 words: [d|dmin][scales×8 u8][mins×8 u8][qs×32 words]).
    /// Lane l reads qs word l: its 8 nibbles belong to sub-blocks 2p and 2p+1 (p = l/8), whose
    /// pre-unpacked scale and min bytes sit side by side in one word each — no divergent decode.
    let dotQ4_K (w: ArrayView<uint32>) (x: ArrayView<float32>) (row: int) (cols: int) (lane: int) : float32 =
        let superBlocks = cols / 256
        let r = w.SubView(int64 row * int64 (superBlocks * 37), int64 (superBlocks * 37))
        let p = lane >>> 3
        let lp = (lane &&& 7) * 4
        // Bytes 2p and 2p+1 of the 8-byte scale (and min) array: word 1 + p/2, shift 16·(p&1).
        let scaleWord = 1 + (p >>> 1)
        let shift = 16 * (p &&& 1)
        let mutable s = 0.0f
        for sb in 0 .. superBlocks - 1 do
            let o = sb * 37
            let dm = r.[o]
            let scales = r.[o + scaleWord] >>> shift
            let mins = r.[o + 2 + scaleWord] >>> shift
            let q = r.[o + 5 + lane]
            let e = sb * 256 + p * 64 + lp
            let mutable dotLo = 0.0f
            let mutable sumLo = 0.0f
            let mutable dotHi = 0.0f
            let mutable sumHi = 0.0f
            for i in 0 .. 3 do
                let b = q >>> (8 * i)
                let xl = x.[e + i]
                let xh = x.[e + 32 + i]
                dotLo <- dotLo + float32 (int (b &&& 0xFu)) * xl
                dotHi <- dotHi + float32 (int ((b >>> 4) &&& 0xFu)) * xh
                sumLo <- sumLo + xl
                sumHi <- sumHi + xh
            let d = DeviceMath.halfToFloat dm
            let dmin = DeviceMath.halfToFloat (dm >>> 16)
            let sc0 = float32 (int (scales &&& 0xFFu))
            let sc1 = float32 (int ((scales >>> 8) &&& 0xFFu))
            let m0 = float32 (int (mins &&& 0xFFu))
            let m1 = float32 (int ((mins >>> 8) &&& 0xFFu))
            s <- s + d * (sc0 * dotLo + sc1 * dotHi) - dmin * (m0 * sumLo + m1 * sumHi)
        s

    /// Q6_K, GPU layout (212 B = 53 words: [ql×32][qh×16][scales×4][d|pad]).
    /// Lane = 16n + 8g + j: one ql word + one qh word → 8 values (k = g and k = g + 2), 2 scales.
    let dotQ6_K (w: ArrayView<uint32>) (x: ArrayView<float32>) (row: int) (cols: int) (lane: int) : float32 =
        let superBlocks = cols / 256
        let r = w.SubView(int64 row * int64 (superBlocks * 53), int64 (superBlocks * 53))
        let n = lane >>> 4
        let g = (lane >>> 3) &&& 1
        let j = lane &&& 7
        let scLo = 8 * n + (j >>> 2) + 2 * g
        let scHi = scLo + 4
        let mutable s = 0.0f
        for sb in 0 .. superBlocks - 1 do
            let o = sb * 53
            let ql = r.[o + 16 * n + 8 * g + j]
            let qh = r.[o + 32 + 8 * n + j]
            let sLo = DeviceMath.sByteOf r.[o + 48 + (scLo >>> 2)] (scLo &&& 3)
            let sHi = DeviceMath.sByteOf r.[o + 48 + (scHi >>> 2)] (scHi &&& 3)
            let d = DeviceMath.halfToFloat r.[o + 52]
            let e = sb * 256 + 128 * n + 32 * g + 4 * j
            let mutable dotLo = 0.0f
            let mutable dotHi = 0.0f
            for i in 0 .. 3 do
                let lb = (ql >>> (8 * i)) &&& 0xFFu
                let hb = (qh >>> (8 * i)) &&& 0xFFu
                let qLo = int ((lb &&& 0xFu) ||| (((hb >>> (2 * g)) &&& 3u) <<< 4)) - 32
                let qHi = int ((lb >>> 4) ||| (((hb >>> (4 + 2 * g)) &&& 3u) <<< 4)) - 32
                dotLo <- dotLo + float32 qLo * x.[e + i]
                dotHi <- dotHi + float32 qHi * x.[e + 64 + i]
            s <- s + d * (sLo * dotLo + sHi * dotHi)
        s

    /// Dispatch on a launch-uniform format code (no divergence).
    let dotWords (format: int) (w: ArrayView<uint32>) (x: ArrayView<float32>) (row: int) (cols: int) (lane: int) : float32 =
        if format = FormatQ8_0 then dotQ8_0 w x row cols lane
        elif format = FormatQ4_K then dotQ4_K w x row cols lane
        elif format = FormatQ6_K then dotQ6_K w x row cols lane
        elif format = FormatBF16 then dotBF16 w x row cols lane
        else dotF16 w x row cols lane

    /// Word-format matvec with x staged in dynamic shared memory once per group; warps then walk
    /// rows with a grid-stride loop, so global traffic is essentially the weights alone.
    let matVecWordsShared (output: ArrayView<float32>) (x: ArrayView<float32>) (w: ArrayView<uint32>) (rows: int) (cols: int) (format: int) =
        let xs = SharedMemory.GetDynamic<float32>()
        let mutable i = Group.IdxX
        while i < cols do
            xs.[i] <- x.[i]
            i <- i + Group.DimX
        Group.Barrier()
        let warps = Group.DimX / 32
        let lane = Warp.LaneIdx
        let mutable row = Grid.IdxX * warps + Warp.WarpIdx
        let stride = Grid.DimX * warps
        while row < rows do
            let total = DeviceMath.warpAllSum (dotWords format w xs row cols lane)
            if lane = 0 then output.[row] <- total
            row <- row + stride

    /// Same as <c>matVecWordsShared</c> reading x from global memory: rows longer than the
    /// device's shared memory per group can stage.
    let matVecWordsGlobal (output: ArrayView<float32>) (x: ArrayView<float32>) (w: ArrayView<uint32>) (rows: int) (cols: int) (format: int) =
        let warps = Group.DimX / 32
        let lane = Warp.LaneIdx
        let mutable row = Grid.IdxX * warps + Warp.WarpIdx
        let stride = Grid.DimX * warps
        while row < rows do
            let total = DeviceMath.warpAllSum (dotWords format w x row cols lane)
            if lane = 0 then output.[row] <- total
            row <- row + stride

    // ------------------------------------------------------------------------------------------
    // Row-pair kernels (K-quants): each warp evaluates two rows per step. The activation fragment
    // of a lane (8 floats from shared memory) and, for Q4_K, its sums are loaded once and reused by
    // both rows — half the shared-memory traffic per element and two independent FMA chains.
    // Inner loops are fully unrolled with constant shifts.
    // ------------------------------------------------------------------------------------------

    /// One Q4_K super-block of one row against a lane's activation fragment (x: low nibbles,
    /// y: high nibbles, sums for the mins).
    let q4kPartial
        (r: ArrayView<uint32>) (o: int) (lane: int) (scaleWord: int) (shift: int)
        (x0: float32) (x1: float32) (x2: float32) (x3: float32)
        (y0: float32) (y1: float32) (y2: float32) (y3: float32)
        (sumLo: float32) (sumHi: float32)
        : float32 =
        let dm = r.[o]
        let scales = r.[o + scaleWord] >>> shift
        let mins = r.[o + 2 + scaleWord] >>> shift
        let q = r.[o + 5 + lane]
        let dotLo =
            float32 (int (q &&& 0xFu)) * x0 + float32 (int ((q >>> 8) &&& 0xFu)) * x1
            + float32 (int ((q >>> 16) &&& 0xFu)) * x2 + float32 (int ((q >>> 24) &&& 0xFu)) * x3
        let dotHi =
            float32 (int ((q >>> 4) &&& 0xFu)) * y0 + float32 (int ((q >>> 12) &&& 0xFu)) * y1
            + float32 (int ((q >>> 20) &&& 0xFu)) * y2 + float32 (int (q >>> 28)) * y3
        let d = DeviceMath.halfToFloat dm
        let dmin = DeviceMath.halfToFloat (dm >>> 16)
        d * (float32 (int (scales &&& 0xFFu)) * dotLo + float32 (int ((scales >>> 8) &&& 0xFFu)) * dotHi)
        - dmin * (float32 (int (mins &&& 0xFFu)) * sumLo + float32 (int ((mins >>> 8) &&& 0xFFu)) * sumHi)

    let matVecQ4KPairs (output: ArrayView<float32>) (x: ArrayView<float32>) (w: ArrayView<uint32>) (rows: int) (cols: int) =
        let xs = SharedMemory.GetDynamic<float32>()
        let mutable i = Group.IdxX
        while i < cols do
            xs.[i] <- x.[i]
            i <- i + Group.DimX
        Group.Barrier()
        let warps = Group.DimX / 32
        let lane = Warp.LaneIdx
        let superBlocks = cols / 256
        let rowWords = superBlocks * 37
        let p = lane >>> 3
        let lp = (lane &&& 7) * 4
        let scaleWord = 1 + (p >>> 1)
        let shift = 16 * (p &&& 1)
        let mutable pair = Grid.IdxX * warps + Warp.WarpIdx
        let stride = Grid.DimX * warps
        while 2 * pair < rows do
            let row0 = 2 * pair
            let row1 = if row0 + 1 < rows then row0 + 1 else row0
            let r0 = w.SubView(int64 row0 * int64 rowWords, int64 rowWords)
            let r1 = w.SubView(int64 row1 * int64 rowWords, int64 rowWords)
            let mutable s0 = 0.0f
            let mutable s1 = 0.0f
            for sb in 0 .. superBlocks - 1 do
                let e = sb * 256 + p * 64 + lp
                let x0 = xs.[e]
                let x1 = xs.[e + 1]
                let x2 = xs.[e + 2]
                let x3 = xs.[e + 3]
                let y0 = xs.[e + 32]
                let y1 = xs.[e + 33]
                let y2 = xs.[e + 34]
                let y3 = xs.[e + 35]
                let sumLo = (x0 + x1) + (x2 + x3)
                let sumHi = (y0 + y1) + (y2 + y3)
                let o = sb * 37
                s0 <- s0 + q4kPartial r0 o lane scaleWord shift x0 x1 x2 x3 y0 y1 y2 y3 sumLo sumHi
                s1 <- s1 + q4kPartial r1 o lane scaleWord shift x0 x1 x2 x3 y0 y1 y2 y3 sumLo sumHi
            let t0 = DeviceMath.warpAllSum s0
            let t1 = DeviceMath.warpAllSum s1
            if lane = 0 then
                output.[row0] <- t0
                if row1 <> row0 then output.[row1] <- t1
            pair <- pair + stride

    /// One Q6_K super-block of one row (lane = 16n + 8g + j) against the lane's fragment
    /// (x: elements of k = g, y: elements of k = g + 2).
    let q6kPartial
        (r: ArrayView<uint32>) (o: int) (n: int) (g: int) (j: int) (scLo: int) (scHi: int)
        (x0: float32) (x1: float32) (x2: float32) (x3: float32)
        (y0: float32) (y1: float32) (y2: float32) (y3: float32)
        : float32 =
        let ql = r.[o + 16 * n + 8 * g + j]
        let qh = r.[o + 32 + 8 * n + j] >>> (2 * g)
        let sLo = DeviceMath.sByteOf r.[o + 48 + (scLo >>> 2)] (scLo &&& 3)
        let sHi = DeviceMath.sByteOf r.[o + 48 + (scHi >>> 2)] (scHi &&& 3)
        let d = DeviceMath.halfToFloat r.[o + 52]
        let dotLo =
            float32 (int ((ql &&& 0xFu) ||| ((qh &&& 3u) <<< 4)) - 32) * x0
            + float32 (int (((ql >>> 8) &&& 0xFu) ||| (((qh >>> 8) &&& 3u) <<< 4)) - 32) * x1
            + float32 (int (((ql >>> 16) &&& 0xFu) ||| (((qh >>> 16) &&& 3u) <<< 4)) - 32) * x2
            + float32 (int (((ql >>> 24) &&& 0xFu) ||| (((qh >>> 24) &&& 3u) <<< 4)) - 32) * x3
        let dotHi =
            float32 (int (((ql >>> 4) &&& 0xFu) ||| (((qh >>> 4) &&& 3u) <<< 4)) - 32) * y0
            + float32 (int (((ql >>> 12) &&& 0xFu) ||| (((qh >>> 12) &&& 3u) <<< 4)) - 32) * y1
            + float32 (int (((ql >>> 20) &&& 0xFu) ||| (((qh >>> 20) &&& 3u) <<< 4)) - 32) * y2
            + float32 (int ((ql >>> 28) ||| (((qh >>> 28) &&& 3u) <<< 4)) - 32) * y3
        d * (sLo * dotLo + sHi * dotHi)

    let matVecQ6KPairs (output: ArrayView<float32>) (x: ArrayView<float32>) (w: ArrayView<uint32>) (rows: int) (cols: int) =
        let xs = SharedMemory.GetDynamic<float32>()
        let mutable i = Group.IdxX
        while i < cols do
            xs.[i] <- x.[i]
            i <- i + Group.DimX
        Group.Barrier()
        let warps = Group.DimX / 32
        let lane = Warp.LaneIdx
        let superBlocks = cols / 256
        let rowWords = superBlocks * 53
        let n = lane >>> 4
        let g = (lane >>> 3) &&& 1
        let j = lane &&& 7
        let scLo = 8 * n + (j >>> 2) + 2 * g
        let scHi = scLo + 4
        let mutable pair = Grid.IdxX * warps + Warp.WarpIdx
        let stride = Grid.DimX * warps
        while 2 * pair < rows do
            let row0 = 2 * pair
            let row1 = if row0 + 1 < rows then row0 + 1 else row0
            let r0 = w.SubView(int64 row0 * int64 rowWords, int64 rowWords)
            let r1 = w.SubView(int64 row1 * int64 rowWords, int64 rowWords)
            let mutable s0 = 0.0f
            let mutable s1 = 0.0f
            for sb in 0 .. superBlocks - 1 do
                let e = sb * 256 + 128 * n + 32 * g + 4 * j
                let x0 = xs.[e]
                let x1 = xs.[e + 1]
                let x2 = xs.[e + 2]
                let x3 = xs.[e + 3]
                let y0 = xs.[e + 64]
                let y1 = xs.[e + 65]
                let y2 = xs.[e + 66]
                let y3 = xs.[e + 67]
                let o = sb * 53
                s0 <- s0 + q6kPartial r0 o n g j scLo scHi x0 x1 x2 x3 y0 y1 y2 y3
                s1 <- s1 + q6kPartial r1 o n g j scLo scHi x0 x1 x2 x3 y0 y1 y2 y3
            let t0 = DeviceMath.warpAllSum s0
            let t1 = DeviceMath.warpAllSum s1
            if lane = 0 then
                output.[row0] <- t0
                if row1 <> row0 then output.[row1] <- t1
            pair <- pair + stride
