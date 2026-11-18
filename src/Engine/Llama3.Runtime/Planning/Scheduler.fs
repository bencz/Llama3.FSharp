namespace Llama3.Runtime

open System
open System.Text
open Llama3

/// Where a component runs. GPU indices refer to <see cref="ExecutionPlan.Gpus"/>.
type Placement =
    | OnCpu
    | OnGpu of gpu: int

type WeightResidency =
    | Resident
    /// Whole layer paged into GPU slots every token (dense models, single GPU).
    | Streamed

/// Placement of one decoder block: attention and FFN are independent, so a MoE can keep its
/// attention on a GPU while its experts run on the CPU.
type LayerPlan =
    { Layer: int
      Attention: Placement
      Ffn: Placement
      Weights: WeightResidency
      KvCache: Placement }

/// The placement of every component, decided before any device memory is allocated.
type ExecutionPlan =
    { Mode: ExecutionMode
      /// GPUs available to the plan (OnGpu i indexes this list).
      Gpus: GpuDevice list
      Capacity: int
      Layers: LayerPlan[]
      Output: Placement
      /// Usable bytes per GPU.
      GpuBudgets: int64[]
      /// Bytes the plan expects to allocate per GPU.
      GpuBytes: int64[]
      /// Device bytes of the streaming slots (included in GpuBytes).
      StagingBytes: int64
      Notes: string list }

    /// GPU indices referenced by any component.
    member p.UsedGpus =
        [ for l in p.Layers do
              for x in [ l.Attention; l.Ffn; l.KvCache ] do
                  match x with
                  | OnGpu g -> yield g
                  | OnCpu -> ()
          match p.Output with
          | OnGpu g -> yield g
          | OnCpu -> () ]
        |> List.distinct
        |> List.sort

    member p.UsesGpu = not p.UsedGpus.IsEmpty

    member p.Describe() =
        let sb = StringBuilder()
        let line (s: string) = sb.AppendLine s |> ignore
        let fmt = MemoryManager.formatBytes
        let name = function OnCpu -> "CPU" | OnGpu g -> $"GPU%d{g}"
        line $"Execution plan: %A{p.Mode}, context %d{p.Capacity} tokens"
        if p.Gpus.IsEmpty then line "  GPU: none"
        p.Gpus
        |> List.iteri (fun i g ->
            let used = if List.contains i p.UsedGpus then "" else " (unused)"
            line $"  GPU%d{i}: %s{g.Name} (%A{g.Kind}:%d{g.Index}, %s{g.Architecture}, %s{fmt g.TotalMemory}), budget %s{fmt p.GpuBudgets.[i]}, planned %s{fmt p.GpuBytes.[i]}%s{used}")
        // Compress consecutive layers with identical placement into ranges.
        let same (a: LayerPlan) (b: LayerPlan) =
            a.Attention = b.Attention && a.Ffn = b.Ffn && a.Weights = b.Weights && a.KvCache = b.KvCache
        let groups =
            p.Layers
            |> Array.fold (fun acc l ->
                match acc with
                | (first, last: LayerPlan) :: rest when same last l && last.Layer + 1 = l.Layer -> (first, l) :: rest
                | _ -> (l, l) :: acc) []
            |> List.rev
        for (first, last) in groups do
            let range = if first.Layer = last.Layer then $"%d{first.Layer}" else $"%d{first.Layer}-%d{last.Layer}"
            let compute =
                if first.Attention = first.Ffn then $"compute=%s{name first.Attention}"
                else $"attention=%s{name first.Attention} ffn=%s{name first.Ffn}"
            line $"  layers %-7s{range} %s{compute} weights=%A{first.Weights} kv=%s{name first.KvCache}"
        line $"  output   compute=%s{name p.Output}"
        if p.StagingBytes > 0L then line $"  streaming slots: %s{fmt p.StagingBytes}"
        for n in p.Notes do line $"  note: %s{n}"
        sb.ToString()

module Scheduler =

    /// Assigns items (layers in pipeline order) to GPUs from the back: the last GPU takes the
    /// longest suffix that fits, the previous GPU the suffix before it, and so on. Unassigned items
    /// (−1) form a CPU-resident prefix. For a contiguous pipeline this maximises the number of items
    /// on GPUs. <paramref name="remaining"/> is updated in place.
    let fillBackward (items: int64[]) (remaining: int64[]) : int[] =
        let assignment = Array.create items.Length -1
        let mutable g = remaining.Length - 1
        let mutable i = items.Length - 1
        while i >= 0 && g >= 0 do
            if items.[i] <= remaining.[g] then
                assignment.[i] <- g
                remaining.[g] <- remaining.[g] - items.[i]
                i <- i - 1
            else g <- g - 1
        assignment

    /// Contiguous pipeline partition proportional to each GPU's room, so every GPU carries a share
    /// of the work and keeps headroom in proportion (used when everything fits). Falls back to
    /// <see cref="fillBackward"/> when the items do not all fit or the proportional cut fragments.
    /// <paramref name="remaining"/> is updated in place.
    let fillProportional (items: int64[]) (remaining: int64[]) : int[] =
        let total = Array.sum items
        let capacity = Array.sum remaining
        let proportional () =
            let room = Array.copy remaining
            let assignment = Array.create items.Length -1
            let share g = float remaining.[g] / float capacity * float total
            let mutable g = 0
            let mutable target = share 0
            let mutable cumulative = 0L
            let mutable ok = true
            for i in 0 .. items.Length - 1 do
                if ok then
                    // Move on once this item's midpoint would cross the GPU's cumulative share.
                    while g < room.Length - 1 && (float cumulative + float items.[i] / 2.0 > target || items.[i] > room.[g]) do
                        g <- g + 1
                        target <- target + share g
                    if items.[i] <= room.[g] then
                        assignment.[i] <- g
                        room.[g] <- room.[g] - items.[i]
                        cumulative <- cumulative + items.[i]
                    else ok <- false
            if ok then
                Array.blit room 0 remaining 0 room.Length
                Some assignment
            else None
        if items.Length = 0 || remaining.Length = 0 then Array.create items.Length -1
        elif remaining.Length > 1 && total <= capacity then
            match proportional () with
            | Some assignment -> assignment
            | None -> fillBackward items remaining
        else fillBackward items remaining

    /// Plans the execution. <paramref name="gpus"/> pairs each usable GPU with its free memory;
    /// <paramref name="stagingBytes"/> returns the device memory needed to stream the given layers.
    let plan
        (options: ModelOptions)
        (estimate: MemoryEstimate)
        (capacity: int)
        (gpus: (GpuDevice * int64) list)
        (stagingBytes: int[] -> int64)
        : Result<ExecutionPlan, string> =
        let n = estimate.LayerCount
        let devices = gpus |> List.map fst
        let budgets =
            gpus
            |> List.map (fun (d, free) -> MemoryManager.gpuBudget free d.TotalMemory options.GpuReserveMB options.GpuMemoryLimitMB)
            |> Array.ofList
        let gpuCount = budgets.Length
        let kvOnGpu = options.KvCache = WithCompute
        let kv = estimate.KvPerLayer
        let layerBytes = Array.init n estimate.LayerWeights
        let placeOf (gpu: int) = if gpu >= 0 then OnGpu gpu else OnCpu
        /// Budget left after the activation workspace each participating GPU allocates.
        let usable () = budgets |> Array.map (fun b -> max 0L (b - estimate.Workspace))
        let planned (remaining: int64[]) (used: int list) =
            Array.init gpuCount (fun g -> if List.contains g used then budgets.[g] - remaining.[g] else 0L)
        let basePlan =
            { Mode = options.ExecutionMode
              Gpus = devices
              Capacity = capacity
              Layers = Array.empty
              Output = OnCpu
              GpuBudgets = budgets
              GpuBytes = Array.zeroCreate gpuCount
              StagingBytes = 0L
              Notes = [] }

        let cpuPlan notes =
            { basePlan with
                Layers = Array.init n (fun i -> { Layer = i; Attention = OnCpu; Ffn = OnCpu; Weights = Resident; KvCache = OnCpu })
                Notes = notes }

        let finish (layers: LayerPlan[]) (output: Placement) (remaining: int64[]) staging notes =
            let p = { basePlan with Layers = layers; Output = output; StagingBytes = staging; Notes = notes }
            { p with GpuBytes = planned remaining p.UsedGpus }

        /// Reserves the classifier on the last GPU when it fits there.
        let reserveOutput (remaining: int64[]) =
            let last = gpuCount - 1
            if remaining.[last] >= estimate.Output then
                remaining.[last] <- remaining.[last] - estimate.Output
                OnGpu last
            else OnCpu

        let layerItems () = Array.init n (fun i -> layerBytes.[i] + (if kvOnGpu then kv else 0L))

        let denseLayers (assignment: int[]) =
            Array.init n (fun i ->
                let p = placeOf assignment.[i]
                { Layer = i; Attention = p; Ffn = p; Weights = Resident; KvCache = (if kvOnGpu then p else OnCpu) })

        /// Dense split: GPUs take the longest suffix of whole layers that fits (contiguous segments,
        /// last GPU last); the CPU runs the prefix. With a forced count k, the last k layers go to
        /// the GPUs; those beyond the budget are streamed (StreamWeights, one GPU, dense) or else
        /// over-commit the GPU with the most room left, and the excess is charged and reported.
        let splitPlan (forced: int option) =
            let remaining = usable ()
            let output = reserveOutput remaining
            let items = layerItems ()
            match forced with
            | Some k when k <= 0 -> cpuPlan [ "GpuLayers = 0: every layer runs on the CPU" ]
            | _ ->
                let streamed = Collections.Generic.HashSet<int>()
                let mutable staging = 0L
                let mutable overcommitted = false
                let assignment =
                    match forced with
                    | None -> fillBackward items remaining
                    | Some k ->
                        let k = min n k
                        let tail = fillBackward items.[n - k ..] remaining
                        let assignment = Array.append (Array.create (n - k) -1) tail
                        let unplaced = [| for i in n - k .. n - 1 do if assignment.[i] < 0 then yield i |]
                        if unplaced.Length > 0 then
                            if options.Offload = StreamWeights && gpuCount = 1 && not estimate.IsMoe then
                                staging <- stagingBytes unplaced
                                remaining.[0] <- remaining.[0] - staging - (if kvOnGpu then kv * int64 unplaced.Length else 0L)
                                for i in unplaced do
                                    streamed.Add i |> ignore
                                    assignment.[i] <- 0
                            else
                                for i in unplaced do
                                    let g = Array.findIndex ((=) (Array.max remaining)) remaining
                                    remaining.[g] <- remaining.[g] - items.[i]
                                    assignment.[i] <- g
                            overcommitted <- remaining |> Array.exists (fun r -> r < 0L)
                        assignment
                let layers =
                    denseLayers assignment
                    |> Array.map (fun l -> if streamed.Contains l.Layer then { l with Weights = Streamed } else l)
                let onGpu = assignment |> Array.filter (fun g -> g >= 0) |> Array.length
                finish layers output remaining staging
                    [ if onGpu < n then yield $"%d{n - onGpu} of %d{n} layers run on the CPU"
                      if streamed.Count > 0 then yield $"%d{streamed.Count} forced GPU layers stream their weights from pinned host memory"
                      if overcommitted then yield "forced GPU layer count exceeds the estimated budget (allocation may fail)"
                      if output = OnCpu then yield "classifier runs on the CPU (does not fit on the last GPU)" ]

        /// Every layer on GPUs. Several GPUs: all resident, partitioned across them. One GPU: the
        /// first k layers resident and the rest streamed from pinned host memory (dense only).
        let gpuComputePlan () =
            let remaining = usable ()
            let output = reserveOutput remaining
            if output = OnCpu then None
            elif gpuCount > 1 || estimate.IsMoe then
                let assignment = fillProportional (layerItems ()) remaining
                if assignment |> Array.forall (fun g -> g >= 0) then Some(finish (denseLayers assignment) output remaining 0L [])
                else None
            else
                let available = remaining.[0] - (if kvOnGpu then kv * int64 n else 0L)
                let cost k =
                    let resident = Array.sum layerBytes.[0 .. k - 1]
                    let staging = if k < n then stagingBytes [| k .. n - 1 |] else 0L
                    resident + staging, staging
                let mutable k = n
                while k >= 0 && fst (cost k) > available do k <- k - 1
                if k < 0 then None
                else
                    let total, staging = cost k
                    remaining.[0] <- available - total
                    let layers =
                        Array.init n (fun i ->
                            { Layer = i
                              Attention = OnGpu 0
                              Ffn = OnGpu 0
                              Weights = (if i < k then Resident else Streamed)
                              KvCache = (if kvOnGpu then OnGpu 0 else OnCpu) })
                    Some(finish layers output remaining staging
                            (if k < n then [ $"%d{n - k} of %d{n} layers stream their weights from pinned host memory" ] else []))

        /// Sparse split: attention + KV of every layer spread over the GPUs in proportion to their
        /// room (small, used every token), then each layer's experts join its attention GPU while
        /// budget remains (no extra transfer), else any other GPU with room (one activation round
        /// trip, still far faster than CPU experts), else the CPU. Layers are visited from the back
        /// so a forced count keeps the GPU experts at the end of the pipeline.
        let moePlan (forced: int option) =
            let remaining = usable ()
            let output = reserveOutput remaining
            let attentionItems = Array.init n (fun i -> estimate.AttentionWeights.[i] + (if kvOnGpu then kv else 0L))
            let attention = fillProportional attentionItems remaining
            if output = OnCpu || attention |> Array.exists (fun g -> g < 0) then
                let p = splitPlan forced
                { p with Notes = "attention of every layer does not fit on the GPUs: splitting whole layers" :: p.Notes }
            else
                let ffn = Array.create n -1
                let limit = forced |> Option.defaultValue n
                let mutable placed = 0
                let mutable crossing = 0
                let assign i g =
                    ffn.[i] <- g
                    remaining.[g] <- remaining.[g] - estimate.FfnWeights.[i]
                    placed <- placed + 1
                    if g <> attention.[i] then crossing <- crossing + 1
                let roomiest () = Array.findIndex ((=) (Array.max remaining)) remaining
                match forced with
                | Some _ ->
                    // The last k layers get GPU experts: own GPU, else the roomiest (over-commit).
                    for i in n - 1 .. -1 .. max 0 (n - limit) do
                        assign i (if estimate.FfnWeights.[i] <= remaining.[attention.[i]] then attention.[i] else roomiest ())
                | None ->
                    // Pass 1: experts next to their attention (no transfer). Pass 2: leftovers on
                    // any GPU that still has room.
                    for i in n - 1 .. -1 .. 0 do
                        if estimate.FfnWeights.[i] <= remaining.[attention.[i]] then assign i attention.[i]
                    for i in n - 1 .. -1 .. 0 do
                        if ffn.[i] < 0 then
                            let g = roomiest ()
                            if estimate.FfnWeights.[i] <= remaining.[g] then assign i g
                let layers =
                    Array.init n (fun i ->
                        { Layer = i
                          Attention = OnGpu attention.[i]
                          Ffn = placeOf ffn.[i]
                          Weights = Resident
                          KvCache = (if kvOnGpu then OnGpu attention.[i] else OnCpu) })
                finish layers output remaining 0L
                    [ if placed < n then yield $"experts of %d{n - placed} of %d{n} layers run on the CPU (attention, KV and classifier stay on the GPUs)"
                      if crossing > 0 then yield $"experts of %d{crossing} layers run on another GPU than their attention"
                      if remaining |> Array.exists (fun r -> r < 0L) then yield "forced GPU expert layer count exceeds the estimated budget (allocation may fail)" ]

        match options.ExecutionMode, gpuCount with
        | CpuOnly, _ -> Ok(cpuPlan [])
        | _, 0 when options.UseCpuFallback -> Ok(cpuPlan [ "no usable GPU found: running on the CPU" ])
        | mode, 0 -> Error $"%A{mode} requested but no usable GPU was found (enable UseCpuFallback to run on the CPU)"
        | GpuOnly, _ ->
            match gpuComputePlan () with
            | Some p -> Ok p
            | None when options.UseCpuFallback ->
                let p = if estimate.IsMoe then moePlan None else splitPlan None
                let reason =
                    if estimate.IsMoe then "experts do not fit on the GPUs (MoE experts are not streamed): splitting with the CPU"
                    elif gpuCount > 1 then "the model does not fit across the GPUs (streaming is single-GPU): splitting with the CPU"
                    else "GPU too small even for streaming: splitting with the CPU"
                Ok { p with Notes = reason :: p.Notes }
            | None -> Error "The model does not fit in the GPU budget"
        | Hybrid, _ when estimate.IsMoe -> Ok(moePlan options.GpuLayers)
        | Hybrid, _ ->
            match options.Offload, options.GpuLayers with
            | StreamWeights, None ->
                match gpuComputePlan () with
                | Some p -> Ok p
                | None -> Ok(splitPlan None)
            | _, forced -> Ok(splitPlan forced)

    /// Ensures a requested capacity is sane for the model.
    let capacityFor (options: ModelOptions) (modelContext: int) =
        match options.ContextLength with
        | Some c when c > 0 -> min c modelContext
        | Some c -> raise (ArgumentOutOfRangeException("ContextLength", c, "must be positive"))
        | None -> min modelContext 4096
