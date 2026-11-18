namespace Llama3.Runtime

open System
open Llama3
open Llama3.Core
open Llama3.Backend.Cpu
open Llama3.Backend.Cuda

/// A GPU visible to the runtime.
type GpuDevice =
    { Kind: DeviceKind
      Index: int
      Name: string
      TotalMemory: int64
      Architecture: string }

/// Discovers devices and opens backends. CUDA discovery is lazy and failure-tolerant: a machine
/// without the NVIDIA driver simply reports no GPUs. Future vendors (ROCm, OpenCL) plug in here.
[<Sealed>]
type DeviceManager() =
    let cuda = lazy (CudaPlatform.TryCreate())
    let mutable disposed = false

    /// Why no CUDA device is listed (None when CUDA is available).
    member _.CudaUnavailableReason =
        match cuda.Value with
        | Ok _ -> None
        | Error reason -> Some reason

    member _.Gpus: GpuDevice list =
        match cuda.Value with
        | Error _ -> []
        | Ok p ->
            p.Devices
            |> Array.map (fun d ->
                { Kind = Cuda
                  Index = d.Index
                  Name = d.Name
                  TotalMemory = d.TotalMemory
                  Architecture = d.Architecture })
            |> Array.toList

    /// Resolves the requested devices to available GPUs, in the requested order (empty request =
    /// every GPU). Unknown device ids are reported rather than silently dropped.
    member this.ResolveGpus(requested: Device list) : Result<GpuDevice list, string> =
        let gpus = this.Gpus
        match requested |> List.filter (fun d -> d <> CPU) with
        | [] when requested.IsEmpty -> Ok gpus
        | [] -> Ok []
        | wanted ->
            let resolved =
                wanted |> List.map (fun d ->
                    match d with
                    | CUDA id ->
                        match gpus |> List.tryFind (fun g -> g.Kind = Cuda && g.Index = id) with
                        | Some g -> Ok g
                        | None -> Error id
                    | CPU -> Error -1)
            match resolved |> List.choose (function Error id -> Some id | Ok _ -> None) with
            | [] -> Ok(resolved |> List.choose (function Ok g -> Some g | Error _ -> None) |> List.distinct)
            | missing ->
                let reason = this.CudaUnavailableReason |> Option.map (fun r -> $" (%s{r})") |> Option.defaultValue ""
                Error $"CUDA device(s) %A{missing} not found (%d{gpus.Length} available)%s{reason}"

    member _.OpenGpu(device: GpuDevice) : IComputeBackend =
        match device.Kind, cuda.Value with
        | Cuda, Ok p -> p.CreateBackend(device.Index) :> IComputeBackend
        | Cuda, Error reason -> raise (NotSupportedException $"CUDA device %d{device.Index}: %s{reason}")
        | kind, _ -> raise (NotSupportedException $"No GPU backend for %A{kind} device %d{device.Index}")

    member _.OpenCpu(options: ModelOptions) : CpuBackend =
        new CpuBackend({ Threads = options.CpuThreads; QuantizedActivations = options.QuantizedCpuActivations; Isa = options.CpuIsa })

    interface IDisposable with
        member _.Dispose() =
            if not disposed then
                disposed <- true
                if cuda.IsValueCreated then
                    match cuda.Value with
                    | Ok p -> (p :> IDisposable).Dispose()
                    | Error _ -> ()
