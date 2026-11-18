namespace Llama3.Backend.Cuda

open System
open ILGPU
open ILGPU.Runtime.Cuda

/// Owns the ILGPU context and enumerates CUDA devices. Creation never throws when no driver is
/// installed: <see cref="CudaPlatform.TryCreate"/> returns the reason instead.
[<Sealed>]
type CudaPlatform private (context: Context) =
    let devices =
        // DeviceCollection is a byref-like struct: copy it out eagerly.
        let found = ResizeArray<CudaDevice>()
        let collection = context.GetCudaDevices()
        for i in 0 .. collection.Count - 1 do found.Add collection.[i]
        found.ToArray()
        |> Array.mapi (fun i (d: CudaDevice) ->
            { Index = i
              Name = d.Name
              TotalMemory = d.MemorySize
              Architecture = (if d.Architecture.HasValue then string d.Architecture.Value else "unknown")
              Multiprocessors = d.NumMultiprocessors })

    member _.Devices = devices

    member _.CreateBackend(index: int) : CudaBackend =
        if index < 0 || index >= devices.Length then
            invalidArg (nameof index) $"CUDA device %d{index} does not exist (%d{devices.Length} available)"
        // CudaBackend takes ownership of the accelerator (and releases it if its constructor fails).
        new CudaBackend(context.CreateCudaAccelerator(index), devices.[index])

    /// Opens the CUDA driver. Never throws: a machine without the NVIDIA driver (or without any
    /// device) yields Error with the reason, which callers surface instead of silently hiding GPUs.
    static member TryCreate() : Result<CudaPlatform, string> =
        let mutable context = Unchecked.defaultof<Context>
        try
            context <- Context.Create(fun b -> b.Cuda().EnableAlgorithms().Optimize(OptimizationLevel.O2) |> ignore)
            let platform = new CudaPlatform(context)
            if platform.Devices.Length = 0 then
                (platform :> IDisposable).Dispose()
                Error "the CUDA driver reports no devices"
            else Ok platform
        with e ->
            if not (obj.ReferenceEquals(context, null)) then context.Dispose()
            Error $"CUDA unavailable: %s{e.GetBaseException().Message}"

    interface IDisposable with
        member _.Dispose() = context.Dispose()
