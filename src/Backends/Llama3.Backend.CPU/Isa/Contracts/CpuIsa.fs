namespace Llama3.Backend.Cpu

open System
open System.Numerics
open System.Runtime.Intrinsics.Arm
open System.Runtime.Intrinsics.X86

/// Instruction-set families with a dedicated kernel implementation.
type CpuIsa =
    /// Vector&lt;T&gt; float path + scalar integer path. Runs on any .NET platform.
    | Portable
    /// x86-64 with AVX2 + FMA (AVX-512F widens the float path when present).
    | X86Avx2
    /// x86-64 with AVX-512 F/BW and VNNI (VPDPBUSD): Ice Lake+, Sapphire Rapids+, AMD Zen 4/5.
    | X86Avx512Vnni
    /// ARM64 AdvSimd (NEON); int8 dot products use SDOT when the CPU has the DotProd extension.
    | ArmNeon

/// The ISA-specific hot loops. Everything else in the CPU backend is written once on top of this.
/// Row arguments are the raw ggml blocks of one weight row (array- or file-mapped memory);
/// implementations validate the row and activation lengths before reading.
type ICpuKernels =
    abstract Isa: CpuIsa
    abstract Description: string
    /// sum a[i]*b[i]
    abstract DotF32: a: ReadOnlySpan<float32> * b: ReadOnlySpan<float32> -> float32
    /// y += alpha*x
    abstract Axpy: alpha: float32 * x: ReadOnlySpan<float32> * y: Span<float32> -> unit
    abstract DotQ8_0: row: ReadOnlySpan<byte> * blocks: int * xq: QuantizedActivations -> float32
    abstract DotQ4_K: row: ReadOnlySpan<byte> * superBlocks: int * xq: QuantizedActivations -> float32
    abstract DotQ6_K: row: ReadOnlySpan<byte> * superBlocks: int * xq: QuantizedActivations -> float32

module CpuFeatures =

    let isSupported (isa: CpuIsa) =
        match isa with
        | Portable -> true
        | X86Avx2 -> Avx2.IsSupported && Fma.IsSupported
        | X86Avx512Vnni -> Avx512F.IsSupported && Avx512BW.IsSupported && AvxVnni.V512.IsSupported && Fma.IsSupported
        | ArmNeon -> AdvSimd.Arm64.IsSupported

    /// Best implementation for the running CPU.
    let detect () =
        if isSupported X86Avx512Vnni then X86Avx512Vnni
        elif isSupported X86Avx2 then X86Avx2
        elif isSupported ArmNeon then ArmNeon
        else Portable

    /// Human-readable feature summary (for logs / plan output).
    let describe () =
        let flags =
            [ if Avx2.IsSupported then "AVX2"
              if Fma.IsSupported then "FMA"
              if Avx512F.IsSupported then "AVX-512F"
              if Avx512BW.IsSupported then "AVX-512BW"
              if AvxVnni.V512.IsSupported then "AVX-512-VNNI"
              elif AvxVnni.IsSupported then "AVX-VNNI"
              if AvxVnniInt8.IsSupported then "AVX-VNNI-INT8"
              if Avx10v1.IsSupported then "AVX10.1"
              if Avx10v2.IsSupported then "AVX10.2"
              if AdvSimd.Arm64.IsSupported then "NEON"
              if Dp.IsSupported then "DotProd" ]
        if flags.IsEmpty then $"%d{Vector<byte>.Count * 8}-bit Vector<T>" else String.Join("+", flags)
