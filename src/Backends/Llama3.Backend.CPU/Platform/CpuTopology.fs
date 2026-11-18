namespace Llama3.Backend.Cpu

open System
open System.IO
open System.Runtime.InteropServices
open System.Text.RegularExpressions

/// Physical CPU topology. SIMD-bound kernels gain nothing from SMT siblings (two hyper-threads of a
/// core share its vector units) and suffer when a sibling is preempted mid-job, so the default
/// worker count is the number of physical cores the process may actually use.
module CpuTopology =

    module private Native =
        [<DllImport("kernel32.dll", SetLastError = true)>]
        extern bool GetLogicalProcessorInformationEx(int relationship, byte[] buffer, uint32& returnedLength)

        [<DllImport("libc", EntryPoint = "sysctlbyname", SetLastError = true)>]
        extern int sysctlbyname(string name, int& value, unativeint& size, nativeint newValue, unativeint newSize)

    /// Parses a Linux CPU list such as "0-7,16-23".
    let parseCpuList (text: string) : Set<int> =
        text.Trim().Split(',', StringSplitOptions.RemoveEmptyEntries)
        |> Array.collect (fun part ->
            match part.Split('-') with
            | [| single |] -> [| int single |]
            | [| first; last |] -> [| int first .. int last |]
            | _ -> raise (FormatException $"Invalid CPU list '%s{text}'"))
        |> Set.ofArray

    /// Linux: distinct cores (core_cpus_list) among the CPUs allowed to this process (affinity,
    /// cpusets of containers), so SMT siblings inside the allowed set are not double counted.
    let private linuxPhysicalCores () =
        let root = "/sys/devices/system/cpu"
        if not (Directory.Exists root) then None
        else
            let allowed =
                File.ReadLines "/proc/self/status"
                |> Seq.tryFind (fun l -> l.StartsWith("Cpus_allowed_list:", StringComparison.Ordinal))
                |> Option.map (fun l -> parseCpuList (l.Substring(l.IndexOf ':' + 1)))
            let cores =
                Directory.EnumerateDirectories(root, "cpu*")
                |> Seq.choose (fun d ->
                    let m = Regex.Match(Path.GetFileName d, "^cpu([0-9]+)$")
                    if not m.Success then None
                    else
                        let cpu = int m.Groups.[1].Value
                        let isAllowed = allowed |> Option.forall (Set.contains cpu)
                        let topology = Path.Combine(d, "topology")
                        let file =
                            [ "core_cpus_list"; "thread_siblings_list" ]
                            |> List.map (fun f -> Path.Combine(topology, f))
                            |> List.tryFind File.Exists
                        match isAllowed, file with
                        | true, Some f -> Some((File.ReadAllText f).Trim())
                        | _ -> None)
                |> Set.ofSeq
            if cores.IsEmpty then None else Some cores.Count

    /// Windows: one RelationProcessorCore record per physical core.
    let private windowsPhysicalCores () =
        let mutable length = 0u
        Native.GetLogicalProcessorInformationEx(0, null, &length) |> ignore
        if length = 0u then None
        else
            let buffer = Array.zeroCreate<byte> (int length)
            if not (Native.GetLogicalProcessorInformationEx(0, buffer, &length)) then None
            else
                let mutable offset = 0
                let mutable cores = 0
                while offset + 8 <= int length do
                    let relationship = BitConverter.ToInt32(buffer, offset)
                    let size = BitConverter.ToInt32(buffer, offset + 4)
                    if relationship = 0 then cores <- cores + 1
                    offset <- offset + max size 8
                if cores > 0 then Some cores else None

    /// macOS: hw.physicalcpu.
    let private macPhysicalCores () =
        let mutable value = 0
        let mutable size = unativeint sizeof<int>
        if Native.sysctlbyname ("hw.physicalcpu", &value, &size, 0n, 0un) = 0 && value > 0 then Some value else None

    /// Physical cores usable by this process (never more than the logical processors it may use,
    /// which already reflects affinity and container CPU quotas).
    let physicalCores =
        lazy
            (let logical = Environment.ProcessorCount
             let physical =
                 try
                     if OperatingSystem.IsLinux() then linuxPhysicalCores ()
                     elif OperatingSystem.IsWindows() then windowsPhysicalCores ()
                     elif OperatingSystem.IsMacOS() then macPhysicalCores ()
                     else None
                 with
                 | :? IOException | :? UnauthorizedAccessException | :? FormatException
                 | :? DllNotFoundException | :? EntryPointNotFoundException -> None
             max 1 (min logical (defaultArg physical logical)))
