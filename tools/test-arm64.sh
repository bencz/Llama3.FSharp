#!/usr/bin/env bash
# Runs the CPU test suite and the llama3.cu golden on linux-arm64 from an x86-64 host.
#
# The tests are published self-contained for linux-arm64 and executed in an arm64 container
# through qemu-user (binfmt_misc). The real ARM64 JIT runs, so the NEON / DotProd kernels are
# exercised exactly as on hardware (qemu's "max" CPU exposes asimddp, SVE and SVE2).
#
# Requirements: podman (or docker via ENGINE=docker) and qemu-user-static with binfmt registered
#   Fedora: sudo dnf install qemu-user-static   Debian/Ubuntu: sudo apt install qemu-user-static binfmt-support
# Optional: LLAMA3_MODELS_DIR (default ~/models) is mounted for the model-dependent tests.
set -euo pipefail

ENGINE="${ENGINE:-podman}"
IMAGE="${IMAGE:-docker.io/library/debian:stable}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="${LLAMA3_MODELS_DIR:-$HOME/models}"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

echo "Publishing for linux-arm64 into $OUT ..."
dotnet publish "$REPO/tests/CpuTests" -c Release -r linux-arm64 --self-contained -o "$OUT/tests" -v quiet -nologo
dotnet publish "$REPO/src/Hosts/Llama3.CLI" -c Release -r linux-arm64 --self-contained -o "$OUT/cli" -v quiet -nologo

models=()
[[ -d "$MODELS" ]] && models=(-v "$MODELS:/models:ro,z" -e LLAMA3_MODELS_DIR=/models)

# The repository is mounted read-only at /repo (tests locate it by walking up from their
# directory), with the published tests inside it.
"$ENGINE" run --rm --platform linux/arm64 \
    -v "$REPO:/repo:ro,z" -v "$OUT/tests:/repo/arm64-tests:z" -v "$OUT/cli:/cli:z" "${models[@]}" \
    -e DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1 -w /repo \
    "$IMAGE" sh -c '
        set -e
        echo "CPU: $(uname -m), features: $(grep -o -m1 -w "asimddp\|sve2\|sve" /proc/cpuinfo | sort -u | tr "\n" " ")"
        for isa in neon portable; do
            echo "== golden (stories15M, --cpu-isa $isa)"
            /cli/llama3 --mode cpu --cpu-isa $isa 2>&1 | grep -E "backend:|dream"
        done
        cd /repo/arm64-tests && ./Llama3.CpuTests'
