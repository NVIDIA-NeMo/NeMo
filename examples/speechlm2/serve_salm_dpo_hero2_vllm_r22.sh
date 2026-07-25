#!/usr/bin/env bash
# Serve a tracked Hero2 DPO export with the established r22 vLLM NeMo source.
#
# ``R22_NEMO_SOURCE`` is intentionally a read-only provenance input.  Setuptools
# writes egg-info while building an ordinary wheel, so it cannot be installed
# in-place.  We first make an *explicit, verified, job-local package artifact*,
# then non-editably install that artifact.  The immutable source is never
# modified, no files are overlaid into an installed package, and this script
# never changes the model directory.
set -euo pipefail

: "${R22_NEMO_SOURCE:?R22_NEMO_SOURCE must point to the readable r22 NeMo source}"
: "${R22_TRANSFORMER_ENCODER_SHA256:?R22_TRANSFORMER_ENCODER_SHA256 must pin r22 TransformerEncoder}"
: "${R22_NEMO_SERVER_FINGERPRINT_SHA256:?R22_NEMO_SERVER_FINGERPRINT_SHA256 must pin the r22 serving source}"

fingerprint_r22_server_source() {
    local root="$1"
    local rel actual
    while IFS= read -r rel; do
        test -r "$root/$rel"
        actual="$(sha256sum "$root/$rel" | awk '{print $1}')"
        printf '%s  %s\n' "$actual" "$rel"
    done <<'EOF' | sha256sum | awk '{print $1}'
setup.py
pyproject.toml
nemo/collections/asr/modules/transformer_encoder.py
nemo/collections/speechlm2/models/salm.py
nemo/collections/speechlm2/vllm/salm/__init__.py
EOF
}

stage_verified_r22_package() {
    local source="$R22_NEMO_SOURCE"
    local expected="$R22_NEMO_SERVER_FINGERPRINT_SHA256"
    local source_fingerprint stage_root staged_fingerprint
    source_fingerprint="$(fingerprint_r22_server_source "$source")"
    test "$source_fingerprint" = "$expected"
    test "$(sha256sum "$source/nemo/collections/asr/modules/transformer_encoder.py" | awk '{print $1}')" = "$R22_TRANSFORMER_ENCODER_SHA256"

    stage_root="$(mktemp -d "${TMPDIR:-/tmp}/r22-nemo-package.XXXXXX")"
    R22_STAGE_ROOT="$stage_root"
    export R22_STAGE_ROOT
    # This is an isolated package-input artifact, not an in-place workaround.
    # These are every source input used by setuptools for the normal r22 wheel;
    # deliberately omit r22's unrelated build/tests/examples trees so each
    # serving rank does not stage stale build products.
    mkdir -p "$stage_root/source"
    cp -a "$source/setup.py" "$source/pyproject.toml" "$source/README.md" "$source/MANIFEST.in" "$stage_root/source/"
    mkdir -p "$stage_root/source/nemo"
    # SSHFS metadata latency dominates a serial recursive copy.  These are
    # disjoint top-level package entries, so bounded parallel staging changes
    # no bytes or package semantics.
    find "$source/nemo" -mindepth 1 -maxdepth 1 ! -name collections -print0 | xargs -0 -r -n 1 -P 8 cp -a -t "$stage_root/source/nemo"
    mkdir -p "$stage_root/source/nemo/collections"
    find "$source/nemo/collections" -mindepth 1 -maxdepth 1 -print0 | xargs -0 -r -n 1 -P 8 cp -a -t "$stage_root/source/nemo/collections"
    cp -a "$source/requirements" "$stage_root/source/"
    chmod -R u+w "$stage_root/source"
    staged_fingerprint="$(fingerprint_r22_server_source "$stage_root/source")"
    test "$staged_fingerprint" = "$expected"
    printf 'R22_NEMO_PACKAGE_ARTIFACT source_fingerprint=%s stage_fingerprint=%s source=%s staged=%s\n' \
        "$source_fingerprint" "$staged_fingerprint" "$source" "$stage_root/source"
}

cleanup_stage() {
    if test -n "${R22_STAGE_ROOT:-}"; then
        rm -rf "$R22_STAGE_ROOT"
    fi
}
trap cleanup_stage EXIT

stage_verified_r22_package

if test "${1:-}" = "--verify-r22-package-install"; then
    target="$(mktemp -d "${TMPDIR:-/tmp}/r22-nemo-install-test.XXXXXX")"
    trap 'rm -rf "${target:-}"; cleanup_stage' EXIT
    python3 -m venv "$target/venv"
    "$target/venv/bin/python" -m pip install --no-deps --force-reinstall "$R22_STAGE_ROOT/source"
    "$target/venv/bin/python" - <<'PY'
import pathlib
import nemo
location = pathlib.Path(nemo.__file__).resolve()
target = pathlib.Path(__import__('sys').prefix).resolve()
if target not in location.parents:
    raise SystemExit(f"r22 package did not import from staged install target: {location}")
print(f"R22_NEMO_PACKAGE_INSTALL_REGRESSION passed import={location}")
PY
    exit 0
fi

python3 -m pip install --no-deps --force-reinstall "$R22_STAGE_ROOT/source"
export VLLM_PLUGINS=nemo_speechlm
export VLLM_NO_USAGE_STATS=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_jit_cache}"
exec python3 -m nemo_skills.inference.server.serve_vllm "$@"
