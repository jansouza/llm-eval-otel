#!/usr/bin/env bash
# Build the llm-eval-otel image and push it to a local Nexus Docker registry.
# Local development only; releases go through .github/workflows/release.yml.
#
#   scripts/push-nexus.sh            # tags <version>-<sha> and dev
#   scripts/push-nexus.sh my-tag     # tags my-tag and dev
#
# Environment:
#   NEXUS_REGISTRY   registry host:port (Nexus docker connector)   default: localhost:8082
#   NEXUS_NAMESPACE  optional path prefix, e.g. "team"             default: empty
#   NEXUS_USER       login user; skip login when unset
#   NEXUS_PASSWORD   login password; prompted when NEXUS_USER is set and this is not
#   IMAGE_NAME       image name                                     default: llm-eval-otel
#   PLATFORM         target platform, e.g. linux/arm64              default: host platform
#   CONTAINER_CLI    docker or podman                               default: docker
#   TLS_VERIFY       podman only: "false" for an HTTP Nexus         default: true
#
# These variables are also read from .env at the repository root, when it exists.
# A variable already set in the environment takes precedence over the file.
#
# Docker already treats localhost registries as insecure. For any other HTTP host,
# add it to "insecure-registries" in /etc/docker/daemon.json.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
    declare -A preset=()
    for var in NEXUS_REGISTRY NEXUS_NAMESPACE NEXUS_USER NEXUS_PASSWORD \
        IMAGE_NAME PLATFORM CONTAINER_CLI TLS_VERIFY; do
        if [[ -n "${!var+set}" ]]; then
            preset[$var]="${!var}"
        fi
    done
    echo "==> Reading .env"
    set -a
    # shellcheck source=/dev/null
    source .env
    set +a
    for var in "${!preset[@]}"; do
        export "$var=${preset[$var]}"
    done
fi

NEXUS_REGISTRY="${NEXUS_REGISTRY:-localhost:8082}"
NEXUS_NAMESPACE="${NEXUS_NAMESPACE:-}"
IMAGE_NAME="${IMAGE_NAME:-llm-eval-otel}"
CONTAINER_CLI="${CONTAINER_CLI:-docker}"
TLS_VERIFY="${TLS_VERIFY:-true}"

version="$(sed -n 's/^__version__ = "\(.*\)"/\1/p' src/llm_eval_otel/version.py)"
sha="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
    sha="${sha}-dirty"
fi
tag="${1:-${version}-${sha}}"

repo="${NEXUS_REGISTRY}/${NEXUS_NAMESPACE:+${NEXUS_NAMESPACE}/}${IMAGE_NAME}"

tls_args=()
if [[ "$CONTAINER_CLI" == "podman" ]]; then
    tls_args=(--tls-verify="$TLS_VERIFY")
fi

if [[ -n "${NEXUS_USER:-}" ]]; then
    if [[ -z "${NEXUS_PASSWORD:-}" ]]; then
        read -rsp "Nexus password for ${NEXUS_USER}: " NEXUS_PASSWORD
        echo
    fi
    echo "==> Logging in to ${NEXUS_REGISTRY}"
    printf '%s' "$NEXUS_PASSWORD" |
        "$CONTAINER_CLI" login "${tls_args[@]}" -u "$NEXUS_USER" --password-stdin "$NEXUS_REGISTRY"
fi

build_args=(-t "${repo}:${tag}" -t "${repo}:dev")
if [[ -n "${PLATFORM:-}" ]]; then
    build_args+=(--platform "$PLATFORM")
fi

echo "==> Building ${repo}:${tag}"
"$CONTAINER_CLI" build "${build_args[@]}" .

for t in "$tag" dev; do
    echo "==> Pushing ${repo}:${t}"
    "$CONTAINER_CLI" push "${tls_args[@]}" "${repo}:${t}"
done

echo "==> Done: ${repo}:${tag}"
echo "    Run the demo with it: LLM_EVAL_IMAGE=${repo}:${tag} docker compose -f deploy/docker-compose.yaml up"
