#!/usr/bin/env bash
set -euo pipefail

IMAGE="${COMETS_IMAGE:-dukovski/comets-lab:1.0}"
PORT="${COMETS_PORT:-8888}"
HOST_DIR="${COMETS_HOST_DIR:-/home/nishioka}"
CONTAINER_DIR="${COMETS_CONTAINER_DIR:-/workspace}"
HOME_IN_CONTAINER="${COMETS_HOME_IN_CONTAINER:-/workspace}"

run_in_docker_group() {
  local q=()
  local a
  for a in "$@"; do
    q+=("$(printf '%q' "$a")")
  done
  sg docker -c "${q[*]}"
}

cmd="${1:-jupyter}"
shift || true

case "$cmd" in
  jupyter)
    run_in_docker_group docker run --rm -it \
      -p "${PORT}:8888" \
      -v "${HOST_DIR}:${CONTAINER_DIR}" \
      --workdir "${CONTAINER_DIR}" \
      --user "$(id -u):$(id -g)" \
      -e HOME="${HOME_IN_CONTAINER}" \
      -e XDG_CACHE_HOME="${HOME_IN_CONTAINER}/.cache" \
      -e XDG_CONFIG_HOME="${HOME_IN_CONTAINER}/.config" \
      -e XDG_DATA_HOME="${HOME_IN_CONTAINER}/.local/share" \
      -e JUPYTER_CONFIG_DIR="${HOME_IN_CONTAINER}/.jupyter" \
      -e JUPYTER_DATA_DIR="${HOME_IN_CONTAINER}/.local/share/jupyter" \
      -e JUPYTER_RUNTIME_DIR="${HOME_IN_CONTAINER}/.local/share/jupyter/runtime" \
      "${IMAGE}"
    ;;
  java)
    run_in_docker_group docker run --rm "${IMAGE}" java -version
    ;;
  shell)
    run_in_docker_group docker run --rm -it \
      -v "${HOST_DIR}:${CONTAINER_DIR}" \
      --workdir "${CONTAINER_DIR}" \
      --user "$(id -u):$(id -g)" \
      -e HOME="${HOME_IN_CONTAINER}" \
      "${IMAGE}" bash
    ;;
  *)
    run_in_docker_group docker run --rm -it \
      -v "${HOST_DIR}:${CONTAINER_DIR}" \
      --workdir "${CONTAINER_DIR}" \
      --user "$(id -u):$(id -g)" \
      -e HOME="${HOME_IN_CONTAINER}" \
      "${IMAGE}" "$cmd" "$@"
    ;;
esac

