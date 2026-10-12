#!/usr/bin/env bash
# Build the Zenodo archive for the EML paper from the explicit file list in MANIFEST.txt.
#
#   bash docs/revision/zenodo/build_archive.sh            # -> docs/revision/zenodo/dist/tmcmc_eml_archive_v<VERSION>.zip
#   VERSION=1.0.1 bash docs/revision/zenodo/build_archive.sh
#   OUT_DIR=/some/dir bash docs/revision/zenodo/build_archive.sh
#
# - Only what MANIFEST.txt lists goes into the zip (never the whole repository).
# - A required manifest entry that matches nothing aborts the build.
# - Entries prefixed with "?" are optional (DS/DH stage (iv)): missing -> warning only.
# - __pycache__/, *.pyc, *.pkl, .env* are always excluded.
# - Every included run directory must contain the five run files, otherwise the build aborts.
# - SHA256SUMS (over every file in the archive) is written into the zip root.
set -euo pipefail
shopt -s nullglob globstar dotglob

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
VERSION="${VERSION:-1.0.0}"
NAME="tmcmc_eml_archive_v${VERSION}"
OUT_DIR="${OUT_DIR:-$HERE/dist}"
MANIFEST="$HERE/MANIFEST.txt"
RUN_FILES=(samples.npy logL.npy theta_MAP.json run_record.json config.json)

die() { echo "ERROR: $*" >&2; exit 1; }
warn() { echo "WARNING: $*" >&2; }

[ -f "$MANIFEST" ] || die "manifest not found: $MANIFEST"
command -v zip >/dev/null || die "'zip' not found"
command -v sha256sum >/dev/null || die "'sha256sum' not found"

excluded() {  # $1 = repo-relative path
  case "/$1" in
    */__pycache__/*|*.pyc|*.pkl|*/.env|*/.env.*|*/.DS_Store) return 0 ;;
  esac
  return 1
}

STAGE_ROOT="$(mktemp -d)"
trap 'rm -rf "$STAGE_ROOT"' EXIT
STAGE="$STAGE_ROOT/$NAME"
mkdir -p "$STAGE"

declare -A SEEN=()
FILES=()
missing_required=0
n_optional_missing=0
RUN_DIRS=()

cd "$REPO"
while IFS= read -r raw || [ -n "$raw" ]; do
  line="${raw%%#*}"                       # strip comments
  line="$(echo "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  [ -z "$line" ] && continue
  optional=0
  if [ "${line:0:1}" = "?" ]; then optional=1; line="${line:1}"; fi
  case "$line" in /*|*..*) die "manifest entries must be repo-relative without '..': $line" ;; esac

  # shellcheck disable=SC2206
  expanded=( $line )
  matches=()
  for m in "${expanded[@]}"; do [ -e "$m" ] && matches+=("$m"); done  # a non-glob path expands to itself
  if [ ${#matches[@]} -eq 0 ]; then
    if [ $optional -eq 1 ]; then
      warn "optional entry not present (OK for now): $line"
      n_optional_missing=$((n_optional_missing + 1))
    else
      echo "MISSING (required): $line" >&2
      missing_required=$((missing_required + 1))
    fi
    continue
  fi
  for m in "${matches[@]}"; do
    if [ -d "$m" ]; then
      case "$m" in docs/revision/generated/final_runs/*) RUN_DIRS+=("$m") ;; esac
      while IFS= read -r -d '' f; do
        f="${f#./}"
        excluded "$f" && continue
        [ -n "${SEEN[$f]:-}" ] && continue
        SEEN[$f]=1; FILES+=("$f")
      done < <(find "$m" -type f -print0 | sort -z)
    elif [ -f "$m" ]; then
      excluded "$m" && { warn "excluded by rule: $m"; continue; }
      [ -n "${SEEN[$m]:-}" ] && continue
      SEEN[$m]=1; FILES+=("$m")
    else
      die "not a regular file or directory: $m"
    fi
  done
done < "$MANIFEST"

[ $missing_required -eq 0 ] || die "$missing_required required manifest entr(y/ies) missing -- see above"
[ ${#FILES[@]} -gt 0 ] || die "manifest selected no files"

# Every archived run must be complete
for d in "${RUN_DIRS[@]}"; do
  for f in "${RUN_FILES[@]}"; do
    [ -f "$d/$f" ] || die "incomplete run directory: $d (no $f)"
  done
done

# Stage the files with their repository layout
for f in "${FILES[@]}"; do
  mkdir -p "$STAGE/$(dirname "$f")"
  cp -p "$f" "$STAGE/$f"
done
# Top-level copies so that the record page / unpacked root is self-explanatory
cp -p "$HERE/README.md" "$STAGE/README.md"
cp -p "$HERE/CITATION.cff" "$STAGE/CITATION.cff"
cp -p "$HERE/.zenodo.json" "$STAGE/.zenodo.json"
cp -p "$HERE/MANIFEST.txt" "$STAGE/MANIFEST.txt"

# Final safety net: nothing secret / excluded slipped in
bad="$(cd "$STAGE" && find . \( -name '__pycache__' -o -name '*.pyc' -o -name '*.pkl' -o -name '.env' -o -name '.env.*' \) -print)"
[ -z "$bad" ] || die "excluded files in staging area: $bad"

# Checksums over every file (paths relative to the archive root)
( cd "$STAGE" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum | sed 's#  \./#  #' > SHA256SUMS )

mkdir -p "$OUT_DIR"
ZIP="$OUT_DIR/$NAME.zip"
rm -f "$ZIP"
( cd "$STAGE_ROOT" && zip -q -r -X "$ZIP" "$NAME" )

n_files="$(unzip -Z1 "$ZIP" | grep -v '/$' | wc -l | tr -d ' ')"
uncompressed="$(du -sh "$STAGE" | cut -f1)"
zipsize="$(du -h "$ZIP" | cut -f1)"
n_runs="${#RUN_DIRS[@]}"

echo "------------------------------------------------------------"
echo "archive      : $ZIP"
echo "files        : $n_files (incl. SHA256SUMS and root copies of README/CITATION/.zenodo.json/MANIFEST)"
echo "run dirs     : $n_runs"
echo "uncompressed : $uncompressed"
echo "zip size     : $zipsize"
echo "per component (uncompressed):"
for c in data_5species tmcmc tools docs/generate_fig2_phi_version.py docs/revision/generated/final_runs \
         docs/revision/generated/final_theta_MAP docs/revision/EML_submission; do
  [ -e "$STAGE/$c" ] && printf '  %-48s %s\n' "$c" "$(du -sh "$STAGE/$c" | cut -f1)"
done
if [ $n_optional_missing -gt 0 ]; then
  echo "NOTE: $n_optional_missing optional entr(y/ies) missing -> archive is PROVISIONAL (DS/DH stage (iv) not yet included)"
fi
( cd "$OUT_DIR" && sha256sum "$NAME.zip" > "$NAME.zip.sha256" )
echo "zip sha256   : $(cut -d' ' -f1 "$OUT_DIR/$NAME.zip.sha256")"
