#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Download files from the NCI Genomic Data Commons (GDC) using gdc-client.

Prereqs:
  - Install gdc-client (https://gdc.cancer.gov/access-data/gdc-data-transfer-tool)
  - Obtain a GDC token file for controlled-access data (if needed)
  - Generate a GDC manifest file (from the GDC portal or API)

Usage:
  targets/tcga/gdc_download.sh --manifest MANIFEST.tsv --out OUTDIR [--token TOKEN.txt] [--gdc-client /path/to/gdc-client]

Examples:
  targets/tcga/gdc_download.sh --manifest tcga_svs_manifest.tsv --token gdc_token.txt --out data/gdc_download
  targets/tcga/gdc_download.sh --manifest tcga_maf_manifest.tsv --out data/gdc_download
EOF
}

MANIFEST=""
OUTDIR=""
TOKEN=""
GDC_CLIENT="gdc-client"
RETRY_AMOUNT="100"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) MANIFEST="$2"; shift 2 ;;
    --out) OUTDIR="$2"; shift 2 ;;
    --token) TOKEN="$2"; shift 2 ;;
    --gdc-client) GDC_CLIENT="$2"; shift 2 ;;
    --retry-amount) RETRY_AMOUNT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[gdc_download] Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${MANIFEST}" || -z "${OUTDIR}" ]]; then
  echo "[gdc_download] --manifest and --out are required." >&2
  usage
  exit 2
fi
if [[ ! -f "${MANIFEST}" ]]; then
  echo "[gdc_download] Manifest not found: ${MANIFEST}" >&2
  exit 2
fi
mkdir -p "${OUTDIR}"

# If --token isn't provided, allow token path via:
#   - $GDC_TOKEN_FILE, or
#   - configs/secrets.env (copy from configs/secrets.env.example)
if [[ -z "${TOKEN}" ]]; then
  token_candidate="${GDC_TOKEN_FILE:-}"
  if [[ -z "${token_candidate}" ]]; then
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    repo_root="$(cd "${script_dir}/../.." && pwd)"
    secrets_env="${repo_root}/configs/secrets.env"
    if [[ -f "${secrets_env}" ]]; then
      # shellcheck disable=SC1090
      source "${secrets_env}"
      token_candidate="${GDC_TOKEN_FILE:-}"
    fi
  fi
  if [[ -n "${token_candidate}" ]]; then
    if [[ -f "${token_candidate}" ]]; then
      TOKEN="${token_candidate}"
    else
      echo "[gdc_download] Warning: GDC_TOKEN_FILE not found: ${token_candidate} (continuing without token)" >&2
    fi
  fi
fi

cmd=( "${GDC_CLIENT}" download -m "${MANIFEST}" -d "${OUTDIR}" --retry-amount "${RETRY_AMOUNT}" )
if [[ -n "${TOKEN}" ]]; then
  if [[ ! -f "${TOKEN}" ]]; then
    echo "[gdc_download] Token not found: ${TOKEN}" >&2
    exit 2
  fi
  cmd+=( -t "${TOKEN}" )
fi

echo "[gdc_download] ${cmd[*]}"
exec "${cmd[@]}"
