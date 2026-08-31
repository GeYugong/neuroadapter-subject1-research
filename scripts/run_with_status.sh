#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: run_with_status.sh STATUS_FILE LOG_FILE COMMAND [ARG ...]" >&2
  exit 2
fi

status_file=$1
log_file=$2
shift 2

mkdir -p "$(dirname "$status_file")" "$(dirname "$log_file")"
rm -f "$status_file" "${status_file}.tmp"

{
  echo "started_at=$(date --iso-8601=seconds)"
  printf 'command='
  printf '%q ' "$@"
  printf '\n'
} >> "$log_file"

"$@" >> "$log_file" 2>&1
return_code=$?

{
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "exit_code=$return_code"
} >> "$log_file"

printf '%s\n' "$return_code" > "${status_file}.tmp"
mv "${status_file}.tmp" "$status_file"
exit "$return_code"

