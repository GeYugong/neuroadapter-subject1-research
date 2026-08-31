#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: launch_tmux_task.sh SESSION STATUS_FILE LOG_FILE COMMAND [ARG ...]" >&2
  exit 2
fi

session=$1
status_file=$2
log_file=$3
shift 3

if tmux has-session -t "$session" 2>/dev/null; then
  echo "tmux session already exists: $session" >&2
  exit 1
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
printf -v shell_command '%q ' \
  "$script_dir/run_with_status.sh" "$status_file" "$log_file" "$@"
tmux new-session -d -s "$session" "$shell_command"

sleep 1
if ! tmux has-session -t "$session" 2>/dev/null && [[ ! -f "$status_file" ]]; then
  echo "tmux task exited before producing a status file: $session" >&2
  exit 1
fi

echo "session=$session"
echo "status_file=$status_file"
echo "log_file=$log_file"

