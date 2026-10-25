#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <watch-pid> <output.csv>" >&2
  exit 2
fi

watch_pid=$1
output=$2
ixsmi=/usr/local/corex-4.5.0/bin/ixsmi
fields='timestamp,index,gpu_name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,gpu.power.draw,board.power.draw,clocks.current.sm,clocks.current.memory'

echo "$fields" > "$output"
while kill -0 "$watch_pid" 2>/dev/null; do
  "$ixsmi" --query-gpu="$fields" --format=csv,noheader >> "$output" 2>&1
  sleep 5
done
