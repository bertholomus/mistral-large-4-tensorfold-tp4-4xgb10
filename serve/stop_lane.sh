#!/usr/bin/env bash
# Stop the four ranks: stop_lane.sh [tag]  (ML4_HOSTS as in start_lane.sh)
set -u
TAG="${1:-ml4}"
read -r -a HOSTS <<< "${ML4_HOSTS:-spark1 spark2 spark3 spark4}"
for h in "${HOSTS[@]}"; do ssh "$h" "docker stop -t 5 ml4cp4-$TAG > /dev/null 2>&1 || true"; done
