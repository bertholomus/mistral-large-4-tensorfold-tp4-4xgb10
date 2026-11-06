#!/usr/bin/env bash
# Start the four ranks from any machine that reaches the nodes by ssh, and wait for rank 0's /health:
#   start_lane.sh [tag] [tensorfold serve args, such as --host 0.0.0.0]
#   ML4_HOSTS          the four nodes in rank order (default: spark1 spark2 spark3 spark4)
#   ML4_MASTER_PORT    rank 0's rendezvous port (default: 29561)
#   ML4_TREE ML4_PACK ML4_MASTER ML4_IFNAME ML4_HCA as in tools/ml4/cp4_serve.sh of the engine; the same on every node
set -eu
: "${ML4_TREE:?set ML4_TREE}" "${ML4_PACK:?set ML4_PACK}" "${ML4_MASTER:?set ML4_MASTER}"
: "${ML4_IFNAME:?set ML4_IFNAME}" "${ML4_HCA:?set ML4_HCA}"
TAG="${1:-ml4}"
[ $# -gt 0 ] && shift
ARGS="$*"
PORT="${ML4_MASTER_PORT:-29561}"
read -r -a HOSTS <<< "${ML4_HOSTS:-spark1 spark2 spark3 spark4}"
[ "${#HOSTS[@]}" -eq 4 ] || { echo "ML4_HOSTS needs four nodes" >&2; exit 2; }
VARS="ML4_TREE=$ML4_TREE ML4_PACK=$ML4_PACK ML4_MASTER=$ML4_MASTER ML4_IFNAME=$ML4_IFNAME ML4_HCA=$ML4_HCA EXTRA='${EXTRA:-}'"
for r in 0 1 2 3; do
  ssh "${HOSTS[$r]}" "mkdir -p $ML4_TREE/logs $ML4_TREE/rank-cache; ($VARS nohup bash $ML4_TREE/src/tensorfold/tools/ml4/cp4_serve.sh $r $TAG $PORT $ARGS \
    > /dev/null 2>&1 < /dev/null &)"
done
# A first start compiles nothing (prebuild.sh did) but writes each rank's cache from the pack; later starts read the cache.
for i in $(seq 1 1800); do
  if ssh "${HOSTS[0]}" "curl -s -m 2 127.0.0.1:8561/health" 2>/dev/null | grep -q '"ok": *true'; then
    echo "up after $SECONDS s"; exit 0
  fi
  sleep 2
done
echo "not up after an hour: see $ML4_TREE/logs/$TAG-rank*.log on each node" >&2
exit 1
