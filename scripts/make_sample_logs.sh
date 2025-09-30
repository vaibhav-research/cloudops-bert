#!/usr/bin/env bash
set -euo pipefail
mkdir -p samples
cat > samples/sample_logs.txt <<'TXT'
INFO dfs.FSNamesystem: Starting Namenode
WARN dfs.DataStreamer: Slow block receiver
ERROR dfs.DataNode: Lost connection to namenode
INFO dfs.BlockManager: Block report processed
TXT
echo "✅ Wrote samples/sample_logs.txt"
