#!/usr/bin/env bash
echo "=== ARCHIVOS MAYORES DE 100 MB EN /home/onion ==="
find /home/onion -xdev -type f -size +100M -printf '%s %p\n' 2>/dev/null \
  | sort -rn | head -20 \
  | awk '{printf "%.2f GB  %s\n", $1/1073741824, $2}'
