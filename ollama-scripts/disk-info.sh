#!/usr/bin/env bash
echo "=== ESPACIO ==="
df -h
echo
echo "=== INODOS ==="
df -i
echo
echo "=== DISPOSITIVOS ==="
lsblk -o NAME,SIZE,TYPE,MOUNTPOINT
echo
echo "=== CARPETAS MÁS GRANDES EN /home/onion ==="
du -h --max-depth=1 /home/onion 2>/dev/null | sort -h | tail -10
