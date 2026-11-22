#!/usr/bin/env bash
echo "=== TOP 10 PROCESOS POR CPU ==="
ps -eo pid,user,comm,%cpu,%mem --sort=-%cpu | head -11
echo
echo "=== CARGA ==="
uptime
