#!/usr/bin/env bash
echo "=== RESUMEN ==="
free -h
echo
echo "=== /proc/meminfo ==="
grep -E 'MemTotal|MemAvailable|SwapTotal|SwapFree' /proc/meminfo
echo
echo "=== TOP 5 PROCESOS POR MEMORIA ==="
ps -eo pid,comm,%mem,rss --sort=-%mem | head -6
