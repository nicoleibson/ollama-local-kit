#!/usr/bin/env bash
echo "=== GOVERNOR ==="
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort | uniq -c
echo
echo "=== FRECUENCIAS (cpu0) ==="
grep -H . /sys/devices/system/cpu/cpu0/cpufreq/scaling_{cur,min,max}_freq 2>/dev/null
echo
echo "=== CPU ==="
lscpu 2>/dev/null | grep -E 'Model name|CPU\(s\)|Core|Thread|Socket'
echo
echo "=== CARGA ==="
uptime
