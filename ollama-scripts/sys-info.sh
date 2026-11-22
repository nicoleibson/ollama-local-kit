#!/usr/bin/env bash
echo "=== SISTEMA ==="
hostnamectl 2>/dev/null | sed -n '1,12p' || cat /etc/os-release
echo
echo "=== CPU ==="
lscpu 2>/dev/null | grep -E 'Model name|CPU\(s\)|Thread|Core|Socket|MHz' | head -8
echo
echo "=== MEMORIA ==="
free -h
echo
echo "=== GPU ==="
lspci -nn | grep -i vga
echo
echo "=== ARRANQUE ==="
uptime
