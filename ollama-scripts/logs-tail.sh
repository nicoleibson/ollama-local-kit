#!/usr/bin/env bash
echo "=== ÚLTIMOS 50 MENSAJES DEL SISTEMA ==="
journalctl -n 50 --no-pager
