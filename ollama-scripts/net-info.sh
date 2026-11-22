#!/usr/bin/env bash
echo "=== INTERFACES ==="
ip -br a
echo
echo "=== RUTAS ==="
ip route
echo
echo "=== PUERTOS ABIERTOS ==="
ss -tulnp | head -30
