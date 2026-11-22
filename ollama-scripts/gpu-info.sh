#!/usr/bin/env bash
echo "=== GPU PCIe ==="
lspci -nn | grep -i vga
echo
echo "=== VULKAN ==="
vulkaninfo --summary 2>/dev/null | grep -E 'GPU[0-9]|deviceName|deviceType' | head -12
echo
echo "=== SENSORES ==="
sensors 2>/dev/null | grep -E 'edge:|Composite:|fan1:|PPT:'
