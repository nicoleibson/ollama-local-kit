#!/usr/bin/env python3
"""Genera una copia del GGUF de Qwen3.5-4B con qwen35.pooling_type para Ollama."""
import struct
import sys
import shutil
from pathlib import Path

SRC = Path('/home/onion/Documentos/ModelsIA/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf')
FMTS = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<b', 10: '<Q', 11: '<q', 12: '<d'}


def build(pooling, dst):
    with SRC.open('rb') as f:
        assert f.read(4) == b'GGUF'
        version, tensors, kv = struct.unpack('<IQQ', f.read(20))

        def rs():
            n = struct.unpack('<Q', f.read(8))[0]
            return f.read(n)

        def skip(t):
            if t == 7:
                f.read(1)
            elif t == 8:
                rs()
            elif t == 9:
                et = struct.unpack('<I', f.read(4))[0]
                n = struct.unpack('<Q', f.read(8))[0]
                for _ in range(n):
                    skip(et)
            else:
                f.read(struct.calcsize(FMTS[t]))

        for _ in range(kv):
            rs()
            skip(struct.unpack('<I', f.read(4))[0])
        kv_end = f.tell()

    key1 = b'qwen35.pooling_type'
    extra1 = struct.pack('<Q', len(key1)) + key1 + struct.pack('<I', 4) + struct.pack('<I', pooling)
    key2 = b'qwen35.pad_marker1'
    val2 = (b'ollama-embed-pad' + b'x' * 7)[:23]
    extra2 = struct.pack('<Q', len(key2)) + key2 + struct.pack('<I', 8) + struct.pack('<Q', len(val2)) + val2
    extra = extra1 + extra2
    assert len(extra) % 32 == 0

    with SRC.open('rb') as f, dst.open('wb') as o:
        header = bytearray(f.read(24))
        header[16:24] = struct.pack('<Q', kv + 2)
        o.write(header)
        f.seek(24)
        remaining = kv_end - 24
        while remaining:
            chunk = f.read(min(8 * 1024 * 1024, remaining))
            o.write(chunk)
            remaining -= len(chunk)
        o.write(extra)
        shutil.copyfileobj(f, o, length=8 * 1024 * 1024)
    print('pooling', pooling, '->', dst, dst.stat().st_size)


if __name__ == '__main__':
    build(int(sys.argv[1]), Path(sys.argv[2]))
