"""Selective HTTP-range ZIP downloads with CRC and extraction-path checks."""

from __future__ import annotations

import io
import os
from pathlib import Path
import struct
import time
import urllib.request
import zipfile
import zlib


class RemoteZip(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0
        self.cache = []

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.pos = (
            offset if whence == 0 else (self.pos if whence == 1 else self.size) + offset
        )
        return self.pos

    def tell(self):
        return self.pos

    def fetch(self, start, end):
        end = min(end, self.size)
        for a, b in self.cache:
            if a <= start and end <= a + len(b):
                return b[start - a : end - a]
        for attempt in range(4):
            try:
                # Distinct query keys prevent proxies serving a cached wrong range.
                url = self.url + f"?download=true&range={start}-{end - 1}"
                req = urllib.request.Request(
                    url, headers={"Range": f"bytes={start}-{end - 1}"}
                )
                with urllib.request.urlopen(req, timeout=90) as response:
                    expected = f"bytes {start}-{end - 1}/{self.size}"
                    if (
                        response.status != 206
                        or response.headers.get("Content-Range") != expected
                    ):
                        raise IOError(
                            f"Range unsupported: {response.status}, {response.headers.get('Content-Range')}"
                        )
                    data = response.read()
                if len(data) != end - start:
                    raise IOError("Incomplete HTTP range")
                return data
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(1 + attempt)

    def read(self, n=-1):
        end = self.size if n < 0 else min(self.size, self.pos + n)
        result = self.fetch(self.pos, end)
        self.cache.append((self.pos, result))
        self.pos = end
        return result

    def extract_member(self, info, destination):
        target = (destination / info.filename).resolve()
        if not target.is_relative_to(destination.resolve()):
            raise ValueError(f"Unsafe ZIP member {info.filename}")
        if target.exists() and target.stat().st_size == info.file_size:
            if zlib.crc32(target.read_bytes()) == info.CRC:
                return info.filename, "cached"
        block = self.fetch(
            info.header_offset, info.header_offset + 30 + 4096 + info.compress_size
        )
        header = struct.unpack("<4s5H3I2H", block[:30])
        if header[0] != b"PK\x03\x04":
            raise IOError("Invalid local ZIP header")
        begin = 30 + header[-2] + header[-1]
        if begin + info.compress_size > len(block):
            block = self.fetch(
                info.header_offset, info.header_offset + begin + info.compress_size
            )
        packed = block[begin : begin + info.compress_size]
        if info.compress_type == zipfile.ZIP_DEFLATED:
            data = zlib.decompress(packed, -15)
        elif info.compress_type == zipfile.ZIP_STORED:
            data = packed
        else:
            raise ValueError(f"Unsupported compression: {info.compress_type}")
        if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
            raise IOError(f"CRC mismatch: {info.filename}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + f".partial.{os.getpid()}")
        temporary.write_bytes(data)
        temporary.replace(target)
        return info.filename, len(data)
