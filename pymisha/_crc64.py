"""Shared CRC64-ECMA helpers (parity with C++ CRC64.h)."""

_CRC64_POLY = 0xC96C5795D7870F42
_CRC64_TABLE = None


def _crc64_table():
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ _CRC64_POLY
            else:
                crc >>= 1
        table.append(crc & 0xFFFFFFFFFFFFFFFF)
    return table


def crc64_incremental(crc, data):
    global _CRC64_TABLE
    if _CRC64_TABLE is None:
        _CRC64_TABLE = _crc64_table()
    for byte in data:
        crc = (crc >> 8) ^ _CRC64_TABLE[(crc ^ byte) & 0xFF]
    return crc & 0xFFFFFFFFFFFFFFFF


def crc64_init():
    return 0xFFFFFFFFFFFFFFFF


def crc64_finalize(crc):
    return (~crc) & 0xFFFFFFFFFFFFFFFF
