// pm_crc64: Expose C++ CRC64-ECMA checksum to Python.
// Replaces the pure-Python byte-by-byte loop in _crc64.py.

#include "pymisha.h"
#include "CRC64.h"

static misha::CRC64 s_crc64;

/*
 * pm_crc64_update(crc, data)
 *
 * crc:   int — current CRC state (start with 0xFFFFFFFFFFFFFFFF)
 * data:  bytes or bytearray
 *
 * Returns: int — updated CRC state
 */
PyObject *pm_crc64_update(PyObject *self, PyObject *args)
{
    unsigned long long crc_in;
    Py_buffer buf;

    if (!PyArg_ParseTuple(args, "Ky*", &crc_in, &buf)) {
        return nullptr;
    }

    uint64_t crc = s_crc64.compute_incremental(
        (uint64_t)crc_in,
        (const unsigned char *)buf.buf,
        (size_t)buf.len
    );

    PyBuffer_Release(&buf);
    return PyLong_FromUnsignedLongLong(crc);
}

/*
 * pm_crc64_finalize(crc)
 *
 * crc:  int — current CRC state
 *
 * Returns: int — finalized checksum (~crc)
 */
PyObject *pm_crc64_finalize(PyObject *self, PyObject *args)
{
    unsigned long long crc_in;

    if (!PyArg_ParseTuple(args, "K", &crc_in)) {
        return nullptr;
    }

    uint64_t result = s_crc64.finalize_incremental((uint64_t)crc_in);
    return PyLong_FromUnsignedLongLong(result);
}
