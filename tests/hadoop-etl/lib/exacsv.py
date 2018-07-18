#!/usr/opt/bs-python-2.7/bin/python
# encoding: utf8

import sys
import os

sys.path.append(os.path.realpath(__file__ + '/../../lib'))

column_delim = ','
reserved_chars = ['\n', '\r', column_delim]
line_term = '\n'
read_size = 4096
buf = ''

def get_csv_line(resp):
    global buf
    line = ''
    while True:
        idx = buf.find(line_term)
        if idx == -1:
            data = resp.read(read_size)
            if not data:
                break
            buf += data
        else:
            line = buf[: idx]
            if idx + 1 < len(buf):
                buf = buf[idx + 1 :]
            else:
                buf = ''
            break
    if not line:
        line = buf
        buf = ''
    if not line:
        # Empty strings -> None
        line = None
    return line


def get_csv_columns(resp, line, req_columns):
    cols = [None] * len(req_columns)
    vals = line.split(column_delim)
    for i, val in enumerate(vals):
        if i in req_columns and val:
            cols[req_columns.index(i)] = val
    return cols
