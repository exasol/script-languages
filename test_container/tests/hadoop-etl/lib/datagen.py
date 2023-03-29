#!/usr/bin/env python2.7
# encoding: utf8

import sys
import os
import random
import string
import re

sys.path.append(os.path.realpath(__file__ + '/../../lib'))

from exatest.utils import tempdir
import webhdfs
import exacsv

def get_tinyint():
    return random.randint(-128, 127)

def get_smallint():
    return random.randint(-32768, 32767)

def get_int():
    return random.randint(-2147483648, 2147483647)

def get_bigint():
    return random.randint(-9223372036854775808, 9223372036854775807)

def get_float():
    exp = random.randint(-38, 38)
    mant = random.uniform(0, 1)
    sign = random.choice([-1, 1])
    return sign * mant * (10 ** exp)

def get_double():
    exp = random.randint(-308, 308)
    mant = random.uniform(0, 1)
    sign = random.choice([-1, 1])
    return sign * mant * (10 ** exp)

def get_decimal(prec, scale):
    num = random.randint(0, (10 ** prec) - 1)
    if num == 0:
        return str(num)
    sign = random.choice([-1, 1])
    num_str = str(num).zfill(prec)
    num_str = num_str[: prec - scale] + "." + num_str[prec - scale :]
    num_str = num_str.rstrip('0').rstrip('.')
    num_str = num_str.lstrip('0')
    if num_str[0] == '.':
        num_str = '0' + num_str
    if sign == -1:
        num_str = "-" + num_str
    return num_str

def get_date():
    year = random.randint(1902, 2037)
    month = random.randint(1, 12)
    if month == 2:
        days = 28
    elif month in [4, 6, 9, 11]:
        days = 30
    else:
        days = 31
    day = random.randint(1, days)
    return str(year) + "-" + str(month).zfill(2) + "-" + str(day).zfill(2)

def get_timestamp():
    hour = random.randint(0, 23)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    nanosec = random.randint(0, 999999999)
    time = str(hour).zfill(2) + ":" + str(minute).zfill(2) + ":" + str(second).zfill(2)
    time = time + "." + str(nanosec).zfill(9)
    return get_date() + " " + time

def get_string(len = None):
    if len == None:
        len = random.randint(0, 1024)
    chars = string.letters + string.digits
    return ''.join(random.choice(chars) for i in range(len))

def get_char(len = None):
    if len == None:
        len = random.randint(0, 255)
    return get_string(len)

def get_varchar(len = None):
    if len == None:
        len = random.randint(0, 65355)
    return get_string(len)

def get_boolean():
    return bool(random.getrandbits(1))

def get_binary(len = None):
    if len == None:
        len = random.randint(0, 1024)
    bytes = bytearray(os.urandom(len))
    # Disallow reserved chars (LF, CR, column delimiter)
    for idx, _ in enumerate(bytes):
        while bytes[idx] in map(ord, exacsv.reserved_chars):
            bytes[idx] = os.urandom(1)
    return bytes


def parse_primitive_type(col):
    idx = string.find(col, '(')
    ridx = string.rfind(col, ')')
    if idx < 0 and ridx < 0:
        return [col]
    elif idx > 0 and ridx > 0:
        type = col[: idx].strip()
        specs = col[idx + 1 : ridx]
        specs = specs.split(',')
        specs = map(str.strip, specs)
        return [type, specs]
    else:
        raise RuntimeError("Invalid primitive type: " + col)

def parse_complex_type(col, st_idx):
    idx = re.search(r'[<>,]', col[st_idx :]).start()
    if idx == None:
        raise RuntimeError("Invalid complex type: " + col[st_ : idx])
    type_list = []
    idx += st_idx
    if col[idx] == '<':
        # Start complex type
        type = col[st_idx : idx].strip()
        (idx, types) = parse_complex_type(col, idx + 1)
        for t in types:
            type_list.append(parse_primitive_type(t))
        if len(type_list) == 1:
            type_list = type_list[0]
        return (idx, [type, type_list])
    elif col[idx] == '>':
        # End complex type
        nidx = idx + 1
        if nidx >= len(col):
            nidx = None
        type = (col[st_idx : idx]).strip()
        return (nidx, [type] if type else [])
    else:
        # Comma-separated type list in complex type
        lp_idx = string.find(col, '(', st_idx, idx)
        rp_idx = string.find(col, ')', st_idx)
        if lp_idx >= 0 and rp_idx >= 0 and lp_idx < idx and rp_idx > idx:
            # Comma is part of type, separates parentheses (e.g. 'decimal(18,0)')
            idx = rp_idx
            type_list.append(col[st_idx : idx + 1].strip())
        else:
            type_list.append((col[st_idx : idx]).strip())
        idx += 1
        while True:
            nidx = re.search(r'[<>,]', col[idx :]).start()
            if nidx == None:
                raise RuntimeError("Invalid complex type in list: " + col[idx :])
            nidx += idx
            if col[nidx] == ',':
                lp_idx = string.find(col, '(', idx, nidx)
                rp_idx = string.find(col, ')', idx)
                if lp_idx >= 0 and rp_idx >= 0 and lp_idx < nidx and rp_idx > nidx:
                    # Comma is part of type, separates parentheses (e.g. 'decimal(18,0)')
                    nidx = rp_idx
                    type_list.append(col[idx : rp_idx + 1].strip())
                else:
                    type_list.append((col[idx : nidx]).strip())
                idx = nidx + 1
            else:
                (idx, types) = parse_complex_type(col, idx)
                if len(types) == 1:
                    # Single item
                    type_list.append(types[0])
                elif len(types) > 1:
                    # Complex type
                    type_list.append(types)
                if idx is not None and col[idx] == ',':
                    # List continues
                    idx = idx + 1
                else:
                    break;
        return (idx, type_list)

def parse_col(col):
    if '<' in col:
        return parse_complex_type(col, 0)[1]
    else:
        return parse_primitive_type(col)

def parse_cols(cols):
    cols = map(str.strip, cols)
    col_list = []
    for col in cols:
        col_list.append(parse_col(col))
    return col_list

def write_col(file, col, recur_lvl):
    if len(col) == 0:
        raise RuntimeError("Invalid column definition: " + col)
    if recur_lvl >= 8:
        raise RuntimeError("Invalid recursion level: " + recur_lvl)
    if isinstance(col, basestring):
        type = str.lower(col)
    else:
        type = str.lower(col[0])
    if type == 'tinyint':
        file.write(str(get_tinyint()))
    elif type == 'smallint':
        file.write(str(get_smallint()))
    elif type == 'int':
        file.write(str(get_int()))
    elif type == 'bigint':
        file.write(str(get_bigint()))
    elif type == 'float':
        file.write(str(get_float()))
    elif type == 'double':
        file.write(str(get_double()))
    elif type == 'decimal':
        prec = 10
        scale = 0
        if len(col) > 1:
            num_args = len(col[1])
            if num_args > 0:
                prec = col[1][0]
            if num_args > 1:
                scale = col[1][1]
        file.write(str(get_decimal(int(prec), int(scale))))
    elif type == 'date':
        file.write(get_date())
    elif type == 'timestamp':
        file.write(get_timestamp())
    elif type == 'string':
        size = None
        if len(col) > 1:
            size = int(col[1][0])
        file.write(get_string(size))
    elif type == 'char':
        size = None
        if len(col) > 1:
            size = int(col[1][0])
        file.write(get_char(size))
    elif type == 'varchar':
        size = None
        if len(col) > 1:
            size = int(col[1][0])
        file.write(get_varchar(size))
    elif type == 'boolean':
        file.write(str(get_boolean()))
    elif type == 'binary':
        size = None
        if len(col) > 1:
            size = int(col[1][0])
        file.write(str(get_binary(size)))
    elif type == 'array':
        if len(col) < 2:
            raise RuntimeError("Invalid array definition: " + col)
        max_size = 10
        size = random.randint(0, max_size)
        sep = bytearray(1)
        sep[0] = recur_lvl + 1
        for i in range(size):
            if i != 0:
                 file.write(sep)
            write_col(file, col[1], recur_lvl + 1)
    elif type == 'map':
        if len(col) < 2 or len(col[1]) < 2:
            raise RuntimeError("Invalid map definition: " + col)
        sep = bytearray(1)
        sep[0] = recur_lvl + 2
        write_col(file, col[1][0], recur_lvl + 1)
        file.write(sep)
        write_col(file, col[1][1], recur_lvl + 1)
    elif type == 'struct':
        if len(col) < 2 or len(col[1]) < 1:
            raise RuntimeError("Invalid struct definition: " + col)
        sep = bytearray(1)
        sep[0] = recur_lvl + 1
        for i in range(len(col[1])):
            if i != 0:
                 file.write(sep)
            s_type = col[1][i][0]
            # Ignore struct field name if given before type (e.g., 'name:type')
            s_type = [s_type.split(':')[-1]]
            write_col(file, s_type, recur_lvl + 1)
    elif type == 'uniontype':
        if len(col) < 2 or len(col[1]) < 1:
            raise RuntimeError("Invalid uniontype definition: " + col)
        sep = bytearray(1)
        sep[0] = recur_lvl + 1
        type_idx = random.randint(0, len(col[1]) - 1)
        file.write(str(type_idx))
        file.write(sep)
        write_col(file, col[1][type_idx], recur_lvl + 1)
    else:
        raise RuntimeError("Invalid type: " + type)

def gen_csv_local(col_defs, num_rows, path, has_id_col = False):
    col_list = parse_cols(col_defs)
    with open(path, 'w') as file:
        for i in range(int(num_rows)):
            for c in range(len(col_list)):
                if c > 0:
                    file.write(',')
                if c == 0 and has_id_col:
                    file.write(str(i))
                else:
                    write_col(file, col_list[c], 0)
            file.write('\n')

def gen_csv_webhdfs(col_defs, num_rows, host, user, hdfs_path, has_id_col = False, use_kerberos = False):
    col_list = parse_cols(col_defs)
    with tempdir() as tmpdir:
        filename = os.path.basename(hdfs_path)
        tmp_path = os.path.join(tmpdir, filename)
        with open(tmp_path, 'w') as file:
            for i in range(int(num_rows)):
                for c in range(len(col_list)):
                    if c > 0:
                        file.write(',')
                    if c == 0 and has_id_col:
                        file.write(str(i))
                    else:
                        write_col(file, col_list[c], 1)
                file.write('\n')
        webhdfs.create_file(host, user, hdfs_path, tmp_path, use_kerberos = use_kerberos)
