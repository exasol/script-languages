#!/usr/bin/env python2.7
# encoding: utf8

import os
import sys
import logging
import string
import binascii
import json
import re
import datetime
import uuid
import subprocess

sys.path.append(os.path.realpath(__file__ + '/../../lib'))

import udf
from exatest.clients.odbc import ODBCClient
import hadoopenv
import webhdfs
import datagen
import exacsv
import hiveodbc

sys.path.append(hadoopenv.krb_dir)
import create_conn


class HiveTestCase(udf.TestCase):
    def __init__(self, *args):
        super(HiveTestCase, self).__init__(*args)
        self.uuid = None
        self.json_result_types = ['array', 'map', 'struct', 'uniontype']
        self.use_kerberos = False
        self._set_params()

    def setUp(self):
        success = setup(self)
        if not success:
            raise RuntimeError('setUp failed')

    def tearDown(self):
        success = teardown(self)
        if not success:
            raise RuntimeError('tearDown failed')

    def _set_params(self):
        test_params = udf.opts.testparam.split(',') if udf.opts.testparam else None
        if test_params:
            for param in test_params:
                if param.lower() == 'kerberos':
                    self.use_kerberos = True


def setup(test):
    script_dir = '/x/u/zg1089/hadoop-etl/sql'
    test.uuid = uuid.uuid4().hex
    test.hive_table = test.hive_table % test.uuid
    exa_schema = hadoopenv.hive_schema
    client = test._client
    log = test.log
    success = True
    if not client:
        log.critical('client not given')
        return False
    script_dir = os.path.realpath(script_dir)
    if not os.path.isdir(script_dir):
        log.critical('%s does not exist', script_dir)
        return False
    log.info('searching for function definitions beneath %s', script_dir)
    # Schema for ETL scripts
    udf._sql(client, 'DROP SCHEMA ETL CASCADE', may_fail=True)
    udf._sql(client, 'CREATE SCHEMA ETL', fatal_error=True)
    for file in udf._walk(script_dir):
        log.info('loading functions from file %s', file)
        success = udf._load_file(client, file, None) and success
    # Schema for tests
    udf._sql(client, 'DROP SCHEMA %s CASCADE' % exa_schema, may_fail=True)
    udf._sql(client, 'CREATE SCHEMA %s' % exa_schema, fatal_error=True)
    if test.use_kerberos:
        create_conn = get_krb_connection_stmt()
        udf._sql(client, create_conn, fatal_error=True)
        os.environ['KRB5_CONFIG'] = hadoopenv.krb_dir + "/" + hadoopenv.krb_conf
        keytab = hadoopenv.krb_dir + "/" + hadoopenv.krb_keytab
        ld_lib_orig = os.environ['LD_LIBRARY_PATH']
        os.environ['LD_LIBRARY_PATH'] = ''
        # Call kinit to get Kerberos tickets
        subprocess.check_output(['kinit', '-k', '-t', keytab, hadoopenv.krb_test_princ])
        os.environ['LD_LIBRARY_PATH'] = ld_lib_orig
    client.commit()
    create_hive_table(test)
    return success


def teardown(test):
    success = True
    drop_hive_table(test)
    if test.use_kerberos:
        ld_lib_orig = os.environ['LD_LIBRARY_PATH']
        os.environ['LD_LIBRARY_PATH'] = ''
        # Call kdestroy to delete Kerberos tickets
        subprocess.check_call(['kdestroy'])
        os.environ['LD_LIBRARY_PATH'] = ld_lib_orig
    return success


def create_hive_table(test):
    text_table = test.hive_table
    if hasattr(test, 'hive_file_format'):
        # Intermediary textfile table
        text_table = 'text_' + text_table
    hive = hiveodbc.HiveOdbc(test.use_kerberos)
    hive.connect()
    hive_col_defs = get_column_defs(test.hive_col_types)
    query = 'CREATE TABLE %s.%s' % (hadoopenv.hive_schema, text_table)
    query += '(%s)' % ', '.join(hive_col_defs)
    query += ''' row format delimited'''
    query += ''' fields terminated by '%s' ''' % exacsv.column_delim
    udf._sql(hive, query)
    path = hadoopenv.hive_hdfs_path + '%s.csv' % text_table
    datagen.gen_csv_webhdfs(test.hive_col_types, test.num_rows, hadoopenv.webhdfs_host, hadoopenv.user, path, test.has_id_col, test.use_kerberos)
    query = '''LOAD DATA INPATH '%s' OVERWRITE INTO TABLE %s.%s''' % (path, hadoopenv.hive_schema, text_table)
    udf._sql(hive, query)
    # Create table in another file format from textfile table
    if hasattr(test, 'hive_file_format'):
        # Get partitions
        hive_part_col_defs = []
        if hasattr(test, 'hive_partition_col_nums'):
            for col in test.hive_partition_col_nums:
                hive_part_col_defs.append(hive_col_defs[col - 1])
            for col in hive_part_col_defs:
                hive_col_defs.remove(col)
        # Get buckets
        hive_bucket_col_names = []
        if hasattr(test, 'hive_bucket_col_nums'):
            for col in test.hive_bucket_col_nums:
                hive_bucket_col_names.append(hive_col_defs[col - 1])
                hive_bucket_col_names = get_column_names(hive_bucket_col_names)
        # Get table properties
        hive_tbl_props = []
        if hasattr(test, 'hive_table_props'):
            hive_tbl_props = test.hive_table_props
        query = 'CREATE TABLE %s.%s' % (hadoopenv.hive_schema, test.hive_table)
        query += '(%s)' % ', '.join(hive_col_defs)
        if hive_part_col_defs:
            query += ' partitioned by (%s)' % ', '.join(hive_part_col_defs)
        if hive_bucket_col_names:
            query += ' clustered by (%s) into %s buckets' % (', '.join(hive_bucket_col_names), test.hive_num_buckets)
        query += ''' stored as %s''' % test.hive_file_format
        if hive_tbl_props:
            query += ' tblproperties (%s)' % ', '.join(hive_tbl_props)
        udf._sql(hive, query)
        if hasattr(test, 'hive_config_props'):
            for opt in test.hive_config_props:
                udf._sql(hive, '''SET %s''' % opt)
        query = '''INSERT OVERWRITE TABLE %s.%s''' % (hadoopenv.hive_schema, test.hive_table)
        if hive_part_col_defs:
            query += ' partition (%s)' % ', '.join(get_column_names(hive_part_col_defs))
        query += ''' SELECT %s from %s.%s''' % (', '.join(get_column_names(hive_col_defs + hive_part_col_defs)), hadoopenv.hive_schema, text_table)
        udf._sql(hive, query)
    hive.close()


def drop_hive_table(test):
    hive = hiveodbc.HiveOdbc(test.use_kerberos)
    hive.connect()
    query = 'DROP TABLE %s.%s' % (hadoopenv.hive_schema, test.hive_table)
    udf._sql(hive, query)
    if hasattr(test, 'hive_file_format'):
        # Intermediary textfile table
        text_table = 'text_' + test.hive_table
        query = 'DROP TABLE %s.%s' % (hadoopenv.hive_schema, text_table)
        udf._sql(hive, query)
    hive.close()


def test_import(test):
    create_table = get_create_table_query(test.hive_table, test.exa_col_types)
    test.query(udf.fixindent(create_table))
    create_as_import = get_import_query(test.hive_table, test.exa_col_types, test.use_kerberos)
    test.query(udf.fixindent(create_as_import))
    count = get_count_query(test.hive_table)
    rows = test.query(udf.fixindent(count))
    test.assertRowEqual((test.num_rows,), rows[0])


def validate_import_odbc(test):
    hive = hiveodbc.HiveOdbc(test.use_kerberos)
    hive.connect()
    hive_col_defs = get_column_defs(test.hive_col_types)
    hive_rows = hive.query('select * from %s.%s order by %s' % (hadoopenv.hive_schema, test.hive_table, get_column_names(hive_col_defs)[0]))
    hive_col_info = hive.columns(table=test.hive_table, schema=hadoopenv.hive_schema)
    hive.close()
    if len(hive_rows) <= 0:
        raise RuntimeError('Hive table %s has no rows' % test.hive_table)
    if len(hive_rows[0]) <= 0:
        raise RuntimeError('Hive table %s has no columns' % test.hive_table)
    exa_col_defs = get_column_defs(test.exa_col_types)
    exa_rows = test.query('select * from exa_%s order by %s' % (test.hive_table, get_column_names(exa_col_defs)[0]))
    test.assertEqual(len(hive_rows), len(exa_rows))
    test.assertEqual(len(hive_rows[0]), len(exa_rows[0]))
    validate_rows(test, hive_col_info, hive_rows, exa_rows)


def validate_rows(test, hive_col_info, hive_rows, exa_rows):
    rows = range(0, len(hive_rows))
    cols = range(1 if test.has_id_col else 0, len(hive_rows[0]))
    regex = re.compile(r'''^{(?!")([\w.+-]*)(?!"):''')
    for ri in rows:
        for ci in cols:
            hive_val = hive_rows[ri][ci]
            exa_val = exa_rows[ri][ci]
            if any(test.hive_col_types[ci].lower().startswith(s) for s in test.json_result_types):
                # Fix Hive JSON--name of object must be quoted (string)
                if hive_val:
                    hive_val = regex.sub(r'''{"\1":''', hive_val)
                # Deserialize JSON to Python object
                hive_res = json.loads(hive_val) if hive_val else None
                exa_res = json.loads(exa_val) if exa_val else None
            elif isinstance(hive_val, datetime.datetime):
                # Remove microsecond resolution from timestamp
                hive_val = hive_val.replace(microsecond = hive_val.microsecond / 1000 * 1000);
            elif hive_col_info[ci].type_name.lower() == 'binary':
                if not hive_val:
                    hive_val = None
                else:
                    hive_val = binascii.hexlify(hive_val)
            if not hive_val and hive_col_info[ci].type_name.lower() == 'string':
                # Empty string -> NULL
                hive_val = None
            test.assertEqual(hive_val, exa_val)


def validate_import_csv(test):
    (val_create, val_import) = get_val_import_queries(test.hive_table, test.hive_col_types, test.exa_col_types, test.use_kerberos)
    if val_create and not val_import:
        raise RuntimeError('CREATE statement without an IMPORT statement')
    elif val_import and not val_create:
        raise RuntimeError('IMPORT statement without a CREATE statement')
    if val_create and val_import:
        test.query(udf.fixindent(val_create))
        test.query(udf.fixindent(val_import))
        val_count = get_val_count_query(test.hive_table)
        rows = test.query(udf.fixindent(val_count))
        test.assertRowEqual((test.num_rows,), rows[0])
    exa_col_defs = get_column_defs(test.exa_col_types)
    if test.has_id_col:
        del exa_col_defs[0]
    binary_cols = [idx for idx, col in enumerate(test.hive_col_types) if col == 'binary']
    if binary_cols:
        # Get binary data (from csv)
        binary_data = get_csv_data(test, test.hive_table, binary_cols)
        convert_to_hex(binary_data)
    for idx, col in enumerate(exa_col_defs):
        exa_col_idx = idx + 1 if test.has_id_col else idx
        if exa_col_idx in binary_cols:
            # Validate binary column
            query_text = get_select_query(test.hive_table, col)
            exa_rows = test.query(udf.fixindent(query_text))
            csv_rows = [(row[binary_cols.index(exa_col_idx)],) for row in binary_data]
            test.assertRowsEqual(exa_rows, csv_rows)
        else:
            # Validate all other columns
            val_compare = get_val_compare_query(test.hive_table, col, test.has_id_col)
            rows = test.query(udf.fixindent(val_compare))
            test.assertRowsEqual([], rows)


def convert_to_hex(rows):
    for r, row in enumerate(rows):
        for c, _ in enumerate(row):
            if rows[r][c]:
                rows[r][c] = binascii.hexlify(rows[r][c])
                # Empty string -> None
                if not rows[r][c]:
                    rows[r][c] = None


def get_csv_data(test, hive_table, req_columns):
    data = []
    path = hadoopenv.hive_hdfs_path + '%s.csv' % hive_table
    resp = webhdfs.get_file_http_response_object(hadoopenv.webhdfs_host, hadoopenv.user, path)
    while True:
        line = exacsv.get_csv_line(resp)
        if not line:
            break
        row = exacsv.get_csv_columns(resp, line, req_columns)
        data.append(row)
    return data


def get_hcat_files_query(hive_table, use_kerberos):
    query_text = '''SELECT HCAT_TABLE_FILES('%s', '%s', '%s',''' \
                % (hadoopenv.hive_schema, hive_table, hadoopenv.webhcat_host)
    if use_kerberos:
        query_text += ''' '%s', nproc(), '', '', 'kerberos', 'hadoop')''' \
                % (hadoopenv.krb_hdfs_princ)
    else:
        query_text += ''' '%s', nproc())''' \
                % (hadoopenv.user)
    return query_text


def get_krb_connection_stmt():
    conn_name = 'krb_conn'
    replace = True
    user = hadoopenv.krb_test_princ
    conf = hadoopenv.krb_dir + '/' + hadoopenv.krb_conf
    keytab = hadoopenv.krb_dir + '/' + hadoopenv.krb_keytab
    return create_conn.getcreateconn(conn_name, replace, user, conf, keytab)


def get_column_defs(col_types):
    col_defs = []
    for idx, col in enumerate(col_types):
        col_defs.append('c' + str(idx + 1) + ' ' + col)
    return col_defs


def get_column_names(col_defs):
    return [c.split()[0] for c in col_defs]


def get_hive_import_query(hive_table, exa_col_types, use_kerberos):
    exa_col_defs = get_column_defs(exa_col_types)
    query_text ='''SELECT IMPORT_HIVE_TABLE_FILES(hdfspath, input_format, serde, column_info, partition_info, serde_props, hdfs_server_port, hdfs_user, auth_type, conn_name, '')
                    emits (%s)
                FROM (
                    %s
                )
                group by import_partition''' \
                % (', '.join(exa_col_defs), get_hcat_files_query(hive_table, use_kerberos))
    return query_text


def get_create_table_query(hive_table, exa_col_types):
    query_text = '''CREATE OR REPLACE TABLE %s (%s)''' \
                % ('exa_' + hive_table, ', '.join(get_column_defs(exa_col_types)))
    return query_text


def get_count_query(hive_table):
    query_text = '''SELECT COUNT(*) FROM %s''' \
                % ('exa_' + hive_table)
    return query_text


def get_import_query(hive_table, exa_col_types, use_kerberos):
    query_text = '''IMPORT INTO %s FROM SCRIPT ETL.IMPORT_HCAT_TABLE WITH
                    HCAT_DB = '%s'
                    HCAT_TABLE = '%s'
                    HCAT_ADDRESS = '%s'
                    HDFS_USER = '%s'
                ''' \
                % ('exa_' + hive_table, hadoopenv.hive_schema, hive_table, hadoopenv.webhcat_host, hadoopenv.krb_hdfs_princ)
    if use_kerberos:
        query_text += '''AUTH_TYPE = 'kerberos' AUTH_KERBEROS_CONNECTION = 'krb_conn' '''
    return query_text


def get_val_import_query(hive_table, hive_col_types, exa_col_types, use_kerberos):
    files = webhdfs.list_dir(hadoopenv.webhdfs_host, hadoopenv.user, hadoopenv.hive_schema, hive_table, use_kerberos)
    url = 'http://' + hadoopenv.webhdfs_host
    url += '/webhdfs/v1/user/hive/warehouse/%s.db/%s/' % (hadoopenv.hive_schema, hive_table)
    binary_cols = [idx for idx, col in enumerate(hive_col_types) if col == 'binary']
    query_text = ''
    if len(binary_cols) == len(exa_col_types):
        return query_text
    query_text = '''IMPORT INTO %s FROM CSV AT '%s'  ''' \
                % ('exa_val_' + hive_table, url)
    for file in files:
        query_text += '''FILE '%s?op=open&user.name=%s'  ''' % (file, hadoopenv.user)
    if binary_cols and len(binary_cols) != len(exa_col_types):
        select_cols = []
        for idx, col in enumerate(exa_col_types):
            if idx not in binary_cols:
                select_cols.append(str(idx + 1))
        query_text += '''(%s) ''' % (', '.join(select_cols))
    query_text += '''COLUMN DELIMITER = ''  '''
    return query_text


def get_val_create_table_query(hive_table, hive_col_types, exa_col_types):
    exa_col_defs = get_column_defs(exa_col_types)
    binary_cols = [idx for idx, col in enumerate(hive_col_types) if col == 'binary']
    query_text = ''
    if not binary_cols:
        query_text = '''CREATE OR REPLACE TABLE %s (%s)''' \
                    % ('exa_val_' + hive_table, ', '.join(exa_col_defs))
    elif len(binary_cols) != len(exa_col_types):
        query_text = '''CREATE OR REPLACE TABLE %s (''' \
                    % ('exa_val_' + hive_table)
        select_cols = []
        for idx, col in enumerate(exa_col_defs):
            if idx not in binary_cols:
                select_cols.append(col)
        query_text += '''%s)''' % (', '.join(select_cols))
    return query_text


def get_val_import_queries(hive_table, hive_col_types, exa_col_types, use_kerberos):
    return (get_val_create_table_query(hive_table, hive_col_types, exa_col_types),
            get_val_import_query(hive_table, hive_col_types, exa_col_types, use_kerberos))


def get_val_count_query(hive_table):
    query_text = '''SELECT COUNT(*) FROM %s''' \
                % ('exa_val_' + hive_table)
    return query_text


def get_select_query(hive_table, col_def):
    idx = string.index(col_def, ' ')
    col_name = col_def[: idx]
    query_text = '''SELECT %s FROM %s''' \
                % (col_name, 'exa_' + hive_table)
    return query_text


def get_val_compare_query(hive_table, col_def, has_id_col):
    exa_table = 'exa_' + hive_table
    exa_val_table = 'exa_val_' + hive_table
    idx = string.index(col_def, ' ')
    col_name = col_def[: idx]
    col_type = col_def[idx + 1 :]
    if has_id_col:
        id_col = 'c1'
        if col_type in ['double', 'float', 'number', 'real']:
            # Approximate equality check for floating point values
            # abs(a - b) <= (abs(a) + abs(b)) / 2 * tolerance --> OK
            tolerance = '1e-3'
            query_text = '''select %(exa_table)s.%(col_name)s, %(exa_val_table)s.%(col_name)s
                            from %(exa_table)s, %(exa_val_table)s
                            where
                            %(exa_table)s.%(id_col)s = %(exa_val_table)s.%(id_col)s
                            and
                            abs(%(exa_table)s.%(col_name)s - %(exa_val_table)s.%(col_name)s) >
                            ((abs(%(exa_table)s.%(col_name)s) + abs(%(exa_val_table)s.%(col_name)s)) / 2) * %(tolerance)s''' \
                            % locals()
        else:
            query_text = '''select %(exa_table)s.%(col_name)s, %(exa_val_table)s.%(col_name)s
                            from %(exa_table)s, %(exa_val_table)s
                            where
                            %(exa_table)s.%(id_col)s = %(exa_val_table)s.%(id_col)s
                            and
                            %(exa_table)s.%(col_name)s != %(exa_val_table)s.%(col_name)s''' \
                            % locals()
    else:
        query_text = '''((select %(col_name)s from %(exa_table)s) minus (select %(col_name)s from %(exa_val_table)s))
                        union
                        ((select %(col_name)s from %(exa_val_table)s) minus (select %(col_name)s from %(exa_table)s))''' \
                        % locals()
    return query_text
