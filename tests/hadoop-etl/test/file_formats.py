#!/usr/bin/env python2.7
# encoding: utf8

import os
import sys

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))
sys.path.append(os.path.realpath(__file__ + '/../../lib'))

import udf
import utils
import datagen
import hadoopenv


class TestTextfile(utils.HiveTestCase):
    hive_file_format = 'textfile'
    hive_table = '{file_format}_%s'.format(file_format = hive_file_format)
    hive_col_types = ['decimal(36,0)', 'tinyint', 'smallint', 'int', 'bigint', 'decimal(18,5)', 'float', 'double', \
                    'date', 'timestamp', 'string', 'char(20)', 'varchar(50)', 'boolean', 'binary', \
                    'array<timestamp>', 'map<int, string>', 'struct<s1:int, s2:smallint, s3:date>', \
                    'uniontype<bigint, timestamp, boolean, varchar(100)>']
    exa_col_types = ['decimal(36,0)', 'tinyint', 'smallint', 'int', 'bigint', 'decimal(18,5)', 'float', 'double', \
                    'date', 'timestamp', 'varchar(5000)', 'char(20)', 'varchar(50)', 'boolean', 'varchar(5000) ASCII', \
                    'varchar(5000)', 'varchar(5000)', 'varchar(5000)', \
                    'varchar(5000)']
    num_rows = 1000
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)


class TestRcfile(utils.HiveTestCase):
    hive_file_format = 'rcfile'
    hive_table = '{file_format}_%s'.format(file_format = hive_file_format)
    hive_col_types = ['decimal(36,0)', 'tinyint', 'smallint', 'int', 'bigint', 'decimal(18,5)', 'float', 'double', \
                    'date', 'timestamp', 'string', 'char(20)', 'varchar(50)', 'boolean', 'binary', \
                    'array<timestamp>', 'map<int, string>', 'struct<s1:int, s2:smallint, s3:date>', \
                    'uniontype<bigint, timestamp, boolean, varchar(100)>']
    exa_col_types = ['decimal(36,0)', 'tinyint', 'smallint', 'int', 'bigint', 'decimal(18,5)', 'float', 'double', \
                    'date', 'timestamp', 'varchar(5000)', 'char(20)', 'varchar(50)', 'boolean', 'varchar(5000) ASCII', \
                    'varchar(5000)', 'varchar(5000)', 'varchar(5000)', \
                    'varchar(5000)']
    num_rows = 1000
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)


class TestOrc(utils.HiveTestCase):
    hive_file_format = 'orc'
    hive_table = '{file_format}_%s'.format(file_format = hive_file_format)
    hive_col_types = ['decimal(36,0)', 'tinyint', 'smallint', 'int', 'bigint', 'decimal(18,5)', 'float', 'double', \
                    'date', 'timestamp', 'string', 'char(20)', 'varchar(50)', 'boolean', 'binary', \
                    'array<timestamp>', 'map<int, string>', 'struct<s1:int, s2:smallint, s3:date>', \
                    'uniontype<tinyint, int>']
    exa_col_types = ['decimal(36,0)', 'tinyint', 'smallint', 'int', 'bigint', 'decimal(18,5)', 'float', 'double', \
                    'date', 'timestamp', 'varchar(5000)', 'char(20)', 'varchar(50)', 'boolean', 'varchar(5000) ASCII', \
                    'varchar(5000)', 'varchar(5000)', 'varchar(5000)', \
                    'varchar(5000)']
    num_rows = 1000
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)


class TestSequencefile(utils.HiveTestCase):
    hive_file_format = 'sequencefile'
    hive_table = '{file_format}_%s'.format(file_format = hive_file_format)
    hive_col_types = ['decimal(36,0)', 'tinyint', 'smallint', 'int', 'bigint', 'decimal(18,5)', 'float', 'double', \
                    'date', 'timestamp', 'string', 'char(20)', 'varchar(50)', 'boolean', 'binary', \
                    'array<timestamp>', 'map<int, string>', 'struct<s1:int, s2:smallint, s3:date>', \
                    'uniontype<bigint, timestamp, boolean, varchar(100)>']
    exa_col_types = ['decimal(36,0)', 'tinyint', 'smallint', 'int', 'bigint', 'decimal(18,5)', 'float', 'double', \
                    'date', 'timestamp', 'varchar(5000)', 'char(20)', 'varchar(50)', 'boolean', 'varchar(5000) ASCII', \
                    'varchar(5000)', 'varchar(5000)', 'varchar(5000)', \
                    'varchar(5000)']
    num_rows = 1000
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)


class TestParquet(utils.HiveTestCase):
    hive_file_format = 'parquet'
    hive_table = '{file_format}_%s'.format(file_format = hive_file_format)
    # Unsupported types: date, binary, uniontype
    hive_col_types = ['decimal(36,0)', 'tinyint', 'smallint', 'int', 'bigint', 'decimal(18,5)', 'float', 'double', \
                    'timestamp', 'string', 'char(20)', 'varchar(50)', 'boolean', \
                    'array<int>', 'map<string, int>', 'struct<s1:int, s2:smallint, s3:string>'
                    ]
    exa_col_types = ['decimal(36,0)', 'tinyint', 'smallint', 'int', 'bigint', 'decimal(18,5)', 'float', 'double', \
                    'timestamp', 'varchar(5000)', 'char(20)', 'varchar(50)', 'boolean', \
                    'varchar(5000)', 'varchar(5000)', 'varchar(5000)'
                    ]
    num_rows = 1000
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)


class TestAvro(utils.HiveTestCase):
    hive_file_format = 'avro'
    hive_table = '{file_format}_%s'.format(file_format = hive_file_format)
    # Unsupported types: date, timestamp, uniontype
    hive_col_types = ['decimal(36,0)', 'tinyint', 'smallint', 'int', 'bigint', 'decimal(18,5)', 'float', 'double', \
                    'string', 'char(20)', 'varchar(50)', 'boolean', 'binary', \
                    'array<boolean>', 'map<string, int>', \
                    ]
    exa_col_types = ['decimal(36,0)', 'tinyint', 'smallint', 'int', 'bigint', 'decimal(18,5)', 'float', 'double', \
                    'varchar(5000)', 'char(20)', 'varchar(50)', 'boolean', 'varchar(5000) ASCII', \
                    'varchar(5000)', 'varchar(5000)', \
                    ]
    num_rows = 1000
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)


if __name__ == '__main__':
    udf.main()
