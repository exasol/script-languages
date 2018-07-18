#!/usr/opt/bs-python-2.7/bin/python
# encoding: utf8

import os
import sys

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))
sys.path.append(os.path.realpath(__file__ + '/../../lib'))

import udf
import utils
import datagen
import hadoopenv


class TestDeflate(utils.HiveTestCase):
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
    hive_config_props = ['hive.exec.compress.output=true', \
                        'mapred.output.compression.codec=org.apache.hadoop.io.compress.DefaultCodec']
    num_rows = 1000
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)


class TestGzip(utils.HiveTestCase):
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
    hive_config_props = ['hive.exec.compress.output=true', \
                        'mapred.output.compression.codec=org.apache.hadoop.io.compress.GzipCodec']
    num_rows = 1000
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)


class TestBzip2(utils.HiveTestCase):
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
    hive_config_props = ['hive.exec.compress.output=true', \
                        'mapred.output.compression.codec=org.apache.hadoop.io.compress.BZip2Codec']
    num_rows = 1000
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)


class TestLzo(utils.HiveTestCase):
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
    hive_config_props = ['hive.exec.compress.output=true', \
                        'mapred.output.compression.codec=com.hadoop.compression.lzo.LzoCodec']
    num_rows = 1000
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)


class TestLzop(utils.HiveTestCase):
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
    hive_config_props = ['hive.exec.compress.output=true', \
                        'mapred.output.compression.codec=com.hadoop.compression.lzo.LzopCodec']
    num_rows = 1000
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)


'''
# Requires native libraries
class TestLz4(utils.HiveTestCase):
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
    hive_config_props = ['hive.exec.compress.output=true', \
                        'mapred.output.compression.codec=org.apache.hadoop.io.compress.Lz4Codec']
    num_rows = 1000
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)
'''


'''
# Requires native libraries
class TestSnappy(utils.HiveTestCase):
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
    hive_config_props = ['hive.exec.compress.output=true', \
                        'mapred.output.compression.codec=org.apache.hadoop.io.compress.SnappyCodec']
    num_rows = 1000
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)
'''


if __name__ == '__main__':
    udf.main()
