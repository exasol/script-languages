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


class TestBucketTinyint(utils.HiveTestCase):
    hive_file_format = 'textfile'
    hive_table = '{file_format}_%s'.format(file_format = hive_file_format)
    hive_col_types = ['decimal(36,0)', 'tinyint', 'smallint', 'int', 'bigint', 'decimal(18,5)', 'float', 'double', \
                      'date', 'timestamp', 'string', 'char(20)', 'varchar(50)', 'boolean', 'binary']
    exa_col_types = ['decimal(36,0)', 'tinyint', 'smallint', 'int', 'bigint', 'decimal(18,5)', 'float', 'double', \
                     'date', 'timestamp', 'varchar(5000)', 'char(20)', 'varchar(50)', 'boolean', 'varchar(5000) ASCII']
    hive_config_props = ['hive.enforce.bucketing=true']
    hive_bucket_col_nums = [2]
    hive_num_buckets = 10
    num_rows = 1000
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)


class TestBucketDecimal(utils.HiveTestCase):
    hive_file_format = 'textfile'
    hive_table = '{file_format}_%s'.format(file_format = hive_file_format)
    hive_col_types = ['decimal(36,0)', 'tinyint', 'smallint', 'int', 'bigint', 'decimal(18,5)', 'float', 'double', \
                      'date', 'timestamp', 'string', 'char(20)', 'varchar(50)', 'boolean', 'binary']
    exa_col_types = ['decimal(36,0)', 'tinyint', 'smallint', 'int', 'bigint', 'decimal(18,5)', 'float', 'double', \
                     'date', 'timestamp', 'varchar(5000)', 'char(20)', 'varchar(50)', 'boolean', 'varchar(5000) ASCII']
    hive_config_props = ['hive.enforce.bucketing=true']
    hive_bucket_col_nums = [6]
    hive_num_buckets = 56
    num_rows = 1000
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)


class TestBucketBigint(utils.HiveTestCase):
    hive_file_format = 'textfile'
    hive_table = '{file_format}_%s'.format(file_format = hive_file_format)
    hive_col_types = ['decimal(36,0)', 'tinyint', 'smallint', 'int', 'bigint', 'decimal(18,5)', 'float', 'double', \
                      'date', 'timestamp', 'string', 'char(20)', 'varchar(50)', 'boolean', 'binary']
    exa_col_types = ['decimal(36,0)', 'tinyint', 'smallint', 'int', 'bigint', 'decimal(18,5)', 'float', 'double', \
                     'date', 'timestamp', 'varchar(5000)', 'char(20)', 'varchar(50)', 'boolean', 'varchar(5000) ASCII']
    hive_config_props = ['hive.enforce.bucketing=true']
    hive_bucket_col_nums = [5]
    hive_num_buckets = 30
    num_rows = 1000
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)


class TestBucketString(utils.HiveTestCase):
    hive_file_format = 'textfile'
    hive_table = '{file_format}_%s'.format(file_format = hive_file_format)
    hive_col_types = ['decimal(36,0)', 'tinyint', 'smallint', 'int', 'bigint', 'decimal(18,5)', 'float', 'double', \
                      'date', 'timestamp', 'string', 'char(20)', 'varchar(50)', 'boolean', 'binary']
    exa_col_types = ['decimal(36,0)', 'tinyint', 'smallint', 'int', 'bigint', 'decimal(18,5)', 'float', 'double', \
                     'date', 'timestamp', 'varchar(5000)', 'char(20)', 'varchar(50)', 'boolean', 'varchar(5000) ASCII']
    hive_config_props = ['hive.enforce.bucketing=true']
    hive_bucket_col_nums = [11]
    hive_num_buckets = 64
    num_rows = 1000
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)


class TestBucketTimestamp(utils.HiveTestCase):
    hive_file_format = 'textfile'
    hive_table = '{file_format}_%s'.format(file_format = hive_file_format)
    hive_col_types = ['decimal(36,0)', 'tinyint', 'smallint', 'int', 'bigint', 'decimal(18,5)', 'float', 'double', \
                      'date', 'timestamp', 'string', 'char(20)', 'varchar(50)', 'boolean', 'binary']
    exa_col_types = ['decimal(36,0)', 'tinyint', 'smallint', 'int', 'bigint', 'decimal(18,5)', 'float', 'double', \
                     'date', 'timestamp', 'varchar(5000)', 'char(20)', 'varchar(50)', 'boolean', 'varchar(5000) ASCII']
    hive_config_props = ['hive.enforce.bucketing=true']
    hive_bucket_col_nums = [10]
    hive_num_buckets = 15
    num_rows = 1000
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)


class TestBucketIntDate(utils.HiveTestCase):
    hive_file_format = 'textfile'
    hive_table = '{file_format}_%s'.format(file_format = hive_file_format)
    hive_col_types = ['decimal(36,0)', 'tinyint', 'smallint', 'int', 'bigint', 'decimal(18,5)', 'float', 'double', \
                      'date', 'timestamp', 'string', 'char(20)', 'varchar(50)', 'boolean', 'binary']
    exa_col_types = ['decimal(36,0)', 'tinyint', 'smallint', 'int', 'bigint', 'decimal(18,5)', 'float', 'double', \
                     'date', 'timestamp', 'varchar(5000)', 'char(20)', 'varchar(50)', 'boolean', 'varchar(5000) ASCII']
    hive_config_props = ['hive.enforce.bucketing=true']
    hive_bucket_col_nums = [4, 9]
    hive_num_buckets = 20
    num_rows = 1000
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)


class TestBucketBooleanSmallint(utils.HiveTestCase):
    hive_file_format = 'textfile'
    hive_table = '{file_format}_%s'.format(file_format = hive_file_format)
    hive_col_types = ['decimal(36,0)', 'tinyint', 'smallint', 'int', 'bigint', 'decimal(18,5)', 'float', 'double', \
                      'date', 'timestamp', 'string', 'char(20)', 'varchar(50)', 'boolean', 'binary']
    exa_col_types = ['decimal(36,0)', 'tinyint', 'smallint', 'int', 'bigint', 'decimal(18,5)', 'float', 'double', \
                     'date', 'timestamp', 'varchar(5000)', 'char(20)', 'varchar(50)', 'boolean', 'varchar(5000) ASCII']
    hive_config_props = ['hive.enforce.bucketing=true']
    hive_bucket_col_nums = [14, 3]
    hive_num_buckets = 12
    num_rows = 1000
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)


class TestBucketCharIntDate(utils.HiveTestCase):
    hive_file_format = 'textfile'
    hive_table = '{file_format}_%s'.format(file_format = hive_file_format)
    hive_col_types = ['decimal(36,0)', 'tinyint', 'smallint', 'int', 'bigint', 'decimal(18,5)', 'float', 'double', \
                      'date', 'timestamp', 'string', 'char(20)', 'varchar(50)', 'boolean', 'binary']
    exa_col_types = ['decimal(36,0)', 'tinyint', 'smallint', 'int', 'bigint', 'decimal(18,5)', 'float', 'double', \
                     'date', 'timestamp', 'varchar(5000)', 'char(20)', 'varchar(50)', 'boolean', 'varchar(5000) ASCII']
    hive_config_props = ['hive.enforce.bucketing=true']
    hive_bucket_col_nums = [12, 4, 9]
    hive_num_buckets = 24
    num_rows = 1000
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)


if __name__ == '__main__':
    udf.main()
