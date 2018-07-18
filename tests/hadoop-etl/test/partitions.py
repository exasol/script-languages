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


class TestPartitionDate(utils.HiveTestCase):
    hive_file_format = 'textfile'
    hive_table = '{file_format}_%s'.format(file_format = hive_file_format)
    hive_col_types = ['decimal(36,0)', 'tinyint', 'smallint', 'int', 'bigint', 'decimal(18,5)', 'float', 'double', \
                      'date', 'timestamp', 'string', 'char(20)', 'varchar(50)', 'boolean', 'binary']
    # Partition columns listed last
    exa_col_types = ['decimal(36,0)', 'tinyint', 'smallint', 'int', 'bigint', 'decimal(18,5)', 'float', 'double', \
                     'timestamp', 'varchar(5000)', 'char(20)', 'varchar(50)', 'boolean', 'varchar(5000) ASCII', 'date']
    hive_config_props = ['hive.exec.dynamic.partition=true', \
                         'hive.exec.dynamic.partition.mode=nonstrict']
    hive_partition_col_nums = [9]
    num_rows = 100
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)


class TestPartitionTinyint(utils.HiveTestCase):
    hive_file_format = 'textfile'
    hive_table = '{file_format}_%s'.format(file_format = hive_file_format)
    hive_col_types = ['decimal(36,0)', 'tinyint', 'smallint', 'int', 'bigint', 'decimal(18,5)', 'float', 'double', \
                      'date', 'timestamp', 'string', 'char(20)', 'varchar(50)', 'boolean', 'binary']
    # Partition columns listed last
    exa_col_types = ['decimal(36,0)', 'smallint', 'int', 'bigint', 'decimal(18,5)', 'float', 'double', \
                     'date', 'timestamp', 'varchar(5000)', 'char(20)', 'varchar(50)', 'boolean', 'varchar(5000) ASCII', 'tinyint']
    hive_config_props = ['hive.exec.dynamic.partition=true', \
                         'hive.exec.dynamic.partition.mode=nonstrict']
    hive_partition_col_nums = [2]
    num_rows = 100
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)


class TestPartitionChar(utils.HiveTestCase):
    hive_file_format = 'textfile'
    hive_table = '{file_format}_%s'.format(file_format = hive_file_format)
    hive_col_types = ['decimal(36,0)', 'tinyint', 'smallint', 'int', 'bigint', 'decimal(18,5)', 'float', 'double', \
                      'date', 'timestamp', 'string', 'char(20)', 'varchar(50)', 'boolean', 'binary']
    # Partition columns listed last
    exa_col_types = ['decimal(36,0)', 'tinyint', 'smallint', 'int', 'bigint', 'decimal(18,5)', 'float', 'double', \
                     'date', 'timestamp', 'varchar(5000)', 'varchar(50)', 'boolean', 'varchar(5000) ASCII', 'char(20)']
    hive_config_props = ['hive.exec.dynamic.partition=true', \
                         'hive.exec.dynamic.partition.mode=nonstrict']
    hive_partition_col_nums = [12]
    num_rows = 100
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)


class TestPartitionDouble(utils.HiveTestCase):
    hive_file_format = 'textfile'
    hive_table = '{file_format}_%s'.format(file_format = hive_file_format)
    hive_col_types = ['decimal(36,0)', 'tinyint', 'smallint', 'int', 'bigint', 'decimal(18,5)', 'float', 'double', \
                      'date', 'timestamp', 'string', 'char(20)', 'varchar(50)', 'boolean', 'binary']
    # Partition columns listed last
    exa_col_types = ['decimal(36,0)', 'tinyint', 'smallint', 'int', 'bigint', 'decimal(18,5)', 'float', \
                     'date', 'timestamp', 'varchar(5000)', 'char(20)', 'varchar(50)', 'boolean', 'varchar(5000) ASCII', 'double']
    hive_config_props = ['hive.exec.dynamic.partition=true', \
                         'hive.exec.dynamic.partition.mode=nonstrict']
    hive_partition_col_nums = [8]
    num_rows = 100
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)


class TestPartitionTinyintDate(utils.HiveTestCase):
    hive_file_format = 'textfile'
    hive_table = '{file_format}_%s'.format(file_format = hive_file_format)
    hive_col_types = ['decimal(36,0)', 'tinyint', 'smallint', 'int', 'bigint', 'decimal(18,5)', 'float', 'double', \
                      'date', 'timestamp', 'string', 'char(20)', 'varchar(50)', 'boolean', 'binary']
    # Partition columns listed last
    exa_col_types = ['decimal(36,0)', 'smallint', 'int', 'bigint', 'decimal(18,5)', 'float', 'double', \
                     'timestamp', 'varchar(5000)', 'char(20)', 'varchar(50)', 'boolean', 'varchar(5000) ASCII', 'tinyint', 'date']
    hive_config_props = ['hive.exec.dynamic.partition=true', \
                         'hive.exec.dynamic.partition.mode=nonstrict']
    hive_partition_col_nums = [2, 9]
    num_rows = 100
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)


# Hive bug: Boolean partition values are always given as 'true'
'''
class TestPartitionBooleanTimestamp(utils.HiveTestCase):
    hive_file_format = 'textfile'
    hive_table = '{file_format}_%s'.format(file_format = hive_file_format)
    hive_col_types = ['decimal(36,0)', 'tinyint', 'smallint', 'int', 'bigint', 'decimal(18,5)', 'float', 'double', \
                      'date', 'timestamp', 'string', 'char(20)', 'varchar(50)', 'boolean', 'binary']
    # Partition columns listed last
    exa_col_types = ['decimal(36,0)', 'tinyint', 'smallint', 'int', 'bigint', 'decimal(18,5)', 'float', 'double', \
                     'date', 'varchar(5000)', 'char(20)', 'varchar(50)', 'varchar(5000) ASCII', 'boolean', 'timestamp']
    hive_config_props = ['hive.exec.dynamic.partition=true', \
                         'hive.exec.dynamic.partition.mode=nonstrict']
    hive_partition_col_nums = [14, 10]
    num_rows = 100
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)


class TestPartitionBooleanCharInt(utils.HiveTestCase):
    hive_file_format = 'textfile'
    hive_table = '{file_format}_%s'.format(file_format = hive_file_format)
    hive_col_types = ['decimal(36,0)', 'tinyint', 'smallint', 'int', 'bigint', 'decimal(18,5)', 'float', 'double', \
                      'date', 'timestamp', 'string', 'char(20)', 'varchar(50)', 'boolean', 'binary']
    # Partition columns listed last
    exa_col_types = ['decimal(36,0)', 'tinyint', 'smallint', 'bigint', 'decimal(18,5)', 'float', 'double', \
                     'date', 'timestamp', 'varchar(5000)', 'varchar(50)', 'varchar(5000) ASCII', 'boolean', 'char(20)', 'int']
    hive_config_props = ['hive.exec.dynamic.partition=true', \
                         'hive.exec.dynamic.partition.mode=nonstrict']
    hive_partition_col_nums = [14, 12, 4]
    num_rows = 100
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)
'''


class TestPartitionCharDateInt(utils.HiveTestCase):
    hive_file_format = 'textfile'
    hive_table = '{file_format}_%s'.format(file_format = hive_file_format)
    hive_col_types = ['decimal(36,0)', 'tinyint', 'smallint', 'int', 'bigint', 'decimal(18,5)', 'float', 'double', \
                      'date', 'timestamp', 'string', 'char(20)', 'varchar(50)', 'boolean', 'binary']
    # Partition columns listed last
    exa_col_types = ['decimal(36,0)', 'tinyint', 'smallint', 'bigint', 'decimal(18,5)', 'float', 'double', \
                     'timestamp', 'varchar(5000)', 'varchar(50)', 'boolean', 'varchar(5000) ASCII', 'char(20)', 'date', 'int']
    hive_config_props = ['hive.exec.dynamic.partition=true', \
                         'hive.exec.dynamic.partition.mode=nonstrict']
    hive_partition_col_nums = [12, 9, 4]
    num_rows = 100
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)


if __name__ == '__main__':
    udf.main()
