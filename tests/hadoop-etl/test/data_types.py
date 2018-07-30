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


class TestTinyint(utils.HiveTestCase):
    hive_table = 'tinyint_%s'
    hive_col_types = ['decimal(36,0)', 'tinyint', 'tinyint', 'tinyint', 'tinyint', 'tinyint']
    exa_col_types = hive_col_types
    num_rows = 1000
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)


class TestSmallint(utils.HiveTestCase):
    hive_table = 'smallint_%s'
    hive_col_types = ['decimal(36,0)', 'smallint', 'smallint', 'smallint', 'smallint', 'smallint']
    exa_col_types = hive_col_types
    num_rows = 1000
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)


class TestInt(utils.HiveTestCase):
    hive_table = 'int_%s'
    hive_col_types = ['decimal(36,0)', 'int', 'int', 'int', 'int', 'int']
    exa_col_types = hive_col_types
    num_rows = 1000
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)


class TestBigint(utils.HiveTestCase):
    hive_table = 'bigint_%s'
    hive_col_types = ['decimal(36,0)', 'bigint', 'bigint', 'bigint', 'bigint', 'bigint']
    exa_col_types = hive_col_types
    num_rows = 1000
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)


class TestDecimal(utils.HiveTestCase):
    hive_table = 'decimal_%s'
    hive_col_types = ['decimal(36,0)', 'decimal(3,3)', 'decimal(9,1)', 'decimal(10,0)', 'decimal(18,8)', 'decimal(36,0)']
    exa_col_types = hive_col_types
    num_rows = 1000
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)


class TestFloat(utils.HiveTestCase):
    hive_table = 'float_%s'
    hive_col_types = ['decimal(36,0)', 'float', 'float', 'float', 'float', 'float']
    exa_col_types = hive_col_types
    num_rows = 1000
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)


class TestDouble(utils.HiveTestCase):
    hive_table = 'double_%s'
    hive_col_types = ['decimal(36,0)', 'double', 'double', 'double', 'double', 'double']
    exa_col_types = hive_col_types
    num_rows = 1000
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)


class TestBigintErr(utils.HiveTestCase):
    hive_table = 'bigint_err_%s'
    hive_col_types = ['bigint']
    exa_col_types = ['tinyint']
    num_rows = 1000
    has_id_col = False

    def test(self):
        create_table = utils.get_create_table_query(self.hive_table, self.exa_col_types)
        self.query(udf.fixindent(create_table))
        create_as_import = utils.get_import_query(self.hive_table, self.exa_col_types, self.use_kerberos)
        with self.assertRaisesRegexp(Exception, '.*ExaDataTypeException: emit column.*'):
            self.query(udf.fixindent(create_as_import))


class TestDoubleErr(utils.HiveTestCase):
    hive_table = 'double_err_%s'
    hive_col_types = ['double']
    exa_col_types = ['int']
    num_rows = 1000
    has_id_col = False

    def test(self):
        create_table = utils.get_create_table_query(self.hive_table, self.exa_col_types)
        self.query(udf.fixindent(create_table))
        create_as_import = utils.get_import_query(self.hive_table, self.exa_col_types, self.use_kerberos)
        with self.assertRaisesRegexp(Exception, '.*ExaDataTypeException: emit column.*'):
            self.query(udf.fixindent(create_as_import))


class TestDate(utils.HiveTestCase):
    hive_table = 'date_%s'
    hive_col_types = ['decimal(36,0)', 'date', 'date', 'date', 'date', 'date']
    exa_col_types = hive_col_types
    num_rows = 1000
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)


class TestTimestamp(utils.HiveTestCase):
    hive_table = 'timestamp_%s'
    hive_col_types = ['decimal(36,0)', 'timestamp', 'timestamp', 'timestamp', 'timestamp', 'timestamp']
    exa_col_types = hive_col_types
    num_rows = 1000
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)


class TestDateErr(utils.HiveTestCase):
    hive_table = 'date_err_%s'
    hive_col_types = ['date']
    exa_col_types = ['int']
    num_rows = 1000
    has_id_col = False

    def test(self):
        create_table = utils.get_create_table_query(self.hive_table, self.exa_col_types)
        self.query(udf.fixindent(create_table))
        create_as_import = utils.get_import_query(self.hive_table, self.exa_col_types, self.use_kerberos)
        with self.assertRaisesRegexp(Exception, '.*ExaDataTypeException: emit column.*'):
            self.query(udf.fixindent(create_as_import))


class TestTimestampErr(utils.HiveTestCase):
    hive_table = 'timestamp_err_%s'
    hive_col_types = ['timestamp']
    exa_col_types = ['date']
    num_rows = 1000
    has_id_col = False

    def test(self):
        create_table = utils.get_create_table_query(self.hive_table, self.exa_col_types)
        self.query(udf.fixindent(create_table))
        create_as_import = utils.get_import_query(self.hive_table, self.exa_col_types, self.use_kerberos)
        with self.assertRaisesRegexp(Exception, '.*ExaDataTypeException: emit column.*'):
            self.query(udf.fixindent(create_as_import))


class TestString(utils.HiveTestCase):
    hive_table = 'string_%s'
    hive_col_types = ['decimal(36,0)', 'string', 'string', 'string', 'string', 'string']
    exa_col_types = ['decimal(36,0)', 'varchar(2000)', 'varchar(2000)', 'varchar(2000)', 'varchar(2000)', 'varchar(2000)']
    num_rows = 1000
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)


class TestChar(utils.HiveTestCase):
    hive_table = 'char_%s'
    hive_col_types = ['decimal(36,0)', 'char(10)', 'char(40)', 'char(100)', 'char(150)', 'char(255)']
    exa_col_types = hive_col_types
    num_rows = 1000
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)


class TestVarchar(utils.HiveTestCase):
    hive_table = 'varchar_%s'
    hive_col_types = ['decimal(36,0)', 'varchar(10)', 'varchar(50)', 'varchar(250)', 'varchar(5000)', 'varchar(65355)']
    exa_col_types = hive_col_types
    num_rows = 1000
    has_id_col = True

    def test(self):
        if self.use_kerberos:
            # Hive bug: VARCHAR with Kerberos causes Hive and/or the Hive ODBC driver to crash for unknown reasons
            utils.HiveTestCase.num_rows = 1
            utils.HiveTestCase.hive_col_types = ['decimal(36,0)', 'varchar(1)', 'varchar(1)', 'varchar(1)', 'varchar(1)', 'varchar(1)']
            pass
        else:
            utils.test_import(self)
            utils.validate_import_odbc(self)


class TestStringErr(utils.HiveTestCase):
    hive_table = 'string_err_%s'
    hive_col_types = ['string']
    exa_col_types = ['int']
    num_rows = 1000
    has_id_col = False

    def test(self):
        create_table = utils.get_create_table_query(self.hive_table, self.exa_col_types)
        self.query(udf.fixindent(create_table))
        create_as_import = utils.get_import_query(self.hive_table, self.exa_col_types, self.use_kerberos)
        with self.assertRaisesRegexp(Exception, '.*ExaDataTypeException: emit column.*'):
            self.query(udf.fixindent(create_as_import))


class TestBoolean(utils.HiveTestCase):
    hive_table = 'boolean_%s'
    hive_col_types = ['decimal(36,0)', 'boolean', 'boolean', 'boolean', 'boolean', 'boolean']
    exa_col_types = hive_col_types
    num_rows = 1000
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)


class TestBinary(utils.HiveTestCase):
    hive_table = 'binary_%s'
    hive_col_types = ['decimal(36,0)', 'binary', 'binary', 'binary', 'binary', 'binary']
    exa_col_types = ['decimal(36,0)', 'varchar(5000) ASCII', 'varchar(5000) ASCII', 'varchar(5000) ASCII', 'varchar(5000) ASCII', 'varchar(5000) ASCII']
    num_rows = 1000
    has_id_col = True

    def test(self):
        utils.test_import(self)
        # Hive has a bug, which sometimes gives the wrong data for a column
        # Verification using ODBC (not CSV) will still succeed because both wrong values are the same
        utils.validate_import_odbc(self)


class TestBooleanErr(utils.HiveTestCase):
    hive_table = 'boolean_err_%s'
    hive_col_types = ['boolean']
    exa_col_types = ['int']
    num_rows = 1000
    has_id_col = False

    def test(self):
        create_table = utils.get_create_table_query(self.hive_table, self.exa_col_types)
        self.query(udf.fixindent(create_table))
        create_as_import = utils.get_import_query(self.hive_table, self.exa_col_types, self.use_kerberos)
        with self.assertRaisesRegexp(Exception, '.*ExaDataTypeException: emit column.*'):
            self.query(udf.fixindent(create_as_import))


class TestBinaryErr(utils.HiveTestCase):
    hive_table = 'binary_err_%s'
    hive_col_types = ['binary']
    exa_col_types = ['int']
    num_rows = 1000
    has_id_col = False

    def test(self):
        create_table = utils.get_create_table_query(self.hive_table, self.exa_col_types)
        self.query(udf.fixindent(create_table))
        create_as_import = utils.get_import_query(self.hive_table, self.exa_col_types, self.use_kerberos)
        with self.assertRaisesRegexp(Exception, '.*ExaDataTypeException: emit column.*'):
            self.query(udf.fixindent(create_as_import))


class TestArray(utils.HiveTestCase):
    hive_table = 'array_%s'
    # No tests with floating-point types. The values can be slightly different and the JSON comparison will fail.
    hive_col_types = ['decimal(36,0)', 'array<decimal(18,0)>', 'array<bigint>', 'array<varchar(200)>', 'array<timestamp>', 'array<boolean>']
    exa_col_types = ['decimal(36,0)', 'varchar(5000)', 'varchar(5000)', 'varchar(5000)', 'varchar(5000)', 'varchar(5000)']
    num_rows = 1000
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)


class TestMap(utils.HiveTestCase):
    hive_table = 'map_%s'
    # No tests with floating-point types. The values can be slightly different and the JSON comparison will fail.
    hive_col_types = ['decimal(36,0)', 'map<int, string>', 'map<varchar(50), decimal(10,5)>', 'map<tinyint, timestamp>', 'map<date, boolean>', 'map<char(50), bigint>']
    exa_col_types = ['decimal(36,0)', 'varchar(5000)', 'varchar(5000)', 'varchar(5000)', 'varchar(5000)', 'varchar(5000)']
    num_rows = 1000
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)


class TestStruct(utils.HiveTestCase):
    hive_table = 'struct_%s'
    # No tests with floating-point types. The values can be slightly different and the JSON comparison will fail.
    # Provide struct field names for table creation
    hive_col_types = ['decimal(36,0)', 'struct<s1:tinyint, s2:string>', 'struct<s1:int, s2:smallint, s3:date>', 'struct<s1:bigint, s2:timestamp, s3:boolean, s4:varchar(100)>']
    exa_col_types = ['decimal(36,0)', 'varchar(5000)', 'varchar(5000)', 'varchar(5000)']
    num_rows = 1000
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)


class TestUnion(utils.HiveTestCase):
    hive_table = 'union_%s'
    # No tests with floating-point types. The values can be slightly different and the JSON comparison will fail.
    hive_col_types = ['decimal(36,0)', 'uniontype<tinyint, string>', 'uniontype<int, smallint, date>', 'uniontype<bigint, timestamp, boolean, varchar(100)>']
    exa_col_types = ['decimal(36,0)', 'varchar(5000)', 'varchar(5000)', 'varchar(5000)']
    num_rows = 1000
    has_id_col = True

    def test(self):
        utils.test_import(self)
        utils.validate_import_odbc(self)


class TestArrayErr(utils.HiveTestCase):
    hive_table = 'array_err_%s'
    hive_col_types = ['array<int>']
    exa_col_types = ['int']
    num_rows = 1000
    has_id_col = False

    def test(self):
        create_table = utils.get_create_table_query(self.hive_table, self.exa_col_types)
        self.query(udf.fixindent(create_table))
        create_as_import = utils.get_import_query(self.hive_table, self.exa_col_types, self.use_kerberos)
        with self.assertRaisesRegexp(Exception, '.*ExaDataTypeException: emit column.*'):
            self.query(udf.fixindent(create_as_import))


if __name__ == '__main__':
    udf.main()
