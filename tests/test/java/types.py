#!/usr/opt/bs-python-2.7/bin/python
# encoding: utf8

import os
import sys

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf

class JavaTypesInteger(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')
        self.query('CREATE TABLE T (intVal DECIMAL(9,0), longVal DECIMAL(18,0), bigdecimalVal DECIMAL(36,0), decimalVal DECIMAL(9,2), \
                    doubleVal DOUBLE, doubleIntVal DOUBLE, stringVal VARCHAR(100), booleanVal BOOLEAN, dateVal DATE, timestampVal TIMESTAMP)')
        self.query('''INSERT INTO T values (123456789, 123456789123456789, 123456789123456789123456789123456789, 1234567.12, \
                    123456789.123, 15.0, 'string#String!12345', true, '2014-05-21', '2014-05-21 15:13:30.123')''')

    def test_integer_integer(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                INT_INT(x SHORTINT) RETURNS SHORTINT AS
                class INT_INT {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getInteger("x");
                    }
                }
                '''))
        rows = self.query('SELECT INT_INT(intVal) FROM T')
        self.assertRowsEqual([(123456789,)], rows)

    def test_integer_long(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                INT_LONG(x INTEGER) RETURNS SHORTINT AS
                class INT_LONG {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getInteger("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'column can only have maximum value of'):
            rows = self.query('SELECT INT_LONG(longVal) FROM T')

    def test_integer_bigdecimal(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                INT_BIGDEC(x BIGINT) RETURNS SHORTINT AS
                class INT_BIGDEC {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getInteger("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'java.lang.ArithmeticException: Overflow'):
            rows = self.query('SELECT INT_BIGDEC(bigdecimalVal) FROM T')

    def test_integer_decimal(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                INT_DECIMAL(x DECIMAL(9,2)) RETURNS SHORTINT AS
                class INT_DECIMAL {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getInteger("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'java.lang.ArithmeticException: Rounding necessary'):
            rows = self.query('SELECT INT_DECIMAL(decimalVal) FROM T')

    def test_integer_double(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                INT_DOUBLE(x DOUBLE) RETURNS SHORTINT AS
                class INT_DOUBLE {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getInteger("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'has a non-integer value of'):
            rows = self.query('SELECT INT_DOUBLE(doubleVal) FROM T')

    def test_integer_double_int(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                INT_DOUBLE_INT(x DOUBLE) RETURNS SHORTINT AS
                class INT_DOUBLE_INT {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getInteger("x");
                    }
                }
                '''))
        rows = self.query('SELECT INT_DOUBLE_INT(doubleIntVal) FROM T')
        self.assertRowsEqual([(15,)], rows)

    def test_integer_string(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                INT_STRING(x VARCHAR(100)) RETURNS SHORTINT AS
                class INT_STRING {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getInteger("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'getInteger cannot convert column'):
            rows = self.query('SELECT INT_STRING(stringVal) FROM T')

    def test_integer_boolean(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                INT_BOOL(x BOOLEAN) RETURNS SHORTINT AS
                class INT_BOOL {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getInteger("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'getInteger cannot convert column'):
            rows = self.query('SELECT INT_BOOL(booleanVal) FROM T')

    def test_integer_date(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                INT_DATE(x DATE) RETURNS SHORTINT AS
                class INT_DATE {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getInteger("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'getInteger cannot convert column'):
            rows = self.query('SELECT INT_DATE(dateVal) FROM T')

    def test_integer_timestamp(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                INT_TIMESTAMP(x TIMESTAMP) RETURNS SHORTINT AS
                class INT_TIMESTAMP {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getInteger("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'getInteger cannot convert column'):
            rows = self.query('SELECT INT_TIMESTAMP(timestampVal) FROM T')

class JavaTypesLong(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')
        self.query('CREATE TABLE T (intVal DECIMAL(9,0), longVal DECIMAL(18,0), bigdecimalVal DECIMAL(36,0), decimalVal DECIMAL(9,2), \
                    doubleVal DOUBLE, doubleIntVal DOUBLE, stringVal VARCHAR(100), booleanVal BOOLEAN, dateVal DATE, timestampVal TIMESTAMP)')
        self.query('''INSERT INTO T values (123456789, 123456789123456789, 123456789123456789123456789123456789, 1234567.12, \
                    123456789.123, 15.0, 'string#String!12345', true, '2014-05-21', '2014-05-21 15:13:30.123')''')

    def test_long_integer(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                LONG_INT(x SHORTINT) RETURNS INTEGER AS
                class LONG_INT {
                    static long run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getLong("x");
                    }
                }
                '''))
        rows = self.query('SELECT LONG_INT(intVal) FROM T')
        self.assertRowsEqual([(123456789,)], rows)

    def test_long_long(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                LONG_LONG(x INTEGER) RETURNS INTEGER AS
                class LONG_LONG {
                    static long run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getLong("x");
                    }
                }
                '''))
        rows = self.query('SELECT LONG_LONG(longVal) FROM T')
        self.assertRowsEqual([(123456789123456789,)], rows)

    def test_long_bigdecimal(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                LONG_BIGDEC(x BIGINT) RETURNS INTEGER AS
                class LONG_BIGDEC {
                    static long run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getLong("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'java.lang.ArithmeticException: Overflow'):
            rows = self.query('SELECT LONG_BIGDEC(bigdecimalVal) FROM T')

    def test_long_decimal(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                LONG_DECIMAL(x DECIMAL(9,2)) RETURNS INTEGER AS
                class LONG_DECIMAL {
                    static long run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getLong("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'java.lang.ArithmeticException: Rounding necessary'):
            rows = self.query('SELECT LONG_DECIMAL(decimalVal) FROM T')

    def test_long_double(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                LONG_DOUBLE(x DOUBLE) RETURNS INTEGER AS
                class LONG_DOUBLE {
                    static long run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getLong("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'has a non-integer value of'):
            rows = self.query('SELECT LONG_DOUBLE(doubleVal) FROM T')

    def test_long_double_int(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                LONG_DOUBLE_INT(x DOUBLE) RETURNS INTEGER AS
                class LONG_DOUBLE_INT {
                    static long run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getLong("x");
                    }
                }
                '''))
        rows = self.query('SELECT LONG_DOUBLE_INT(doubleIntVal) FROM T')
        self.assertRowsEqual([(15,)], rows)

    def test_long_string(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                LONG_STRING(x VARCHAR(100)) RETURNS INTEGER AS
                class LONG_STRING {
                    static long run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getLong("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'getLong cannot convert column'):
            rows = self.query('SELECT LONG_STRING(stringVal) FROM T')

    def test_long_boolean(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                LONG_BOOL(x BOOLEAN) RETURNS INTEGER AS
                class LONG_BOOL {
                    static long run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getLong("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'getLong cannot convert column'):
            rows = self.query('SELECT LONG_BOOL(booleanVal) FROM T')

    def test_long_date(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                LONG_DATE(x DATE) RETURNS INTEGER AS
                class LONG_DATE {
                    static long run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getLong("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'getLong cannot convert column'):
            rows = self.query('SELECT LONG_DATE(dateVal) FROM T')

    def test_long_timestamp(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                LONG_TIMESTAMP(x TIMESTAMP) RETURNS INTEGER AS
                class LONG_TIMESTAMP {
                    static long run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getLong("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'getLong cannot convert column'):
            rows = self.query('SELECT LONG_TIMESTAMP(timestampVal) FROM T')

class JavaTypesBigDecimal(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')
        self.query('CREATE TABLE T (intVal DECIMAL(9,0), longVal DECIMAL(18,0), bigdecimalVal DECIMAL(36,0), decimalVal DECIMAL(9,2), \
                    doubleVal DOUBLE, doubleIntVal DOUBLE, stringVal VARCHAR(100), booleanVal BOOLEAN, dateVal DATE, timestampVal TIMESTAMP)')
        self.query('''INSERT INTO T values (123456789, 123456789123456789, 123456789123456789123456789123456789, 1234567.12, \
                    123456789.123, 15.0, 'string#String!12345', true, '2014-05-21', '2014-05-21 15:13:30.123')''')

    def test_bigdecimal_integer(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                BIGDECIMAL_INT(x SHORTINT) RETURNS BIGINT AS
                import java.math.BigDecimal;
                class BIGDECIMAL_INT {
                    static BigDecimal run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getBigDecimal("x");
                    }
                }
                '''))
        rows = self.query('SELECT BIGDECIMAL_INT(intVal) FROM T')
        self.assertRowsEqual([(123456789,)], rows)

    def test_bigdecimal_long(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                BIGDECIMAL_LONG(x INTEGER) RETURNS BIGINT AS
                import java.math.BigDecimal;
                class BIGDECIMAL_LONG {
                    static BigDecimal run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getBigDecimal("x");
                    }
                }
                '''))
        rows = self.query('SELECT BIGDECIMAL_LONG(longVal) FROM T')
        self.assertRowsEqual([(123456789123456789,)], rows)

    def test_bigdecimal_bigdecimal(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                BIGDECIMAL_BIGDEC(x BIGINT) RETURNS BIGINT AS
                import java.math.BigDecimal;
                class BIGDECIMAL_BIGDEC {
                    static BigDecimal run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getBigDecimal("x");
                    }
                }
                '''))
        rows = self.query('SELECT BIGDECIMAL_BIGDEC(bigdecimalVal) FROM T')
        self.assertRowsEqual([(123456789123456789123456789123456789,)], rows)

    def test_bigdecimal_decimal(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                BIGDECIMAL_DECIMAL(x DECIMAL(9,2)) RETURNS VARCHAR(100) AS
                import java.math.BigDecimal;
                class BIGDECIMAL_DECIMAL {
                    static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getBigDecimal("x").toString();
                    }
                }
                '''))
        rows = self.query('SELECT BIGDECIMAL_DECIMAL(decimalVal) FROM T')
        self.assertRowsEqual([('1234567.12',)], rows)

    def test_bigdecimal_double(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                BIGDECIMAL_DOUBLE(x DOUBLE) RETURNS VARCHAR(100) AS
                import java.math.BigDecimal;
                class BIGDECIMAL_DOUBLE {
                    static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getBigDecimal("x").toString();
                    }
                }
                '''))
        rows = self.query('''SELECT 'OK' FROM (SELECT CAST(BIGDECIMAL_DOUBLE(doubleVal) as DOUBLE) as val FROM T) WHERE val BETWEEN 123456789.122999 AND 123456789.1230001''')
        self.assertRowsEqual([('OK',)], rows)

    def test_bigdecimal_double_int(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                BIGDECIMAL_DOUBLE_INT(x DOUBLE) RETURNS BIGINT AS
                import java.math.BigDecimal;
                class BIGDECIMAL_DOUBLE_INT {
                    static BigDecimal run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getBigDecimal("x");
                    }
                }
                '''))
        rows = self.query('SELECT BIGDECIMAL_DOUBLE_INT(doubleIntVal) FROM T')
        self.assertRowsEqual([(15,)], rows)

    def test_bigdecimal_string(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                BIGDECIMAL_STRING(x VARCHAR(100)) RETURNS BIGINT AS
                import java.math.BigDecimal;
                class BIGDECIMAL_STRING {
                    static BigDecimal run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getBigDecimal("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'getBigDecimal cannot convert column'):
            rows = self.query('SELECT BIGDECIMAL_STRING(stringVal) FROM T')

    def test_bigdecimal_boolean(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                BIGDECIMAL_BOOL(x BOOLEAN) RETURNS BIGINT AS
                import java.math.BigDecimal;
                class BIGDECIMAL_BOOL {
                    static BigDecimal run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getBigDecimal("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'getBigDecimal cannot convert column'):
            rows = self.query('SELECT BIGDECIMAL_BOOL(booleanVal) FROM T')

    def test_bigdecimal_date(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                BIGDECIMAL_DATE(x DATE) RETURNS BIGINT AS
                import java.math.BigDecimal;
                class BIGDECIMAL_DATE {
                    static BigDecimal run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getBigDecimal("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'getBigDecimal cannot convert column'):
            rows = self.query('SELECT BIGDECIMAL_DATE(dateVal) FROM T')

    def test_bigdecimal_timestamp(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                BIGDECIMAL_TIMESTAMP(x TIMESTAMP) RETURNS BIGINT AS
                import java.math.BigDecimal;
                class BIGDECIMAL_TIMESTAMP {
                    static BigDecimal run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getBigDecimal("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'getBigDecimal cannot convert column'):
            rows = self.query('SELECT BIGDECIMAL_TIMESTAMP(timestampVal) FROM T')

class JavaTypesDouble(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')
        self.query('CREATE TABLE T (intVal DECIMAL(9,0), longVal DECIMAL(18,0), bigdecimalVal DECIMAL(36,0), decimalVal DECIMAL(9,2), \
                    doubleVal DOUBLE, doubleIntVal DOUBLE, stringVal VARCHAR(100), booleanVal BOOLEAN, dateVal DATE, timestampVal TIMESTAMP)')
        self.query('''INSERT INTO T values (123456789, 123456789123456789, 123456789123456789123456789123456789, 1234567.12, \
                    123456789.123, 15.0, 'string#String!12345', true, '2014-05-21', '2014-05-21 15:13:30.123')''')

    def test_double_integer(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                DOUBLE_INT(x SHORTINT) RETURNS DOUBLE AS
                class DOUBLE_INT {
                    static double run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getDouble("x");
                    }
                }
                '''))
        rows = self.query('SELECT DOUBLE_INT(intVal) FROM T')
        self.assertRowsEqual([(123456789,)], rows)

    def test_double_long(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                DOUBLE_LONG(x INTEGER) RETURNS DOUBLE AS
                class DOUBLE_LONG {
                    static double run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getDouble("x");
                    }
                }
                '''))
        rows = self.query('''SELECT TO_CHAR(DOUBLE_LONG(longVal), '9.99999999999999EEEE') FROM T''')
        self.assertRowsEqual([('1.23456789123457E17',)], rows)

    def test_double_bigdecimal(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                DOUBLE_BIGDEC(x BIGINT) RETURNS DOUBLE AS
                class DOUBLE_BIGDEC {
                    static double run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getDouble("x");
                    }
                }
                '''))
        rows = self.query('''SELECT TO_CHAR(DOUBLE_BIGDEC(bigdecimalVal), '9.99999999999999EEEE') FROM T''')
        self.assertRowsEqual([('1.23456789123457E35',)], rows)

    def test_double_decimal(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                DOUBLE_DECIMAL(x DECIMAL(9,2)) RETURNS DOUBLE AS
                class DOUBLE_DECIMAL {
                    static double run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getDouble("x");
                    }
                }
                '''))
        rows = self.query('SELECT DOUBLE_DECIMAL(decimalVal) FROM T')
        self.assertRowsEqual([(1234567.12,)], rows)

    def test_double_double(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                DOUBLE_DOUBLE(x DOUBLE) RETURNS DOUBLE AS
                class DOUBLE_DOUBLE {
                    static double run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getDouble("x");
                    }
                }
                '''))
        rows = self.query('SELECT DOUBLE_DOUBLE(doubleVal) FROM T')
        self.assertRowsEqual([(123456789.123,)], rows)

    def test_double_double_int(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                DOUBLE_DOUBLE_INT(x DOUBLE) RETURNS DOUBLE AS
                class DOUBLE_DOUBLE_INT {
                    static double run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getDouble("x");
                    }
                }
                '''))
        rows = self.query('SELECT DOUBLE_DOUBLE_INT(doubleIntVal) FROM T')
        self.assertRowsEqual([(15,)], rows)

    def test_double_string(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                DOUBLE_STRING(x VARCHAR(100)) RETURNS DOUBLE AS
                class DOUBLE_STRING {
                    static double run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getDouble("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'getDouble cannot convert column'):
            rows = self.query('SELECT DOUBLE_STRING(stringVal) FROM T')

    def test_double_boolean(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                DOUBLE_BOOL(x BOOLEAN) RETURNS DOUBLE AS
                class DOUBLE_BOOL {
                    static double run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getDouble("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'getDouble cannot convert column'):
            rows = self.query('SELECT DOUBLE_BOOL(booleanVal) FROM T')

    def test_double_date(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                DOUBLE_DATE(x DATE) RETURNS DOUBLE AS
                class DOUBLE_DATE {
                    static double run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getDouble("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'getDouble cannot convert column'):
            rows = self.query('SELECT DOUBLE_DATE(dateVal) FROM T')

    def test_double_timestamp(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                DOUBLE_TIMESTAMP(x TIMESTAMP) RETURNS DOUBLE AS
                class DOUBLE_TIMESTAMP {
                    static double run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getDouble("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'getDouble cannot convert column'):
            rows = self.query('SELECT DOUBLE_TIMESTAMP(timestampVal) FROM T')

class JavaTypesString(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')
        self.query('CREATE TABLE T (intVal DECIMAL(9,0), longVal DECIMAL(18,0), bigdecimalVal DECIMAL(36,0), decimalVal DECIMAL(9,2), \
                    doubleVal DOUBLE, doubleIntVal DOUBLE, stringVal VARCHAR(100), booleanVal BOOLEAN, dateVal DATE, timestampVal TIMESTAMP)')
        self.query('''INSERT INTO T values (123456789, 123456789123456789, 123456789123456789123456789123456789, 1234567.12, \
                    123456789.123, 15.0, 'string#String!12345', true, '2014-05-21', '2014-05-21 15:13:30.123')''')

    def test_string_integer(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                STRING_INT(x SHORTINT) RETURNS VARCHAR(100) AS
                class STRING_INT {
                    static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getString("x");
                    }
                }
                '''))
        rows = self.query('SELECT STRING_INT(intVal) FROM T')
        self.assertRowsEqual([('123456789',)], rows)

    def test_string_long(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                STRING_LONG(x INTEGER) RETURNS VARCHAR(100) AS
                class STRING_LONG {
                    static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getString("x");
                    }
                }
                '''))
        rows = self.query('SELECT STRING_LONG(longVal) FROM T')
        self.assertRowsEqual([('123456789123456789',)], rows)

    def test_string_bigdecimal(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                STRING_BIGDEC(x BIGINT) RETURNS VARCHAR(100) AS
                class STRING_BIGDEC {
                    static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getString("x");
                    }
                }
                '''))
        rows = self.query('SELECT STRING_BIGDEC(bigdecimalVal) FROM T')
        self.assertRowsEqual([('123456789123456789123456789123456789',)], rows)

    def test_string_decimal(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                STRING_DECIMAL(x DECIMAL(9,2)) RETURNS VARCHAR(100) AS
                class STRING_DECIMAL {
                    static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getString("x");
                    }
                }
                '''))
        rows = self.query('SELECT STRING_DECIMAL(decimalVal) FROM T')
        self.assertRowsEqual([('1234567.12',)], rows)

    def test_string_double(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                STRING_DOUBLE(x DOUBLE) RETURNS VARCHAR(100) AS
                class STRING_DOUBLE {
                    static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getString("x");
                    }
                }
                '''))
        rows = self.query('SELECT STRING_DOUBLE(doubleVal) FROM T')
        self.assertRowsEqual([('1.23456789123E8',)], rows)

    def test_string_double_int(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                STRING_DOUBLE_INT(x DOUBLE) RETURNS VARCHAR(100) AS
                class STRING_DOUBLE_INT {
                    static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getString("x");
                    }
                }
                '''))
        rows = self.query('SELECT STRING_DOUBLE_INT(doubleIntVal) FROM T')
        self.assertRowsEqual([('15.0',)], rows)

    def test_string_string(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                STRING_STRING(x VARCHAR(100)) RETURNS VARCHAR(100) AS
                class STRING_STRING {
                    static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getString("x");
                    }
                }
                '''))
        rows = self.query('SELECT STRING_STRING(stringVal) FROM T')
        self.assertRowsEqual([('string#String!12345',)], rows)

    def test_string_boolean(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                STRING_BOOL(x BOOLEAN) RETURNS VARCHAR(100) AS
                class STRING_BOOL {
                    static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getString("x");
                    }
                }
                '''))
        rows = self.query('SELECT STRING_BOOL(booleanVal) FROM T')
        self.assertRowsEqual([('true',)], rows)

    def test_string_date(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                STRING_DATE(x DATE) RETURNS VARCHAR(100) AS
                class STRING_DATE {
                    static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getString("x");
                    }
                }
                '''))
        rows = self.query('SELECT STRING_DATE(dateVal) FROM T')
        self.assertRowsEqual([('2014-05-21',)], rows)

    def test_string_timestamp(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                STRING_TIMESTAMP(x TIMESTAMP) RETURNS VARCHAR(100) AS
                class STRING_TIMESTAMP {
                    static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getString("x");
                    }
                }
                '''))
        rows = self.query('SELECT STRING_TIMESTAMP(timestampVal) FROM T')
        self.assertRowsEqual([('2014-05-21 15:13:30.123',)], rows)

class JavaTypesBoolean(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')
        self.query('CREATE TABLE T (intVal DECIMAL(9,0), longVal DECIMAL(18,0), bigdecimalVal DECIMAL(36,0), decimalVal DECIMAL(9,2), \
                    doubleVal DOUBLE, doubleIntVal DOUBLE, stringVal VARCHAR(100), booleanVal BOOLEAN, dateVal DATE, timestampVal TIMESTAMP)')
        self.query('''INSERT INTO T values (123456789, 123456789123456789, 123456789123456789123456789123456789, 1234567.12, \
                    123456789.123, 15.0, 'string#String!12345', true, '2014-05-21', '2014-05-21 15:13:30.123')''')

    def test_boolean_integer(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                BOOLEAN_INT(x SHORTINT) RETURNS BOOLEAN AS
                class BOOLEAN_INT {
                    static boolean run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getBoolean("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'getBoolean cannot convert column'):
            rows = self.query('SELECT BOOLEAN_INT(intVal) FROM T')

    def test_boolean_long(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                BOOLEAN_LONG(x INTEGER) RETURNS BOOLEAN AS
                class BOOLEAN_LONG {
                    static boolean run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getBoolean("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'getBoolean cannot convert column'):
            rows = self.query('''SELECT TO_CHAR(BOOLEAN_LONG(longVal), '9.99999999999999EEEE') FROM T''')

    def test_boolean_bigdecimal(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                BOOLEAN_BIGDEC(x BIGINT) RETURNS BOOLEAN AS
                class BOOLEAN_BIGDEC {
                    static boolean run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getBoolean("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'getBoolean cannot convert column'):
            rows = self.query('''SELECT TO_CHAR(BOOLEAN_BIGDEC(bigdecimalVal), '9.99999999999999EEEE') FROM T''')

    def test_boolean_decimal(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                BOOLEAN_DECIMAL(x DECIMAL(9,2)) RETURNS BOOLEAN AS
                class BOOLEAN_DECIMAL {
                    static boolean run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getBoolean("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'getBoolean cannot convert column'):
            rows = self.query('SELECT BOOLEAN_DECIMAL(decimalVal) FROM T')

    def test_boolean_double(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                BOOLEAN_DOUBLE(x DOUBLE) RETURNS BOOLEAN AS
                class BOOLEAN_DOUBLE {
                    static boolean run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getBoolean("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'getBoolean cannot convert column'):
            rows = self.query('SELECT BOOLEAN_DOUBLE(doubleVal) FROM T')

    def test_boolean_double_int(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                BOOLEAN_DOUBLE_INT(x DOUBLE) RETURNS BOOLEAN AS
                class BOOLEAN_DOUBLE_INT {
                    static boolean run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getBoolean("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'getBoolean cannot convert column'):
            rows = self.query('SELECT BOOLEAN_DOUBLE_INT(doubleIntVal) FROM T')

    def test_boolean_string(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                BOOLEAN_STRING(x VARCHAR(100)) RETURNS BOOLEAN AS
                class BOOLEAN_STRING {
                    static boolean run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getBoolean("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'getBoolean cannot convert column'):
            rows = self.query('SELECT BOOLEAN_STRING(stringVal) FROM T')

    def test_boolean_boolean(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                BOOLEAN_BOOL(x BOOLEAN) RETURNS BOOLEAN AS
                class BOOLEAN_BOOL {
                    static boolean run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getBoolean("x");
                    }
                }
                '''))
        rows = self.query('''SELECT 'OK' FROM (SELECT BOOLEAN_BOOL(booleanVal) as val FROM T) WHERE val = TRUE''')
        self.assertRowsEqual([('OK',)], rows)

    def test_boolean_date(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                BOOLEAN_DATE(x DATE) RETURNS BOOLEAN AS
                class BOOLEAN_DATE {
                    static boolean run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getBoolean("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'getBoolean cannot convert column'):
            rows = self.query('SELECT BOOLEAN_DATE(dateVal) FROM T')

    def test_boolean_timestamp(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                BOOLEAN_TIMESTAMP(x TIMESTAMP) RETURNS BOOLEAN AS
                class BOOLEAN_TIMESTAMP {
                    static boolean run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getBoolean("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'getBoolean cannot convert column'):
            rows = self.query('SELECT BOOLEAN_TIMESTAMP(timestampVal) FROM T')

class JavaTypesDate(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')
        self.query('CREATE TABLE T (intVal DECIMAL(9,0), longVal DECIMAL(18,0), bigdecimalVal DECIMAL(36,0), decimalVal DECIMAL(9,2), \
                    doubleVal DOUBLE, doubleIntVal DOUBLE, stringVal VARCHAR(100), booleanVal BOOLEAN, dateVal DATE, timestampVal TIMESTAMP)')
        self.query('''INSERT INTO T values (123456789, 123456789123456789, 123456789123456789123456789123456789, 1234567.12, \
                    123456789.123, 15.0, 'string#String!12345', true, '2014-05-21', '2014-05-21 15:13:30.123')''')

    def test_date_integer(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                DATE_INT(x SHORTINT) RETURNS DATE AS
                import java.sql.Date;
                class DATE_INT {
                    static Date run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getDate("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'getDate cannot convert column'):
            rows = self.query('SELECT DATE_INT(intVal) FROM T')

    def test_date_long(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                DATE_LONG(x INTEGER) RETURNS DATE AS
                import java.sql.Date;
                class DATE_LONG {
                    static Date run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getDate("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'getDate cannot convert column'):
            rows = self.query('SELECT DATE_LONG(longVal) FROM T')

    def test_date_bigdecimal(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                DATE_BIGDEC(x BIGINT) RETURNS DATE AS
                import java.sql.Date;
                class DATE_BIGDEC {
                    static Date run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getDate("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'getDate cannot convert column'):
            rows = self.query('SELECT DATE_BIGDEC(bigdecimalVal) FROM T')

    def test_date_decimal(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                DATE_DECIMAL(x DECIMAL(9,2)) RETURNS DATE AS
                import java.sql.Date;
                class DATE_DECIMAL {
                    static Date run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getDate("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'getDate cannot convert column'):
            rows = self.query('SELECT DATE_DECIMAL(decimalVal) FROM T')

    def test_date_double(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                DATE_DOUBLE(x DOUBLE) RETURNS DATE AS
                import java.sql.Date;
                class DATE_DOUBLE {
                    static Date run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getDate("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'getDate cannot convert column'):
            rows = self.query('SELECT DATE_DOUBLE(doubleVal) FROM T')

    def test_date_double_int(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                DATE_DOUBLE_INT(x DOUBLE) RETURNS DATE AS
                import java.sql.Date;
                class DATE_DOUBLE_INT {
                    static Date run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getDate("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'getDate cannot convert column'):
            rows = self.query('SELECT DATE_DOUBLE_INT(doubleIntVal) FROM T')

    def test_date_string(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                DATE_STRING(x VARCHAR(100)) RETURNS DATE AS
                import java.sql.Date;
                class DATE_STRING {
                    static Date run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getDate("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'getDate cannot convert column'):
            rows = self.query('SELECT DATE_STRING(stringVal) FROM T')

    def test_date_boolean(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                DATE_BOOL(x BOOLEAN) RETURNS DATE AS
                import java.sql.Date;
                class DATE_BOOL {
                    static Date run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getDate("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'getDate cannot convert column'):
            rows = self.query('SELECT DATE_BOOL(booleanVal) FROM T')

    def test_date_date(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                DATE_DATE(x DATE) RETURNS DATE AS
                import java.sql.Date;
                class DATE_DATE {
                    static Date run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getDate("x");
                    }
                }
                '''))
        rows = self.query('SELECT TO_CHAR(DATE_DATE(dateVal)) FROM T')
        self.assertRowsEqual([('2014-05-21',)], rows)

    def test_date_timestamp(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                DATE_TIMESTAMP(x TIMESTAMP) RETURNS DATE AS
                import java.sql.Date;
                class DATE_TIMESTAMP {
                    static Date run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getDate("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'getDate cannot convert column'):
            rows = self.query('SELECT DATE_TIMESTAMP(timestampVal) FROM T')

class JavaTypesTimestamp(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')
        self.query('CREATE TABLE T (intVal DECIMAL(9,0), longVal DECIMAL(18,0), bigdecimalVal DECIMAL(36,0), decimalVal DECIMAL(9,2), \
                    doubleVal DOUBLE, doubleIntVal DOUBLE, stringVal VARCHAR(100), booleanVal BOOLEAN, dateVal DATE, timestampVal TIMESTAMP)')
        self.query('''INSERT INTO T values (123456789, 123456789123456789, 123456789123456789123456789123456789, 1234567.12, \
                    123456789.123, 15.0, 'string#String!12345', true, '2014-05-21', '2014-05-21 15:13:30.123')''')

    def test_timestamp_integer(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                TIMESTAMP_INT(x SHORTINT) RETURNS TIMESTAMP AS
                import java.sql.Timestamp;
                class TIMESTAMP_INT {
                    static Timestamp run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getTimestamp("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'getTimestamp cannot convert column'):
            rows = self.query('SELECT TIMESTAMP_INT(intVal) FROM T')

    def test_timestamp_long(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                TIMESTAMP_LONG(x INTEGER) RETURNS TIMESTAMP AS
                import java.sql.Timestamp;
                class TIMESTAMP_LONG {
                    static Timestamp run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getTimestamp("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'getTimestamp cannot convert column'):
            rows = self.query('SELECT TIMESTAMP_LONG(longVal) FROM T')

    def test_timestamp_bigdecimal(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                TIMESTAMP_BIGDEC(x BIGINT) RETURNS TIMESTAMP AS
                import java.sql.Timestamp;
                class TIMESTAMP_BIGDEC {
                    static Timestamp run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getTimestamp("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'getTimestamp cannot convert column'):
            rows = self.query('SELECT TIMESTAMP_BIGDEC(bigdecimalVal) FROM T')

    def test_timestamp_decimal(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                TIMESTAMP_DECIMAL(x DECIMAL(9,2)) RETURNS TIMESTAMP AS
                import java.sql.Timestamp;
                class TIMESTAMP_DECIMAL {
                    static Timestamp run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getTimestamp("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'getTimestamp cannot convert column'):
            rows = self.query('SELECT TIMESTAMP_DECIMAL(decimalVal) FROM T')

    def test_timestamp_double(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                TIMESTAMP_DOUBLE(x DOUBLE) RETURNS TIMESTAMP AS
                import java.sql.Timestamp;
                class TIMESTAMP_DOUBLE {
                    static Timestamp run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getTimestamp("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'getTimestamp cannot convert column'):
            rows = self.query('SELECT TIMESTAMP_DOUBLE(doubleVal) FROM T')

    def test_timestamp_double_int(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                TIMESTAMP_DOUBLE_INT(x DOUBLE) RETURNS TIMESTAMP AS
                import java.sql.Timestamp;
                class TIMESTAMP_DOUBLE_INT {
                    static Timestamp run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getTimestamp("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'getTimestamp cannot convert column'):
            rows = self.query('SELECT TIMESTAMP_DOUBLE_INT(doubleIntVal) FROM T')

    def test_timestamp_string(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                TIMESTAMP_STRING(x VARCHAR(100)) RETURNS TIMESTAMP AS
                import java.sql.Timestamp;
                class TIMESTAMP_STRING {
                    static Timestamp run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getTimestamp("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'getTimestamp cannot convert column'):
            rows = self.query('SELECT TIMESTAMP_STRING(stringVal) FROM T')

    def test_timestamp_boolean(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                TIMESTAMP_BOOL(x BOOLEAN) RETURNS TIMESTAMP AS
                import java.sql.Timestamp;
                class TIMESTAMP_BOOL {
                    static Timestamp run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getTimestamp("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'getTimestamp cannot convert column'):
            rows = self.query('SELECT TIMESTAMP_BOOL(booleanVal) FROM T')

    def test_timestamp_date(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                TIMESTAMP_DATE(x DATE) RETURNS TIMESTAMP AS
                import java.sql.Timestamp;
                class TIMESTAMP_DATE {
                    static Timestamp run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getTimestamp("x");
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'getTimestamp cannot convert column'):
            rows = self.query('SELECT TIMESTAMP_DATE(dateVal) FROM T')

    def test_timestamp_timestamp(self):
        self.query(udf.fixindent('''
                CREATE java SET SCRIPT
                TIMESTAMP_TIMESTAMP(x TIMESTAMP) RETURNS TIMESTAMP AS
                import java.sql.Timestamp;
                class TIMESTAMP_TIMESTAMP {
                    static Timestamp run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return ctx.getTimestamp("x");
                    }
                }
                '''))
        rows = self.query('SELECT TO_CHAR(TIMESTAMP_TIMESTAMP(timestampVal)) FROM T')
        self.assertRowsEqual([('2014-05-21 15:13:30.123000',)], rows)

if __name__ == '__main__':
    udf.main()

# vim: ts=4:sts=4:sw=4:et:fdm=indent
