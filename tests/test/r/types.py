#!/usr/bin/env python2.7

import os
import sys

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf

class RTypes(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')

    def test_do_not_convert_string_to_double(self):
        self.query('''
                CREATE r SCALAR SCRIPT
                wrong_type() RETURNS DOUBLE AS
                run <- function(ctx) {
                    "1.5"
                }
                ''')
        with self.assertRaisesRegex(Exception, r'Value for column RETURN is not of type double'):
            self.query('''SELECT wrong_type() FROM DUAL''')

    def test_convert_int_to_double(self):
        self.query('''
                CREATE r SCALAR SCRIPT
                return_int() RETURNS DOUBLE AS

                run <-function(ctx) {
                    as.integer(4)
                }
                ''')
        self.query('SELECT return_int() FROM DUAL')

    def test_raises_with_incompatible_type(self):
        self.query('''
                CREATE r SCALAR SCRIPT
                wrong_type() RETURNS DOUBLE AS
                run <- function(ctx) {
                    "one point five"
                }
                ''')
        with self.assertRaisesRegex(Exception, r'Value for column RETURN is not of type double'):
            self.query('''SELECT wrong_type() FROM DUAL''')

    def test_scalar_with_vector_and_ints(self):
        self.query('''
                CREATE r SCALAR SCRIPT
                vector_and_ints(a SMALLINT, b INT, c BIGINT, d DECIMAL(11,0))
                RETURNS DECIMAL(11,0) AS
                run <- function(ctx) {
                    ctx$next_row(NA)
                    ctx$emit(ctx$a + ctx$b + ctx$c + ctx$d)
                    NULL
                }
                ''')
        self.query('''CREATE TABLE vector_and_ints_table1(a SMALLINT, b INT, c BIGINT, d DECIMAL(11,0))''')
        self.query('''INSERT INTO vector_and_ints_table1 VALUES (1, 1, 1, 1)''')
        for i in range(17):
            self.query('''
                INSERT INTO vector_and_ints_table1 SELECT CAST(ROWNUM AS SMALLINT) AS a,
                                                          CAST(ROWNUM AS INT) AS b,
                                                          CAST(ROWNUM AS BIGINT) AS c,
                                                          CAST(ROWNUM AS DECIMAL(11,0)) AS d
                                                   FROM vector_and_ints_table1
                ''')
        self.query('''COMMIT''')
        rows = self.query('''SELECT DISTINCT vector_and_ints(a, b, c, d) = a + b + c + d FROM vector_and_ints_table1''')
        self.assertRowsEqual([(True,)], rows)

    def test_set_with_vector_and_ints(self):
        self.query('''
                CREATE r SET SCRIPT
                vector_and_ints(a SMALLINT, b INT, c BIGINT, d DECIMAL(11,0), e DECIMAL(11,0))
                EMITS (f DECIMAL(11,0), g DECIMAL(11,0)) AS
                run <- function(ctx) {
                    ctx$next_row(NA)                    
                    ctx$emit(ctx$a + ctx$b + ctx$c + ctx$d, ctx$e)
                }
                ''')
        self.query('''CREATE TABLE vector_and_ints_table(a SMALLINT, b INT, c BIGINT, d DECIMAL(11,0))''')
        self.query('''INSERT INTO vector_and_ints_table VALUES (1, 1, 1, 1)''')
        for i in range(23):
            self.query('''
                INSERT INTO vector_and_ints_table SELECT CAST(ROWNUM AS SMALLINT) AS a,
                                                         CAST(ROWNUM AS INT) AS b,
                                                         CAST(ROWNUM AS BIGINT) AS c,
                                                         CAST(ROWNUM AS DECIMAL(11,0)) AS d
                                                  FROM vector_and_ints_table
                ''')
        self.query('''COMMIT''')
        rows = self.query('''SELECT DISTINCT f = g FROM (SELECT vector_and_ints(a, b, c, d, a + b + c + d) FROM vector_and_ints_table GROUP BY a MOD 2)''')
        self.assertRowsEqual([(True,)], rows)

if __name__ == '__main__':
    udf.main()

# vim: ts=4:sts=4:sw=4:et:fdm=indent

