#!/usr/bin/env python3

from exasol_python_test_framework import udf


class _Python3UdfSetup(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN1')
        self.query('OPEN SCHEMA FN1')
        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT add_three_doubles(x DOUBLE, y DOUBLE, z DOUBLE)
            RETURNS DOUBLE AS
            def run(ctx):
                if ctx.x is not None and ctx.y is not None and ctx.z is not None:
                    return ctx.x + ctx.y + ctx.z;
                return None
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT add_two_doubles(x DOUBLE, y DOUBLE)
            RETURNS DOUBLE AS
            def run(ctx):
                if ctx.x is not None and ctx.y is not None:
                    return ctx.x + ctx.y;
                return None
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT double_mult("x" double, "y" double)
            RETURNS double AS
            def run(ctx):
              if ctx.x is None or ctx.y is None:
                return None
              else:
                return ctx.x * ctx.y
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT pi()
            RETURNS double AS
            import math
            
            def run(ctx):
              return math.pi
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT split_integer_into_digits("x" INTEGER)
            EMITS (y INTEGER) AS
            import math
            def run(ctx):
                if ctx.x is not None:
                    y = abs(ctx.x)
                    while y > 0:
                        ctx.emit(y % 10)
                        y //= 10
            /
        '''))

class Test(_Python3UdfSetup):
    def test_pi(self):
        rows = self.query('''
            SELECT fn1.pi()
                FROM dual''')
        result = rows[0][0]
        self.assertAlmostEqual(3.1415926535, result)

    def test_select(self):
        rows = self.query('''
            SELECT DISTINCT
	            FN1.double_mult(float1, float2) = float1 * float2 AS a
            FROM test.enginetablebig1
            ORDER BY a
            ''')
        self.assertRowsEqual([(True,), (None,)], rows)

    def test_select_into(self):
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')
        self.query('CREATE TABLE FN2.t(diff double)')
        self.query('''
            INSERT INTO FN2.t
                SELECT
                    fn1.double_mult(float1, float2) - float1 * float2 AS a
                    FROM test.enginetable
            ''')
        self.query('''
            SELECT DISTINCT diff
                FROM FN2.t
                WHERE diff != 0 AND diff IS NOT NULL
            ''')
        self.assertEqual(0, self.rowcount())

    def test_subselect(self):
        rows = self.query('''
            SELECT i, a
            FROM (
                SELECT int_index AS i,
                    (fn1.double_mult(float1, float2) - float1 * float2) AS a
                FROM test.enginetable)
            WHERE a IS NOT NULL
            ORDER BY a
            LIMIT 20''')
        for row in rows:
            rows2 = self.query('''
                SELECT
                    float1,
                    float2,
                    fn1.double_mult(float1, float2) - float1 * float2 AS a
                FROM test.enginetable
                WHERE int_index = ?''',
                               row.I)
            self.assertEqual(row.A, rows2[0].A)

    def test_udf_with_two_doubles(self):
        rows = self.query('''
            SELECT
                fn1.add_two_doubles(NULL, NULL) IS NULL,
                fn1.add_two_doubles(NULL,    0) IS NULL,
                fn1.add_two_doubles(   0, NULL) IS NULL,
                fn1.add_two_doubles(0,0) = 0,
                fn1.add_two_doubles(1,0) = 1,
                fn1.add_two_doubles(0,2) = 2,
                fn1.add_two_doubles(2,3) = 5
            FROM DUAL''')
        self.assertRowsEqual([tuple([True] * 7)], rows)

    def test_udf_with_three_doubles_part1(self):
        rows = self.query('''
            SELECT
                fn1.add_three_doubles(NULL, NULL, NULL) is NULL,
                fn1.add_three_doubles(NULL, NULL,    0) is NULL,
                fn1.add_three_doubles(NULL,    0, NULL) is NULL,
                fn1.add_three_doubles(   0, NULL, NULL) is NULL,
                fn1.add_three_doubles(NULL,    0,    0) is NULL
            FROM DUAL''')
        self.assertRowsEqual([tuple([True] * 5)], rows)

    def test_udf_with_three_doubles_part2(self):
        rows = self.query('''
            SELECT
                fn1.add_three_doubles(   0, NULL,    0) is NULL,
                fn1.add_three_doubles(   0,    0, NULL) is NULL,
                fn1.add_three_doubles(0, 0, 0) = 0,
                fn1.add_three_doubles(1, 0, 0) = 1,
                fn1.add_three_doubles(0, 2, 0) = 2
            FROM DUAL''')
        self.assertRowsEqual([tuple([True] * 5)], rows)

    def test_udf_with_three_doubles_part3(self):
        rows = self.query('''
            SELECT
                fn1.add_three_doubles(0, 0, 3) = 3,
                fn1.add_three_doubles(1, 2, 0) = 3,
                fn1.add_three_doubles(1, 0, 3) = 4,
                fn1.add_three_doubles(0, 2, 3) = 5,
                fn1.add_three_doubles(1, 2, 3) = 6
            FROM DUAL''')
        self.assertRowsEqual([tuple([True] * 5)], rows)

    def test_right_number_of_emitted_rows(self):
        rows = self.query('''
            SELECT fn1.split_integer_into_digits(123)
            FROM DUAL''')
        self.assertRowsEqual([(3,), (2,), (1,)], rows)


if __name__ == '__main__':
    udf.main()
