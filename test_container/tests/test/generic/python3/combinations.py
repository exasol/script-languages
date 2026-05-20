#!/usr/bin/env python3

from exasol_python_test_framework import udf
from exasol_python_test_framework.udf import useData


class _Python3UdfSetup(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN1')
        self.query('OPEN SCHEMA FN1')

        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT basic_range(n INTEGER)
            EMITS (n INTEGER) AS
            def run(ctx):
                if ctx.n is not None:
                    for i in range(ctx.n):
                        ctx.emit(i)
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SET SCRIPT basic_sum(x INTEGER)
            RETURNS INTEGER AS
            def run(ctx):
                s = 0
                while True:
                    if ctx.x is not None:
                        s += ctx.x
                    if not ctx.next():
                        break
                return s
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SET SCRIPT basic_sum_grp(x INTEGER)
            EMITS (s INTEGER) AS
            def run(ctx):
                s = 0
                while True:
                    if ctx.x is not None:
                        s += ctx.x
                    if not ctx.next():
                        break
                ctx.emit(s)
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT SCALAR_EMITS(x double, y double)
            EMITS (x double, y double) AS
            def run(ctx):
                for i in range(int(ctx.x), int(ctx.y+1)):
                    ctx.emit(float(i), float(i * i))
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT SCALAR_RETURNS(x double, y double)
            RETURNS double AS
            def run(ctx):
                return ctx.x + ctx.y
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SET SCRIPT SET_EMITS(x double, y double)
            EMITS (x double, y double) AS
            def run(ctx):
                while True:
                    ctx.emit(ctx.y, ctx.x)
                    if not ctx.next(): break
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SET SCRIPT SET_RETURNS(x double, y double)
            RETURNS double AS
            def run(ctx):
                acc = 0.0
                while True:
                    acc = acc + ctx.x + ctx.y
                    if not ctx.next(): break
                return acc
            /
        '''))
        
        self.query('DROP SCHEMA combinations CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA combinations')
        self.query('CREATE TABLE combinations.small(x DOUBLE, y DOUBLE)')
        self.query('INSERT INTO combinations.small VALUES (0.1, 0.2), (0.2, 0.1)')

class Test(_Python3UdfSetup):
    pass


class Combinations_1_ary(Test):
    def test_set_returns(self):
        rows = self.query('''
                SELECT fn1.SET_RETURNS(x,y)
                FROM combinations.small''')
        self.assertEqual(round(0.6 / 2), round(rows[0][0] / 2))


    def test_scalar_returns(self):
        rows = self.query('''
                SELECT round(fn1.scalar_returns(x,y) / 2)
                FROM combinations.small''')
        self.assertRowsEqual([(round(0.3 / 2),), (round(0.3 / 2),)], rows)

    def test_scalar_emits(self):
        rows = self.query('''
                SELECT fn1.scalar_emits(x * 10 ,y * 10)
                FROM combinations.small''')
        self.assertRowsEqual([(1, 1,), (2, 4,)], rows)

    def test_set_emits(self):
        rows = self.query('''
                SELECT fn1.set_emits(x * 10 ,y * 10)
                FROM combinations.small''')
        self.assertRowsEqual([(2.0, 1.0,), (1.0, 2.0,)], rows)

    def test_two_scalar_returns(self):
        rows = self.query('''
                SELECT
                    fn1.scalar_returns(fn1.scalar_returns(x * 10 ,y * 10),
                    fn1.scalar_returns(y * 10 ,x * 10))
                FROM combinations.small''')
        self.assertRowsEqual([(6,), (6,)], rows)


class Combinations_2_ary_scalar_returns(Test):
    def test_scalar_returns_scalar_emits(self):
        rows = self.query('''
                SELECT fn1.scalar_returns(x * 10 ,y * 10 )
                FROM (
                    SELECT fn1.scalar_emits(x * 10 ,y * 10)
                    FROM combinations.small
                )''')
        self.assertRowsEqual([(20,), (60,)], rows)

    def test_scalar_returns_set_returns_inline(self):
        rows = self.query('''
                SELECT
                    fn1.scalar_returns(
                        fn1.set_returns(x * 10, y * 10),
                        fn1.set_returns(x * 10, y * 10)
                    )
                FROM combinations.small''')
        self.assertRowsEqual([(12,)], rows)

    def test_scalar_returns_set_returns_1(self):
        rows = self.query('''
                SELECT fn1.scalar_returns(a, 5)
                FROM (
                    SELECT fn1.set_returns(x * 10, y * 10) AS a
                    FROM combinations.SMALL
                )
                ''')
        self.assertRowsEqual([(11,)], rows)

    def test_scalar_returns_set_returns_2(self):
        rows = self.query('''
                SELECT fn1.scalar_returns(aa.a, bb.a)
                FROM (
                    SELECT fn1.set_returns(x * 10, y * 20) AS a
                    FROM combinations.SMALL
                ) AS aa,
                (
                    SELECT fn1.set_returns(x * 20, y * 10) AS a
                    FROM combinations.SMALL
                ) AS bb
                ''')
        self.assertRowsEqual([(18,)], rows)

    def test_scalar_returns_set_emits_1(self):
        rows = self.query('''
                SELECT fn1.scalar_returns(x * 10, y * 10)
                FROM (
                    SELECT fn1.set_emits(x * 10, y * 10)
                    FROM combinations.small
                )''')
        self.assertRowsEqual([(30,), (30,)], rows)

    def test_scalar_returns_set_emits_2(self):
        rows = self.query('''
                SELECT fn1.scalar_returns(aa.x * 10, bb.x * 10)
                FROM (
                    SELECT fn1.set_emits(x * 10, y * 10)
                    FROM combinations.small
                ) AS aa,
                (
                    SELECT fn1.set_emits(x * 10, y * 10)
                    FROM combinations.small
                ) AS bb
                WHERE aa.x = bb.y and aa.y = bb.x
                ''')
        self.assertRowsEqual([(30,), (30,)], rows)


class Combinations_2_ary_scalar_emits(Test):
    def test_scalar_emits_scalar_returns_inline(self):
        rows = self.query('''
                SELECT
                    fn1.scalar_emits(
                        fn1.scalar_returns(x * 10, y * 10),
                        fn1.scalar_returns(y * 10, x * 10)
                    )
                FROM combinations.small''')
        self.assertRowsEqual([(3, 9,), (3, 9,)], rows)

    def test_scalar_emits_scalar_returns(self):
        rows = self.query('''
                SELECT fn1.scalar_emits(a, b)
                FROM (
                    SELECT 
                        fn1.scalar_returns(x * 10 ,y * 10) AS A,
                        fn1.scalar_returns(y * 10 ,x * 10) AS B
                    FROM combinations.small
                )
                ''')
        self.assertRowsEqual([(3, 9,), (3, 9,)], rows)

    def test_scalar_emits_scalar_emits(self):
        rows = self.query('''
                SELECT fn1.scalar_emits(x * 10,y * 10)
                FROM (
                    SELECT fn1.scalar_emits(x * 10,y * 10)
                    FROM combinations.small
                )
                ORDER by x,y''')
        r = [(10.0, 100.0)]
        r.extend([(i, i * i) for i in range(20, 41)])
        self.assertNotEqual(r, rows)

    def test_scalar_emits_set_returns_inline(self):
        with self.assertRaisesRegex(Exception, 'encapsulated set function'):
            self.query('''
                    SELECT
                        fn1.scalar_emits(
                            fn1.set_returns(x * 10, y * 10),
                            fn1.set_returns(x * 10, y * 10)
                        )
                    FROM combinations.small''')

    def test_scalar_emits_set_returns(self):
        rows = self.query('''
                SELECT fn1.scalar_emits(a, b)
                FROM (
                    SELECT fn1.set_returns(x * 10, y * 10) AS a
                    FROM combinations.small
                ),
                (
                    SELECT fn1.set_returns(x * 10, y *10) AS b
                    FROM combinations.small
                )
                ''')
        self.assertRowsEqual([(6, 36)], rows)

    def test_scalar_emits_set_emits(self):
        rows = self.query('''
            SELECT fn1.scalar_emits(x * 10, y * 10)
            FROM (
                SELECT fn1.set_emits(x * 10, y * 10)
                FROM combinations.small
            )''')
        r = ([(i, i * i) for i in range(10, 21)])
        self.assertRowsEqual(r, rows)


class Combinations_2_ary_set_returns(Test):
    def test_set_returns_scalar_returns(self):
        rows = self.query('''
            SELECT
                fn1.set_returns(
                    fn1.scalar_returns(x * 10, y * 10),
                    fn1.scalar_returns(y * 10, x * 10)
                )
            FROM combinations.small''')
        self.assertRowsEqual([(12,)], rows)

    def test_set_returns_scalar_emits(self):
        rows = self.query('''
            SELECT fn1.set_returns(x*10, y*10)
            FROM (
                SELECT fn1.scalar_emits(x*10, y*10)
                FROM combinations.small
            )''')
        self.assertRowsEqual([(80,)], rows)

    def test_set_returns_set_returns_inline(self):
        with self.assertRaisesRegex(Exception, 'encapsulated set function'):
            self.query('''
                SELECT  
                    fn1.set_returns(
                        fn1.set_returns(x*10, y*10),
                        fn1.set_returns(x*10, y*10)
                    )
                FROM combinations.small''')

    def test_set_returns_set_returns(self):
        rows = self.query('''
                SELECT fn1.set_returns(a, b)
                FROM(
                    SELECT fn1.set_returns(x*20, y*30) AS a
                    FROM combinations.small
                ),
                (
                    SELECT fn1.set_returns(x*50, y*70) AS b
                    FROM combinations.small
                )''')
        self.assertRowsEqual([(51,)], rows)

    def test_set_returns_set_emits(self):
        rows = self.query('''
            SELECT fn1.set_returns(x*10, y*10)
            FROM (
                SELECT fn1.set_emits(x*10, y*10)
                FROM combinations.small
            )''')
        self.assertRowsEqual([(60,)], rows)


class Combinations_2_ary_set_emits(Test):
    def test_set_emits_scalar_returns(self):
        rows = self.query('''
            SELECT
                fn1.set_emits(
                    fn1.scalar_returns(x*10, y*10),
                    fn1.scalar_returns(y*10, x*10)
                )
            FROM combinations.small''')
        self.assertRowsEqual([(3, 3,), (3, 3,)], rows)

    def test_set_emits_scalar_emits(self):
        rows = self.query('''
                SELECT fn1.set_emits(x*10, y*10)
                FROM (
                    SELECT fn1.scalar_emits(x*10, 10*y)
                    FROM combinations.small
                )
                ORDER BY x, y;''')
        self.assertRowsEqual([(10, 10,), (40, 20,)], rows)

    def test_set_emits_set_returns_inline(self):
        with self.assertRaisesRegex(Exception, 'encapsulated set function'):
            self.query('''
                SELECT
                    fn1.set_emits(
                        fn1.set_returns(x*10, 10*y),
                        fn1.set_returns(10*x, y*10)
                    )
                FROM combinations.small''')

    def test_set_emits_set_returns(self):
        rows = self.query('''
                SELECT fn1.set_emits(a, b)
                FROM (
                    SELECT fn1.set_returns(x*20, 30*y) AS a
                    FROM combinations.small
                ),
                (
                    SELECT fn1.set_returns(50*x, y*70) AS b
                    FROM combinations.small
                )
                ''')
        self.assertRowsEqual([(36, 15,)], rows)

    def test_set_emits_set_emits(self):
        rows = self.query('''
                SELECT fn1.set_emits(x * 10 , 10 * y)
                FROM (
                    SELECT fn1.set_emits(x * 10, y *10)
                    FROM combinations.small
                )
                ORDER BY x, y;''')
        self.assertRowsEqual([(10, 20,), (20, 10,)], rows)


class Combinations_3_ary(Test):
    def test_set_returns_set_emits_scalar_emits(self):
        rows = self.query('''
            SELECT fn1.basic_sum(s)
            FROM (
                SELECT fn1.basic_sum_grp(n)
                FROM(
                    SELECT fn1.basic_range(10)
                    FROM DUAL
                )
            )''')
        self.assertRowsEqual([(45,)], rows)

    def test_set_returns_scalar_emits_scalar_emits(self):
        rows = self.query('''
            SELECT fn1.basic_sum(x)
            FROM (
                SELECT fn1.scalar_emits(n, n+2)
                FROM(
                    SELECT fn1.basic_range(10)
                    FROM DUAL
                )
            )''')
        self.assertRowsEqual([(165,)], rows)

    def test_set_returns_scalar_returns_scalar_emits(self):
        rows = self.query('''
            SELECT fn1.basic_sum(x)
            FROM (
                SELECT fn1.scalar_returns(n, 2) AS x
                FROM(
                    SELECT fn1.basic_range(10)
                    FROM DUAL
                )
            )''')
        self.assertRowsEqual([(65,)], rows)


class Combinations_n_ary(Test):
    @staticmethod
    def partial_sum(n, degree):
        def basic_range(n, d):
            if d == 0:
                return list(range(n))
            else:
                return sum([list(range(x)) for x in basic_range(n + 1, d - 1)], [])

        return len(basic_range(n, degree))

    @useData((i,) for i in range(10))
    def test_n_scalar_emits(self, n):

        self.query(
            'SELECT fn1.basic_range(n+1) FROM (\n' * n +
            'SELECT fn1.basic_range(5) FROM DUAL\n' +
            ')' * n)
        self.assertEqual(self.partial_sum(5, n), self.rowcount())

    @useData((i,) for i in range(10))
    def test_set_returns_n_scalar_emits(self, n):

        rows = self.query(
            'SELECT max(n) FROM (' +
            'SELECT fn1.basic_range(n+1) FROM (\n' * n +
            'SELECT fn1.basic_range(5) FROM DUAL\n' +
            ')' * (n + 1))
        self.assertEqual(4, rows[0][0])


if __name__ == '__main__':
    udf.main()
