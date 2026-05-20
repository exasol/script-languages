#!/usr/bin/env python3

from exasol_python_test_framework import udf
from exasol_python_test_framework.udf import requires


class BasicTest(udf.TestCase):
    
    def setUp(self):
        self.query('DROP SCHEMA FN1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN1')
        self.query('OPEN SCHEMA FN1')
        
        # Create all UDFs needed for BasicTest
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
            CREATE PYTHON3 SCALAR SCRIPT basic_emit_two_ints()
            EMITS (i INTEGER, j INTEGER) AS
            def run(ctx):
                ctx.emit(1,2)
            /
        '''))
        
        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT basic_nth_partial_sum(n INTEGER)
            RETURNS INTEGER AS
            def run(ctx):
                if ctx.n is not None:
                    return ctx.n * (ctx.n + 1) / 2
                return 0
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
            CREATE PYTHON3 SCALAR SCRIPT basic_emit_several_groups(a INTEGER, b INTEGER)
            EMITS (i INTEGER, j VARCHAR(40)) AS
            def run(ctx):
                for n in range(ctx.a):
                    for i in range(ctx.b):
                        ctx.emit(i, repr((exa.meta.vm_id, exa.meta.node_count, exa.meta.node_id)))
            /
        '''))
        
        self.query(udf.fixindent('''
            CREATE PYTHON3 SET SCRIPT basic_test_reset(i INTEGER, j VARCHAR(40))
            EMITS (k INTEGER) AS
            def run(ctx):
                ctx.emit(ctx.i)
                ctx.next()
                ctx.emit(ctx.i)
                ctx.reset()
                ctx.emit(ctx.i)
                ctx.next()
                ctx.emit(ctx.i)
            /
        '''))
        
        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT performance_map_characters(text VARCHAR(1000))
            EMITS (w CHAR(1), c INTEGER) AS
            def run(ctx):
                if ctx.text is not None:
                    for c in ctx.text:
                        ctx.emit(c, 1)
            /
        '''))
        
        self.query(udf.fixindent('''
            CREATE PYTHON3 SET SCRIPT performance_reduce_characters(w CHAR(1), c INTEGER)
            EMITS (w CHAR(1), c INTEGER) AS
            def run(ctx):
                c = 0
                w = ctx.w
                if w is not None:
                    while True:
                        c += 1
                        if not ctx.next(): break
                    ctx.emit(w, c)
            /
        '''))
        
        self.query(udf.fixindent('''
            CREATE PYTHON3 SET SCRIPT set_returns_has_empty_input(a double) 
            RETURNS boolean AS
            def run(ctx):
                return bool(ctx.x is None)
            /
        '''))
        
        self.query(udf.fixindent('''
            CREATE PYTHON3 SET SCRIPT set_emits_has_empty_input(a double) 
            EMITS (x double, y varchar(10)) AS
            def run(ctx):
                if ctx.x is None:
                    ctx.emit(1,'1')
                else:
                    ctx.emit(2,'2')
            /
        '''))

    def test_basic_scalar_emits(self):
        rows = self.query('''
            SELECT fn1.basic_range(3)
            FROM DUAL
            ''')
        self.assertRowsEqual([(x,) for x in range(3)], sorted(rows))
    
    def test_basic_set_returns(self):
        rows = self.query('''
            SELECT fn1.basic_sum(3)
            FROM DUAL
            ''')
        self.assertRowsEqual([(3,)], rows)

    def test_emit_two_ints(self):
        rows = self.query('''
            SELECT fn1.basic_emit_two_ints()
            FROM DUAL''')
        self.assertRowsEqual([(1, 2)], rows)

    def test_simple_combination(self):
        rows = self.query('''
            SELECT fn1.basic_sum(psum)
            FROM (
                SELECT fn1.basic_nth_partial_sum(n) AS PSUM
                FROM (
                    SELECT fn1.basic_range(10)
                    FROM DUAL
                )
            )''')
        self.assertRowsEqual([(165,)], rows)

    def test_simple_combination_grouping(self):
        rows = self.query('''
            SELECT fn1.BASIC_SUM_GRP(psum)
            FROM (
                SELECT MOD(N, 3) AS n,
                    fn1.basic_nth_partial_sum(n) AS psum
                FROM (
                    SELECT fn1.basic_range(10)
                    FROM DUAL
                )
            )
            GROUP BY n
            ORDER BY 1''')
        self.assertRowsEqual([(39.0,), (54.0,), (72.0,)], rows)

    def test_reset(self):
        rows = self.query('''
            SELECT fn1.basic_test_reset(i, j)
            FROM (SELECT fn1.basic_emit_several_groups(16, 8) FROM DUAL)
            GROUP BY i
            ORDER BY 1''')
        self.assertRowsEqual([(0.0,), (0.0,), (0.0,), (0.0,), (1.0,), (1.0,), (1.0,), (1.0,), (2.0,)], rows[:9])

    def test_order_by_clause(self):
        rows = self.query('''
            SELECT fn1.performance_reduce_characters(w, c)
            FROM (
               SELECT fn1.performance_map_characters('hello hello hello abc')
               FROM DUAL
            )
            GROUP BY w
            ORDER BY c DESC''')

        unsorted_list = [tuple(x) for x in rows]
        sorted_list = sorted(unsorted_list, key=lambda x: x[1], reverse=True)
        #for x in zip(unsorted_list, sorted_list):
        #    print x

        self.assertEqual(sorted_list, unsorted_list)


class SetWithEmptyInput(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN1')
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')
        self.query('OPEN SCHEMA FN1')
        self.query('CREATE TABLE FN2.empty_table(c int)')
        
        # Create UDFs needed for SetWithEmptyInput tests
        self.query(udf.fixindent('''
            CREATE PYTHON3 SET SCRIPT set_returns_has_empty_input(a double) 
            RETURNS boolean AS
            def run(ctx):
                return bool(ctx.x is None)
            /
        '''))
        
        self.query(udf.fixindent('''
            CREATE PYTHON3 SET SCRIPT set_emits_has_empty_input(a double) 
            EMITS (x double, y varchar(10)) AS
            def run(ctx):
                if ctx.x is None:
                    ctx.emit(1,'1')
                else:
                    ctx.emit(2,'2')
            /
        '''))

    def test_set_returns_has_empty_input_group_by(self):
        self.query("""select FN1.set_returns_has_empty_input(c) from FN2.empty_table group by 'X'""")
        self.assertEqual(0, self.rowcount())

    def test_set_returns_has_empty_input_no_group_by(self):
        rows = self.query('''select FN1.set_returns_has_empty_input(c) from FN2.empty_table''')
        self.assertRowsEqual([(None,)], rows)


    def test_set_emits_has_empty_input_group_by(self):
        self.query("""select FN1.set_emits_has_empty_input(c) from FN2.empty_table group by 'X'""")
        self.assertEqual(0, self.rowcount())

    def test_set_emits_has_empty_input_no_group_by(self):
        rows = self.query('''select FN1.set_emits_has_empty_input(c) from FN2.empty_table''')
        self.assertRowsEqual([(None,None)], rows)


if __name__ == '__main__':
    udf.main()
