#!/usr/bin/env python3

from exasol_python_test_framework import udf
import pathlib
import re


class BasicTest(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN1')
        self.query('OPEN SCHEMA FN1')
        
        # Load R scripts from SQL files
        lang_path = pathlib.Path(__file__).parent.parent.parent.parent / 'lang' / 'r'
        for sql_filename in ['basic.sql', 'performance.sql']:
            sql_file = lang_path / sql_filename
            with open(sql_file, 'r') as f:
                sql_content = f.read()
            
            # Execute each CREATE SCRIPT statement
            statements = re.split(r'^\s*/\s*$', sql_content, flags=re.MULTILINE)
            for stmt in statements:
                stmt = stmt.strip()
                if stmt and 'CREATE' in stmt.upper():
                    self.query(stmt)

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
        self.query('OPEN SCHEMA FN1')
        
        # Load R scripts from SQL file
        sql_file = pathlib.Path(__file__).parent.parent.parent.parent / 'lang' / 'r' / 'basic.sql'
        with open(sql_file, 'r') as f:
            sql_content = f.read()
        
        # Execute each CREATE SCRIPT statement
        statements = re.split(r'^\s*/\s*$', sql_content, flags=re.MULTILINE)
        for stmt in statements:
            stmt = stmt.strip()
            if stmt and 'CREATE' in stmt.upper():
                self.query(stmt)
        
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')
        self.query('CREATE TABLE FN2.empty_table(c int)')

    def test_set_returns_has_empty_input_group_by(self):
        self.query("""select FN1.set_returns_has_empty_input(c) from empty_table group by 'X'""")
        self.assertEqual(0, self.rowcount())

    def test_set_returns_has_empty_input_no_group_by(self):
        rows = self.query('''select FN1.set_returns_has_empty_input(c) from empty_table''')
        self.assertRowsEqual([(None,)], rows)


    def test_set_emits_has_empty_input_group_by(self):
        self.query("""select FN1.set_emits_has_empty_input(c) from empty_table group by 'X'""")
        self.assertEqual(0, self.rowcount())

    def test_set_emits_has_empty_input_no_group_by(self):
        rows = self.query('''select FN1.set_emits_has_empty_input(c) from empty_table''')
        self.assertRowsEqual([(None,None)], rows)


if __name__ == '__main__':
    udf.main()
