#!/usr/bin/env python3

from exasol_python_test_framework import udf
from exasol_python_test_framework.udf import requires


class BasicTest(udf.TestCase):

    @requires('BASIC_RANGE')
    def test_basic_scalar_emits(self):
        rows = self.query('''
            SELECT fn1.basic_range(3)
            FROM DUAL
            ''')
        self.assertRowsEqual([(x,) for x in range(3)], sorted(rows))
    
    @requires('BASIC_SUM')
    def test_basic_set_returns(self):
        rows = self.query('''
            SELECT fn1.basic_sum(3)
            FROM DUAL
            ''')
        self.assertRowsEqual([(3,)], rows)

    @requires('BASIC_EMIT_TWO_INTS')
    def test_emit_two_ints(self):
        rows = self.query('''
            SELECT fn1.basic_emit_two_ints()
            FROM DUAL''')
        self.assertRowsEqual([(1, 2)], rows)

    @requires('BASIC_SUM')
    @requires('BASIC_NTH_PARTIAL_SUM')
    @requires('BASIC_RANGE')
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

    @requires('BASIC_SUM_GRP')
    @requires('BASIC_NTH_PARTIAL_SUM')
    @requires('BASIC_RANGE')
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

    @requires('BASIC_EMIT_SEVERAL_GROUPS')
    @requires('BASIC_TEST_RESET')
    def test_reset(self):
        rows = self.query('''
            SELECT fn1.basic_test_reset(i, j)
            FROM (SELECT fn1.basic_emit_several_groups(16, 8) FROM DUAL)
            GROUP BY i
            ORDER BY 1''')
        self.assertRowsEqual([(0.0,), (0.0,), (0.0,), (0.0,), (1.0,), (1.0,), (1.0,), (1.0,), (2.0,)], rows[:9])

    @requires('PERFORMANCE_REDUCE_CHARACTERS')
    @requires('PERFORMANCE_MAP_CHARACTERS')
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
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')
        self.query('CREATE TABLE FN2.empty_table(c int)')

    @requires('SET_RETURNS_HAS_EMPTY_INPUT')
    def test_set_returns_has_empty_input_group_by(self):
        self.query("""select FN1.set_returns_has_empty_input(c) from empty_table group by 'X'""")
        self.assertEqual(0, self.rowcount())

    @requires('SET_RETURNS_HAS_EMPTY_INPUT')
    def test_set_returns_has_empty_input_no_group_by(self):
        rows = self.query('''select FN1.set_returns_has_empty_input(c) from empty_table''')
        self.assertRowsEqual([(None,)], rows)


    @requires('SET_EMITS_HAS_EMPTY_INPUT')
    def test_set_emits_has_empty_input_group_by(self):
        self.query("""select FN1.set_emits_has_empty_input(c) from empty_table group by 'X'""")
        self.assertEqual(0, self.rowcount())

    @requires('SET_EMITS_HAS_EMPTY_INPUT')
    def test_set_emits_has_empty_input_no_group_by(self):
        rows = self.query('''select FN1.set_emits_has_empty_input(c) from empty_table''')
        self.assertRowsEqual([(None,None)], rows)


if __name__ == '__main__':
    udf.main()
