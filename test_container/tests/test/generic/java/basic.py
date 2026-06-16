#!/usr/bin/env python3

from exasol_python_test_framework import udf


class _SetEmptyInputUdfs:
    """Mixin providing creation helpers for the set_*_has_empty_input UDFs."""

    def _create_set_returns_has_empty_input(self):
        self.query(udf.fixindent('''
            CREATE java SET SCRIPT
            set_returns_has_empty_input(a double) RETURNS boolean AS
            class SET_RETURNS_HAS_EMPTY_INPUT {
                static boolean run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return ctx.getDouble("a") == null;
                }
            }
            /
        '''))

    def _create_set_emits_has_empty_input(self):
        self.query(udf.fixindent('''
            CREATE java SET SCRIPT
            set_emits_has_empty_input(a double) EMITS (x double, y varchar(10)) AS
            class SET_EMITS_HAS_EMPTY_INPUT {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    if (ctx.getDouble("a") == null)
            ctx.emit(1,"1");
                    else
            ctx.emit(2,"2");
                }
            }
            /
        '''))


class _JavaUdfSetup(_SetEmptyInputUdfs, udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN1')
        self.query('OPEN SCHEMA FN1')
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT
            basic_emit_several_groups(a INTEGER, b INTEGER)
            EMITS (i INTEGER, j VARCHAR(40)) AS
            class BASIC_EMIT_SEVERAL_GROUPS {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    for (int n = 0; n < ctx.getInteger("a"); n++)
            for (int i = 0; i < ctx.getInteger("b"); i++)
                ctx.emit(i, exa.getVmId());
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT
            basic_emit_two_ints()
            EMITS (i INTEGER, j INTEGER) AS
            class BASIC_EMIT_TWO_INTS {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    ctx.emit(1,2);
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT
            basic_nth_partial_sum(n INTEGER)
            RETURNS INTEGER as
            class BASIC_NTH_PARTIAL_SUM {
                static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    if (ctx.getInteger("n") != null)
            return ctx.getInteger("n") * (ctx.getInteger("n") + 1) / 2;
                    return 0;
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT
            basic_range(n INTEGER)
            EMITS (n INTEGER) AS
            class BASIC_RANGE {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    if (ctx.getInteger("n") != null)
            for (int i = 0; i < ctx.getInteger("n"); i++)
                ctx.emit(i);
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            CREATE java SET SCRIPT
            basic_sum(x INTEGER)
            RETURNS INTEGER AS
            class BASIC_SUM {
                static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    int s = 0;
                    while (true) {
            if (ctx.getInteger("x") != null)
                s += ctx.getInteger("x");
            if (!ctx.next())
                break;
                    }
                    return s;
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            CREATE java SET SCRIPT
            basic_sum_grp(x INTEGER)
            EMITS (s INTEGER) AS
            class BASIC_SUM_GRP {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    int s = 0;
                    while (true) {
            if (ctx.getInteger("x") != null)
                s += ctx.getInteger("x");
            if (!ctx.next())
                break;
                    }
                    ctx.emit(s);
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            CREATE java SET SCRIPT
            basic_test_reset(i INTEGER, j VARCHAR(40))
            EMITS (k INTEGER) AS
            class BASIC_TEST_RESET {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    ctx.emit(ctx.getInteger("i"));
                    ctx.next();
                    ctx.emit(ctx.getInteger("i"));
                    ctx.reset();
                    ctx.emit(ctx.getInteger("i"));
                    ctx.next();
                    ctx.emit(ctx.getInteger("i"));
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT
            performance_map_characters(text VARCHAR(1000))
            EMITS (w CHAR(1), c INTEGER) AS
            class PERFORMANCE_MAP_CHARACTERS {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    String text = ctx.getString("text");
                    if (text != null) {
            for (int i = 0; i < text.length(); i++) {
                if (Character.isHighSurrogate(text.charAt(i))) {
                    ctx.emit(text.substring(i, i + 2), 1);
                    i++;
                }
                else {
                    ctx.emit(text.substring(i, i + 1), 1);
                }
            }
                    }
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            CREATE java SET SCRIPT
            performance_reduce_characters(w CHAR(1), c INTEGER)
            EMITS (w CHAR(1), c INTEGER) AS
            class PERFORMANCE_REDUCE_CHARACTERS {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    int c = 0;
                    String w = ctx.getString("w");
                    if (w != null) {
            do {
                c += 1;
            } while (ctx.next());
            ctx.emit(w, c);
                    }
                }
            }
            /
        '''))

        self._create_set_emits_has_empty_input()
        self._create_set_returns_has_empty_input()

class BasicTest(_JavaUdfSetup):

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


class SetWithEmptyInput(_SetEmptyInputUdfs, udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN1')
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')
        self.query('OPEN SCHEMA FN1')
        self.query('CREATE TABLE FN2.empty_table(c int)')
        
        self._create_set_returns_has_empty_input()
        self._create_set_emits_has_empty_input()

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
