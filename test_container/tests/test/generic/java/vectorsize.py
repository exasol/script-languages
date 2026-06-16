#!/usr/bin/env python3

from exasol_python_test_framework import udf
from exasol_python_test_framework.udf import useData


class _JavaUdfSetup(udf.TestCase):
    LANG = 'java'
    def setUp(self):
        self.query('DROP SCHEMA FN1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN1')
        self.query('OPEN SCHEMA FN1')
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
            CREATE JAVA SCALAR SCRIPT
            vectorsize(length DOUBLE, dummy DOUBLE)
            RETURNS VARCHAR(2000000) AS
            import java.util.Map;
            import java.util.HashMap;

            class VECTORSIZE {
                static Map<Integer, String> cache = new HashMap<Integer, String>();

                static void fillCache(int len) {
                    StringBuilder sb = new StringBuilder();
                    for (int i = 0; i < len; i++) {
            sb.append(Integer.toString(i));
                    }
                    cache.put(len, sb.toString());
                }

                static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    int len = ctx.getInteger("length");
                    if (!cache.containsKey(len))
            fillCache(len);
                    return cache.get(len);
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            CREATE JAVA SCALAR SCRIPT vectorsize5000(A DOUBLE)
            RETURNS VARCHAR(2000000) AS
            class VECTORSIZE5000 {
                static String numbers;

                static {
                    StringBuilder sb = new StringBuilder();
                    for (int i = 0; i < 5000; i++) {
            sb.append(Integer.toString(i));
                    }
                    numbers = sb.toString();
                }

                static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return numbers;
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            CREATE JAVA SCALAR SCRIPT
            vectorsize_set(length DOUBLE, n DOUBLE, dummy DOUBLE)
            EMITS (o VARCHAR(2000000)) AS
            import java.util.Map;
            import java.util.HashMap;

            class VECTORSIZE_SET {
                static Map<Integer, String> cache = new HashMap<Integer, String>();

                static void fillCache(int len) {
                    StringBuilder sb = new StringBuilder();
                    for (int i = 0; i < len; i++) {
            sb.append(Integer.toString(i));
                    }
                    cache.put(len, sb.toString());
                }

                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    int len = ctx.getInteger("length");
                    if (!cache.containsKey(len))
            fillCache(len);
                    int n = ctx.getInteger("n");
                    for (int i = 0; i < n; i++) {
            ctx.emit(cache.get(len));
                    }
                }
            }
            /
        '''))

class Vectorsize(_JavaUdfSetup):

    def test_vectorsize_5000(self):
        self.query('''
		        SELECT max(fn1.vectorsize5000(float1))
                FROM TEST.ENGINETABLEBIG1''')

    data = [
            (10,),
            (30,),
            (100,),
            (300,),
            (1000,),
            (3000,),
            (10000,),
            (30000,),
            (100000,),
            (200000,),
            (351850,),
            ]

    @useData(data)
    def test_vectorsize(self, size):
        if size > 3000:
            raise udf.SkipTest('test is to slow')

        self.query('''
                SELECT max(fn1.vectorsize(%d, float1))
                FROM TEST.ENGINETABLEBIG1
                ''' % size)

    data = [
            (10, 10, 10),
            (100, 100, 100),
            (1000, 100, 100),
            (10000, 100, 100),
            (100000, 100, 100),
            (351850, 100, 100),
            (100, 10, 100000),
            (100, 100, 10000),
            (100, 1000, 1000),
            (100, 10000, 100),
            (100, 100000, 10),
            ]

    @useData(data)
    def test_vectorsize_set(self, a, b, c):
        q = '''
                SELECT max(o)
                FROM (
                    SELECT fn1.vectorsize_set(%d, %d, n)
                    FROM (
                        SELECT fn1.basic_range(%d)
                        FROM DUAL
                    )
                )
                ''' % (a, b, c)
        self.query(q)


if __name__ == '__main__':
    udf.main()
