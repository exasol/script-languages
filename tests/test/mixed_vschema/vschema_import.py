#!/usr/opt/bs-python-2.7/bin/python

import os
import sys
import datetime
import unittest
from multiprocessing import Process
from textwrap import dedent

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf

from vschema_common import VSchemaTest, TestUtils

class ExplainVirtualPushdown(VSchemaTest):
    setupDone = False

    def setUp(self):
        # TODO This is another ugly workaround for the problem that the framework doesn't offer us a query in classmethod setUpClass. Rewrite!
        if self.__class__.setupDone:
            self.query(''' CLOSE SCHEMA ''')
            return

        self.createJdbcAdapter()
        self.createNative()
        self.commit()  # We have to commit, otherwise the adapter won't see these tables
        self.createVirtualSchemaJdbc("VS1", "NATIVE", "ADAPTER.JDBC_ADAPTER", False)
        self.commit()
        self.query(''' CLOSE SCHEMA ''')
        self.__class__.setupDone = True

    def testPushdownResponses(self):
        # Single Group
        self.compareWithNativeExtended('''
            SELECT a, c FROM {v}.T;
        ''', ignoreOrder = True, explainResponse='''SELECT A, C FROM NATIVE.T''')
        self.compareWithNativeExtended('''
            SELECT t1.a FROM {v}.t t1, {v}.t t2
        ''', ignoreOrder = True, explainResponse=["SELECT true FROM NATIVE.T","SELECT A FROM NATIVE.T"])
        self.compareWithNativeExtended('''
            SELECT (a+1) a1 FROM {v}.t
        ''', ignoreOrder = True, explainResponse='''SELECT (A + 1) FROM NATIVE.T''')

    def testNestedPushdowns(self):
        self.compareWithNativeExtended('''
            SELECT a FROM (SELECT a FROM {v}.t ORDER BY false);
        ''', ignoreOrder = True, explainResponse='''SELECT A FROM NATIVE.T ORDER BY false''')

        self.compareWithNativeExtended('''
            SELECT a FROM (SELECT a FROM {v}.t ORDER BY false), (SELECT b FROM {v}.t ORDER BY false);
        ''', ignoreOrder = True, explainResponse=['SELECT A FROM NATIVE.T ORDER BY false', 'SELECT NULL FROM NATIVE.T ORDER BY false']) # review!

        self.compareWithNativeExtended('''
            SELECT * FROM {v}.t WHERE a IN (SELECT DISTINCT a FROM {v}.t ORDER BY a DESC LIMIT 2);
        ''', ignoreOrder = True, explainResponse=['''SELECT * FROM NATIVE.T''','''SELECT A FROM NATIVE.T GROUP BY A ORDER BY A DESC LIMIT 2'''])


    def testJoins(self):
        # Equi Join
        self.compareWithNativeExtended('''
            select t1.a FROM {v}.t t1 join {v}.t t2 on t1.b=t2.b
        ''', ignoreOrder = True, explainResponse=['''SELECT A, B FROM NATIVE.T''','''SELECT B FROM NATIVE.T'''])
        # Outer Join
        self.compareWithNativeExtended('''
            select * FROM {v}.t t1 left join {v}.t t2 on t1.a=t2.a where coalesce(t2.a, 1) = 1
        ''', ignoreOrder = True, explainResponse=['''SELECT * FROM NATIVE.T''', '''SELECT * FROM NATIVE.T'''])
        # Cross Join
        self.compareWithNativeExtended('''
            select t1.a FROM {v}.t t1, {v}.t t2
        ''', ignoreOrder = True, explainResponse=['''SELECT A FROM NATIVE.T''', '''SELECT true FROM NATIVE.T'''])
        # Join with native table
        self.compareWithNativeExtended('''
            select * from {v}.t vt join {n}.t nt on vt.a = nt.a where nt.a = 1
        ''', ignoreOrder = True, explainResponse='''SELECT * FROM NATIVE.T''')

    def testSelectListExpressions(self):
        self.compareWithNativeExtended('''
            select a+1 from {v}.t order by c desc
        ''', ignoreOrder = True, explainResponse='''SELECT (A + 1) FROM NATIVE.T ORDER BY C DESC''')

    def testPredicates(self):
        self.compareWithNativeExtended('''
            SELECT a=1, b FROM {v}.t WHERE a=(a*2/2)
        ''', ignoreOrder = True, explainResponse='''SELECT A = 1, B FROM NATIVE.T WHERE A = ((A * 2) / 2)''')


    def testOrderByLimit(self):
        self.compareWithNativeExtended('''
            select a+1 as a1, c from {v}.t order by a+1
        ''', ignoreOrder = True, explainResponse='''SELECT (A + 1), C FROM NATIVE.T ORDER BY (A + 1)''')

    def testAggregation(self):
        # Single Group
        self.compareWithNativeExtended('''
            select count(*) from {v}.t
        ''', ignoreOrder = True, explainResponse='''SELECT COUNT(*) FROM NATIVE.T''')

        # Group By Expression
        self.compareWithNativeExtended('''
            select a*2, count(*), max(b) from {v}.t group by a*2
        ''', ignoreOrder = True, explainResponse='''SELECT (A * 2), COUNT(*), MAX(B) FROM NATIVE.T GROUP BY (A * 2)''')

        # Aggregation On Join
        self.compareWithNativeExtended('''
            select sum(t1.a) from {v}.t t1, {v}.t t2 group by t1.a
        ''', ignoreOrder = True,  explainResponse=['SELECT A FROM NATIVE.T', 'SELECT true FROM NATIVE.T'])

    def testScalarFunctions(self):
        # Aggregation On Join
        self.compareWithNativeExtended('''
            select * from {v}.t where abs(a) = 1
        ''', ignoreOrder = True,  explainResponse='SELECT * FROM NATIVE.T WHERE ABS(A) = 1')

    def testMultiPushdown(self):
        self.createVirtualSchemaJdbc("VS2", "NATIVE", "ADAPTER.JDBC_ADAPTER", True)
        # Create an additional virtual schema using another adapter
        self.createJdbcAdapter(schemaName="ADAPTER2", adapterName="JDBC_ADAPTER")
        self.createVirtualSchemaJdbc("VS3", "NATIVE", "ADAPTER2.JDBC_ADAPTER", True)
        # 1 virtual schema, n virtual tables
        self.compareWithNativeExtended('''
            select * from {v}.t t1, {v}.t t2, {v}.t t3 where t1.a = t2.a and t2.a = t3.a;
        ''', ignoreOrder = True,  explainResponse=['SELECT * FROM NATIVE.T', 'SELECT * FROM NATIVE.T', 'SELECT * FROM NATIVE.T'])

        # 1 adapter, n virtual schemas
        self.compareWithNativeExtended('''
            select * from {v}.t t1, {v2}.t t2, {v}.t t3 where t1.a = t2.a and t2.a = t3.a;
        ''',  ignoreOrder = True, explainResponse=['SELECT * FROM NATIVE.T', 'SELECT * FROM NATIVE.T', 'SELECT * FROM NATIVE.T'])

        # different adapters, different schemas
        self.compareWithNativeExtended('''
            select * from {v}.t t1, {v3}.t t2 where t1.a = t2.a;
        ''', ignoreOrder = True, explainResponse=['SELECT * FROM NATIVE.T', 'SELECT * FROM NATIVE.T'])
        self.compareWithNativeExtended('''
            select * from {v}.t t1, (select a, b from {v3}.t) t2 where t1.a = t2.a;
        ''', ignoreOrder = True,  explainResponse=['SELECT * FROM NATIVE.T','SELECT A, B FROM NATIVE.T'])
        self.compareWithNativeExtended('''
            select * from {v}.t where a in (select distinct a from {v3}.t order by a desc limit 2);
        ''', ignoreOrder = True,  explainResponse=['SELECT * FROM NATIVE.T','SELECT A FROM NATIVE.T GROUP BY A ORDER BY A DESC LIMIT 2'])

    def testWithAnalytical(self):
        self.compareWithNativeExtended('''
            SELECT k, v1, sum(v1) over (PARTITION BY k ORDER BY v1) AS SUM FROM {v}.g order by k desc, sum;
        ''', ignoreOrder = True,  explainResponse='SELECT K, V1 FROM NATIVE.G')

    def testMixed(self):
        # Special Case: c*c is removed from select list, so only lookups in selectlist. Should still pushdown agg.
        self.compareWithNativeExtended('''
            SELECT count(a) FROM (
              SELECT a,c*c as x, sum(c) mysum FROM {v}.t GROUP BY a,c*c) subsel
            WHERE subsel.x<15 AND mysum>2;
        ''', ignoreOrder = True,  explainResponse='''SELECT A FROM NATIVE.T WHERE (C * C) < 15 GROUP BY (C * C), A HAVING 2 < SUM(C)''') # review!

        # ... same with b only in filter
        self.compareWithNativeExtended('''
            SELECT count(a) FROM (
              SELECT a,c*c as x, sum(c) mysum FROM {v}.t WHERE b!='f' GROUP BY a,c*c) subsel
            WHERE subsel.x<15 AND mysum>2;
        ''', ignoreOrder = True,  explainResponse='''SELECT A FROM NATIVE.T WHERE (B != ''f'' AND (C * C) < 15) GROUP BY (C * C), A HAVING 2 < SUM(C)''') # review!

        # ... same with join
        self.compareWithNativeExtended('''
            SELECT count(a) FROM (
              SELECT t1.a,t1.c*t1.c as x, sum(t1.c) mysum FROM {v}.t t1 JOIN {v}.t t2 ON t1.b=t2.b GROUP BY t1.a,t1.c*t1.c) subsel
            WHERE subsel.x<15 AND mysum>2;
        ''', ignoreOrder = True,  explainResponse=['SELECT * FROM NATIVE.T WHERE (C * C) < 15', 'SELECT B FROM NATIVE.T'])

    def testDataTypes(self):
        self.compareWithNativeExtended('''SELECT a1 FROM {v}.t_datatypes;''',  explainResponse='IMPORT INTO (c0 DECIMAL(18, 0)) FROM')
        self.compareWithNativeExtended('''SELECT a2 FROM {v}.t_datatypes;''',  explainResponse='IMPORT INTO (c0 DOUBLE) FROM')
        self.compareWithNativeExtended('''SELECT a3 FROM {v}.t_datatypes;''',  explainResponse='IMPORT INTO (c0 DATE) FROM')
        self.compareWithNativeExtended('''SELECT a4 FROM {v}.t_datatypes;''',  explainResponse='IMPORT INTO (c0 TIMESTAMP) FROM')
        self.compareWithNativeExtended('''SELECT a5 FROM {v}.t_datatypes;''',  explainResponse='IMPORT INTO (c0 VARCHAR(3000) UTF8) FROM')
        self.compareWithNativeExtended('''SELECT a6 FROM {v}.t_datatypes;''',  explainResponse='IMPORT INTO (c0 CHAR(10) UTF8) FROM')
        self.compareWithNativeExtended('''SELECT a7 FROM {v}.t_datatypes;''',  explainResponse='IMPORT INTO (c0 BOOLEAN) FROM')
        #self.compareWithNativeExtended('''SELECT a8 FROM {v}.t_datatypes;''',  explainResponse='IMPORT INTO (c0 interval day to second) FROM')
        #self.compareWithNativeExtended('''SELECT a9 FROM {v}.t_datatypes;''',  explainResponse='IMPORT INTO (c0 interval year to month) FROM')
        self.compareWithNativeExtended('''SELECT a10 FROM {v}.t_datatypes;''',  explainResponse='IMPORT INTO (c0 VARCHAR(2000000) UTF8) FROM')
        self.compareWithNativeExtended('''SELECT a11 FROM {v}.t_datatypes;''',  explainResponse='IMPORT INTO (c0 DECIMAL(10, 5)) FROM')
        self.compareWithNativeExtended('''SELECT a12 FROM {v}.t_datatypes;''',  explainResponse='IMPORT INTO (c0 DOUBLE) FROM')
        self.compareWithNativeExtended('''SELECT a13 FROM {v}.t_datatypes;''',  explainResponse='IMPORT INTO (c0 DECIMAL(36, 0)) FROM')
        self.compareWithNativeExtended('''SELECT a14 FROM {v}.t_datatypes;''',  explainResponse='IMPORT INTO (c0 DECIMAL(18, 0)) FROM')
        self.compareWithNativeExtended('''SELECT a15 FROM {v}.t_datatypes;''',  explainResponse='IMPORT INTO (c0 DECIMAL(29, 0)) FROM')
        self.compareWithNativeExtended('''SELECT a16 FROM {v}.t_datatypes;''',  explainResponse='IMPORT INTO (c0 DECIMAL(18, 0)) FROM')
        self.compareWithNativeExtended('''SELECT a17 FROM {v}.t_datatypes;''',  explainResponse='IMPORT INTO (c0 DECIMAL(25, 0)) FROM')
        self.compareWithNativeExtended('''SELECT a18 FROM {v}.t_datatypes;''',  explainResponse='IMPORT INTO (c0 DECIMAL(27, 9)) FROM')
        self.compareWithNativeExtended('''SELECT a19 FROM {v}.t_datatypes;''',  explainResponse='IMPORT INTO (c0 DOUBLE) FROM')
        self.compareWithNativeExtended('''SELECT a20 FROM {v}.t_datatypes;''',  explainResponse='IMPORT INTO (c0 DECIMAL(18, 0)) FROM')
        self.compareWithNativeExtended('''SELECT a21 FROM {v}.t_datatypes;''',  explainResponse='IMPORT INTO (c0 DOUBLE) FROM')
        self.compareWithNativeExtended('''SELECT a22 FROM {v}.t_datatypes;''',  explainResponse='IMPORT INTO (c0 DECIMAL(1, 0)) FROM')
        self.compareWithNativeExtended('''SELECT a23 FROM {v}.t_datatypes;''',  explainResponse='IMPORT INTO (c0 DECIMAL(3, 2)) FROM')
        self.compareWithNativeExtended('''SELECT a24 FROM {v}.t_datatypes;''',  explainResponse='IMPORT INTO (c0 DECIMAL(18, 0)) FROM')
        self.compareWithNativeExtended('''SELECT a25 FROM {v}.t_datatypes;''',  explainResponse='IMPORT INTO (c0 DECIMAL(6, 0)) FROM')
        self.compareWithNativeExtended('''SELECT a26 FROM {v}.t_datatypes;''',  explainResponse='IMPORT INTO (c0 DECIMAL(6, 3)) FROM')
        self.compareWithNativeExtended('''SELECT a27 FROM {v}.t_datatypes;''',  explainResponse='IMPORT INTO (c0 DOUBLE) FROM')
        self.compareWithNativeExtended('''SELECT a28 FROM {v}.t_datatypes;''',  explainResponse='IMPORT INTO (c0 DECIMAL(9, 0)) FROM')
        self.compareWithNativeExtended('''SELECT a29 FROM {v}.t_datatypes;''',  explainResponse='IMPORT INTO (c0 DECIMAL(9, 0)) FROM')
        self.compareWithNativeExtended('''SELECT a30 FROM {v}.t_datatypes;''',  explainResponse='IMPORT INTO (c0 DECIMAL(3, 0)) FROM')
        self.compareWithNativeExtended('''SELECT a31 FROM {v}.t_datatypes;''',  explainResponse='IMPORT INTO (c0 DATE) FROM')
        self.compareWithNativeExtended('''SELECT a32 FROM {v}.t_datatypes;''',  explainResponse='IMPORT INTO (c0 TIMESTAMP) FROM')

if __name__ == '__main__':
    udf.main()

