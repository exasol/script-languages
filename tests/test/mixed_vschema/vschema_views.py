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

class SubselectTest(VSchemaTest):
    setupDone = False

    def setUp(self):
        # TODO This is another ugly workaround for the problem that the framework doesn't offer us a query in classmethod setUpClass. Rewrite!
        if self.__class__.setupDone:
            self.query(''' CLOSE SCHEMA ''')
            return
        #self.initPool()
        self.createJdbcAdapter()
        self.createNative()
        self.commit()  # We have to commit, otherwise the adapter won't see these tables
        self.createVirtualSchemaJdbc("VS1", "NATIVE", "ADAPTER.JDBC_ADAPTER", True)
        self.commit()
        self.createVirtualSchemaJdbc("VS2", "NATIVE", "ADAPTER.JDBC_ADAPTER", True)
        self.commit()
        self.query(''' CLOSE SCHEMA ''')
        self.__class__.setupDone = True

    def testDifferentOuterSelects(self):
        self.compareWithProfilingExtended('''SELECT a, sum(b), sum(e), max(i) FROM (SELECT a, b, c, e, sum(f) i FROM {v}.numbers1 JOIN {v}.numbers2 on numbers1.d = numbers2.f WHERE b > 2 GROUP BY a, b, c, e ORDER BY b) WHERE a > 3  and e < 8 GROUP BY a ORDER BY a;'''
            , ignoreOrder=True, profilingResponse= [('SELECT * FROM NATIVE.NUMBERS1', ), ('SELECT E, F FROM NATIVE.NUMBERS2', )])
        self.compareWithProfilingExtended('''SELECT a, sum(b), sum(e), max(i) FROM (SELECT a, b, c, e, sum(f) i FROM {v}.numbers1 JOIN {v}.numbers2 on numbers1.d = numbers2.f WHERE b > 2 GROUP BY a, b, c, e ORDER BY b) WHERE a > 3  and e < 8 GROUP BY a;'''
            , ignoreOrder=True, profilingResponse= [('SELECT * FROM NATIVE.NUMBERS1', ), ('SELECT E, F FROM NATIVE.NUMBERS2', )])
        self.compareWithProfilingExtended('''SELECT a, b, e, i FROM (SELECT a, b, c, e, sum(f) i FROM {v}.numbers1 JOIN {v}.numbers2 on numbers1.d = numbers2.f WHERE b > 2 GROUP BY a, b, c, e ORDER BY b) WHERE a > 3  and e < 8 ORDER BY a, b, e, i;'''
            , profilingResponse= [('SELECT * FROM NATIVE.NUMBERS1', ), ('SELECT E, F FROM NATIVE.NUMBERS2', )])
        self.compareWithProfilingExtended('''SELECT a, b, e, i FROM (SELECT a, b, c, e, sum(f) i FROM {v}.numbers1 JOIN {v}.numbers2 on numbers1.d = numbers2.f WHERE b > 2 GROUP BY a, b, c, e ORDER BY b) WHERE a > 3 ORDER BY a, b, e, i;'''
            , profilingResponse= [('SELECT * FROM NATIVE.NUMBERS1 WHERE (2 < B AND 3 < A)', ), ('SELECT E, F FROM NATIVE.NUMBERS2', )])
        self.compareWithProfilingExtended('''SELECT a, b, e, i FROM (SELECT a, b, c, e, sum(f) i FROM {v}.numbers1 JOIN {v}.numbers2 on numbers1.d = numbers2.f WHERE b > 2 GROUP BY a, b, c, e ORDER BY b) ORDER BY a, b, e, i;'''
            , profilingResponse= [('SELECT * FROM NATIVE.NUMBERS1 WHERE 2 < B', ), ('SELECT E, F FROM NATIVE.NUMBERS2', )])
        self.compareWithProfilingExtended('''SELECT a, b, e FROM (SELECT a, b, c, e, sum(f) i FROM {v}.numbers1 JOIN {v}.numbers2 on numbers1.d = numbers2.f WHERE b > 2 GROUP BY a, b, c, e ORDER BY b) ORDER BY a, b, e;'''
            , profilingResponse= [('SELECT * FROM NATIVE.NUMBERS1 WHERE 2 < B', ), ('SELECT E, F FROM NATIVE.NUMBERS2', )])
        self.compareWithProfilingExtended('''SELECT a, b FROM (SELECT a, b, c, e, sum(f) i FROM {v}.numbers1 JOIN {v}.numbers2 on numbers1.d = numbers2.f WHERE b > 2 GROUP BY a, b, c, e ORDER BY b) ORDER BY a, b;'''
            , profilingResponse= [('SELECT * FROM NATIVE.NUMBERS1 WHERE 2 < B', ), ('SELECT E, F FROM NATIVE.NUMBERS2', )])
        self.compareWithProfilingExtended('''SELECT a FROM (SELECT a, b, c, e, sum(f) i FROM {v}.numbers1 JOIN {v}.numbers2 on numbers1.d = numbers2.f WHERE b > 2 GROUP BY a, b, c, e ORDER BY b) ORDER BY a;'''
            , profilingResponse= [('SELECT * FROM NATIVE.NUMBERS1 WHERE 2 < B', ), ('SELECT E, F FROM NATIVE.NUMBERS2', )])
        self.compareWithProfilingExtended('''SELECT * FROM (SELECT a, b, c, e, sum(f) i FROM {v}.numbers1 JOIN {v}.numbers2 on numbers1.d = numbers2.f WHERE b > 2 GROUP BY a, b, c, e ORDER BY b) ORDER BY a, b, c, e, i;'''
            , profilingResponse= [('SELECT * FROM NATIVE.NUMBERS1 WHERE 2 < B', ), ('SELECT E, F FROM NATIVE.NUMBERS2', )])

    def testDifferentSubselects(self):
        self.compareWithProfilingExtended('''SELECT a, sum(b) x FROM (SELECT a, b, c FROM {v}.numbers1 WHERE b > 2 GROUP BY a, b, c ORDER BY b) WHERE a > 3 GROUP BY a ORDER BY a, x;'''
            , ignoreOrder=True, profilingResponse= [('SELECT A, B FROM NATIVE.NUMBERS1 WHERE (2 < B AND 3 < A) GROUP BY A, B, C ORDER BY B', ),])
        self.compareWithProfilingExtended('''SELECT a, b FROM (SELECT a, b, c FROM {v}.numbers1 WHERE b > 2 GROUP BY a, b, c ORDER BY b) WHERE a > 3 ORDER BY a, b;'''
            , profilingResponse= [('SELECT A, B FROM NATIVE.NUMBERS1 WHERE (2 < B AND 3 < A) GROUP BY A, B, C ORDER BY B', ),])
        self.compareWithProfilingExtended('''SELECT a FROM (SELECT a, b, c FROM {v}.numbers1 WHERE b > 2 GROUP BY a, b, c ORDER BY b) WHERE a > 3 ORDER BY a;'''
            , profilingResponse= [('SELECT A, B FROM NATIVE.NUMBERS1 WHERE (2 < B AND 3 < A) GROUP BY A, B, C ORDER BY B', ),])
        self.compareWithProfilingExtended('''SELECT a FROM (SELECT a, b, c FROM {v}.numbers1 WHERE b > 2 GROUP BY a, b, c ORDER BY b) ORDER BY a;'''
            , profilingResponse= [('SELECT A, B FROM NATIVE.NUMBERS1 WHERE 2 < B GROUP BY A, B, C ORDER BY B', ),])
        self.compareWithProfilingExtended('''SELECT * FROM (SELECT a, b, c FROM {v}.numbers1 WHERE b > 2 GROUP BY a, b, c ORDER BY b) ORDER BY a, b, c;'''
            , profilingResponse= [('SELECT A, B, C FROM NATIVE.NUMBERS1 WHERE 2 < B GROUP BY A, B, C ORDER BY B', ),])
        self.compareWithProfilingExtended('''SELECT a, sum(b) x FROM (SELECT a, b, c FROM {v}.numbers1 WHERE b > 2 GROUP BY a, b, c) WHERE a > 3 GROUP BY a ORDER BY a, x;'''
            , ignoreOrder=True, profilingResponse= [('SELECT A, B FROM NATIVE.NUMBERS1 WHERE (2 < B AND 3 < A) GROUP BY A, B, C', ),])
        self.compareWithProfilingExtended('''SELECT a, sum(b) x FROM (SELECT a, b, c FROM {v}.numbers1 WHERE b > 2) WHERE a > 3 GROUP BY a ORDER BY a, x;'''
            , ignoreOrder=True, profilingResponse= [('SELECT A, SUM(B) FROM NATIVE.NUMBERS1 WHERE (3 < A AND 2 < B) GROUP BY A ORDER BY A, SUM(B)', ),])
        self.compareWithProfilingExtended('''SELECT a, sum(b) x FROM (SELECT a, b, c FROM {v}.numbers1) WHERE a > 3 GROUP BY a ORDER BY a, x;'''
            , ignoreOrder=True, profilingResponse= [('SELECT A, SUM(B) FROM NATIVE.NUMBERS1 WHERE 3 < A GROUP BY A ORDER BY A, SUM(B)', ),])
        self.compareWithProfilingExtended('''SELECT a, sum(b) x FROM (SELECT * FROM {v}.numbers1) WHERE a > 3 GROUP BY a ORDER BY a, x;'''
            , ignoreOrder=True, profilingResponse= [('SELECT A, SUM(B) FROM NATIVE.NUMBERS1 WHERE 3 < A GROUP BY A ORDER BY A, SUM(B)', ),])

    def testDifferentJoins(self):
        self.compareWithProfilingExtended('''SELECT a, sum(b), sum(e), max(i) FROM (SELECT a, b, c, e, sum(f) i FROM {v}.numbers1 JOIN NATIVE.numbers2 on numbers1.d = numbers2.f WHERE b > 2 GROUP BY a, b, c, e ORDER BY b) WHERE a > 3  and e < 8 GROUP BY a ORDER BY a;'''
            , ignoreOrder=True, profilingResponse= [('SELECT * FROM NATIVE.NUMBERS1', ),])
        self.query("DROP VIEW IF EXISTS NATIVE.FILTER_VIEW1_NATIVE")
        self.query("DROP VIEW IF EXISTS NATIVE.FILTER_VIEW1_VS1")
        self.query("CREATE VIEW NATIVE.FILTER_VIEW1_NATIVE AS SELECT E,F,G,H FROM NATIVE.numbers2 WHERE G > 4")
        self.query("CREATE VIEW NATIVE.FILTER_VIEW1_VS1 AS SELECT E,F,G,H FROM VS1.numbers2 WHERE G > 4")
        self.compareWithProfilingExtended('''SELECT a, sum(b), sum(e), max(i) FROM (SELECT a, b, c, e, sum(f) i FROM {v}.numbers1 JOIN NATIVE.FILTER_VIEW1_NATIVE on numbers1.d = FILTER_VIEW1_NATIVE.f WHERE b > 2 GROUP BY a, b, c, e ORDER BY b) WHERE a > 3  and e < 8 GROUP BY a ORDER BY a;'''
            , ignoreOrder=True, profilingResponse= [('SELECT * FROM NATIVE.NUMBERS1', )])
        self.compareWithProfilingExtended('''SELECT a, sum(b), sum(e), max(i) FROM (SELECT a, b, c, e, sum(f) i FROM {v}.numbers1 JOIN NATIVE.FILTER_VIEW1_VS1 on numbers1.d = FILTER_VIEW1_VS1.f WHERE b > 2 GROUP BY a, b, c, e ORDER BY b) WHERE a > 3  and e < 8 GROUP BY a ORDER BY a;'''
            , ignoreOrder=True, profilingResponse= [('SELECT * FROM NATIVE.NUMBERS1', ), ('SELECT * FROM NATIVE.NUMBERS2 WHERE 4 < G', )])

class ViewTest(VSchemaTest):

    def createAndTestWithViews(self, view, query, ignoreOrder=False, partialOrder=-1, profilingResponse=None):
        self.query("DROP VIEW IF EXISTS NATIVE.FILTER_VIEW_NATIVE")
        self.query("DROP VIEW IF EXISTS NATIVE.FILTER_VIEW_VS1")
        self.query("CREATE VIEW NATIVE.FILTER_VIEW_NATIVE AS {view}".format(view=view).format(v='NATIVE'))
        self.query("CREATE VIEW NATIVE.FILTER_VIEW_VS1 AS {view}".format(view=view).format(v='VS1'))
        self.compareWithProfilingExtended(query, ignoreOrder, partialOrder, profilingResponse)
        self.query("DROP VIEW IF EXISTS NATIVE.FILTER_VIEW_NATIVE")
        self.query("DROP VIEW IF EXISTS NATIVE.FILTER_VIEW_VS1")

    def testDifferentOuterSelects(self):
        self.query("DROP VIEW IF EXISTS NATIVE.FILTER_VIEW2_NATIVE")
        self.query("DROP VIEW IF EXISTS NATIVE.FILTER_VIEW2_VS1")
        self.query("CREATE VIEW NATIVE.FILTER_VIEW2_NATIVE AS SELECT a, b, c, e, sum(f) i FROM NATIVE.numbers1 JOIN NATIVE.numbers2 on numbers1.d = numbers2.f WHERE b > 2 GROUP BY a, b, c, e ORDER BY b")
        self.query("CREATE VIEW NATIVE.FILTER_VIEW2_VS1 AS SELECT a, b, c, e, sum(f) i FROM VS1.numbers1 JOIN VS1.numbers2 on numbers1.d = numbers2.f WHERE b > 2 GROUP BY a, b, c, e ORDER BY b")
        self.compareWithProfilingExtended('''SELECT a, sum(b), sum(e), max(i) FROM NATIVE.FILTER_VIEW2_{v} WHERE a > 3  and e < 8 GROUP BY a ORDER BY a;'''
            , ignoreOrder=True, profilingResponse= [('SELECT * FROM NATIVE.NUMBERS1 WHERE 2 < B', ), ('SELECT E, F FROM NATIVE.NUMBERS2', )])
        self.compareWithProfilingExtended('''SELECT a, sum(b), sum(e), max(i) FROM NATIVE.FILTER_VIEW2_{v} WHERE a > 3  and e < 8 GROUP BY a;'''
            , ignoreOrder=True, profilingResponse= [('SELECT * FROM NATIVE.NUMBERS1 WHERE 2 < B', ), ('SELECT E, F FROM NATIVE.NUMBERS2', )])
        self.compareWithProfilingExtended('''SELECT a, b, e, i FROM NATIVE.FILTER_VIEW2_{v} WHERE a > 3  and e < 8 ORDER BY a, b, e, i;'''
            , profilingResponse= [('SELECT * FROM NATIVE.NUMBERS1 WHERE 2 < B', ), ('SELECT E, F FROM NATIVE.NUMBERS2', )])
        self.compareWithProfilingExtended('''SELECT a, b, e, i FROM NATIVE.FILTER_VIEW2_{v} WHERE a > 3 ORDER BY a, b, e, i;'''
            , profilingResponse= [('SELECT * FROM NATIVE.NUMBERS1 WHERE 2 < B', ), ('SELECT E, F FROM NATIVE.NUMBERS2', )])
        self.compareWithProfilingExtended('''SELECT a, b, e, i FROM NATIVE.FILTER_VIEW2_{v} ORDER BY a, b, e, i;'''
            , profilingResponse= [('SELECT * FROM NATIVE.NUMBERS1 WHERE 2 < B', ), ('SELECT E, F FROM NATIVE.NUMBERS2', )])
        self.compareWithProfilingExtended('''SELECT a, b, e FROM NATIVE.FILTER_VIEW2_{v} ORDER BY a, b, e;'''
            , profilingResponse= [('SELECT * FROM NATIVE.NUMBERS1 WHERE 2 < B', ), ('SELECT E, F FROM NATIVE.NUMBERS2', )])
        self.compareWithProfilingExtended('''SELECT a, b FROM NATIVE.FILTER_VIEW2_{v} ORDER BY a, b;'''
            , profilingResponse= [('SELECT * FROM NATIVE.NUMBERS1 WHERE 2 < B', ), ('SELECT E, F FROM NATIVE.NUMBERS2', )])
        self.compareWithProfilingExtended('''SELECT a FROM NATIVE.FILTER_VIEW2_{v} ORDER BY a;'''
            , profilingResponse= [('SELECT * FROM NATIVE.NUMBERS1 WHERE 2 < B', ), ('SELECT E, F FROM NATIVE.NUMBERS2', )])
        self.compareWithProfilingExtended('''SELECT * FROM NATIVE.FILTER_VIEW2_{v} ORDER BY a, b, c, e, i;'''
            , profilingResponse= [('SELECT * FROM NATIVE.NUMBERS1 WHERE 2 < B', ), ('SELECT E, F FROM NATIVE.NUMBERS2', )])

    def testDifferentViews(self):
        self.createAndTestWithViews('''SELECT a, b, c FROM {v}.numbers1 WHERE b > 2 GROUP BY a, b, c ORDER BY b''', '''SELECT a, sum(b) x FROM NATIVE.FILTER_VIEW_{v} WHERE a > 3 GROUP BY a ORDER BY a, x;'''
        , ignoreOrder=True, profilingResponse= [('SELECT A, B, C FROM NATIVE.NUMBERS1 WHERE 2 < B GROUP BY A, B, C ORDER BY B', ),])
        self.createAndTestWithViews('''SELECT a, b, c FROM {v}.numbers1 WHERE b > 2 GROUP BY a, b, c ORDER BY b''','''SELECT a, b FROM NATIVE.FILTER_VIEW_{v} WHERE a > 3 ORDER BY a, b;'''
            , profilingResponse= [('SELECT A, B, C FROM NATIVE.NUMBERS1 WHERE 2 < B GROUP BY A, B, C ORDER BY B', ),])
        self.createAndTestWithViews('''SELECT a, b, c FROM {v}.numbers1 WHERE b > 2 GROUP BY a, b, c ORDER BY b''', '''SELECT a FROM NATIVE.FILTER_VIEW_{v} WHERE a > 3 ORDER BY a;'''
            , profilingResponse= [('SELECT A, B, C FROM NATIVE.NUMBERS1 WHERE 2 < B GROUP BY A, B, C ORDER BY B', ),])
        self.createAndTestWithViews('''SELECT a, b, c FROM {v}.numbers1 WHERE b > 2 GROUP BY a, b, c ORDER BY b''', '''SELECT a FROM NATIVE.FILTER_VIEW_{v} ORDER BY a;'''
            , profilingResponse= [('SELECT A, B, C FROM NATIVE.NUMBERS1 WHERE 2 < B GROUP BY A, B, C ORDER BY B', ),])
        self.createAndTestWithViews('''SELECT a, b, c FROM {v}.numbers1 WHERE b > 2 GROUP BY a, b, c ORDER BY b''', '''SELECT * FROM NATIVE.FILTER_VIEW_{v} ORDER BY a, b, c;'''
            , profilingResponse= [('SELECT A, B, C FROM NATIVE.NUMBERS1 WHERE 2 < B GROUP BY A, B, C ORDER BY B', ),])
        self.createAndTestWithViews('''SELECT a, b, c FROM {v}.numbers1 WHERE b > 2 GROUP BY a, b, c''', '''SELECT a, sum(b) x FROM NATIVE.FILTER_VIEW_{v} WHERE a > 3 GROUP BY a ORDER BY a, x;'''
            , ignoreOrder=True, profilingResponse= [('SELECT A, B, C FROM NATIVE.NUMBERS1 WHERE 2 < B GROUP BY A, B, C', ),])
        self.createAndTestWithViews('''SELECT a, b, c FROM {v}.numbers1 WHERE b > 2''', '''SELECT a, sum(b) x FROM NATIVE.FILTER_VIEW_{v} WHERE a > 3 GROUP BY a ORDER BY a, x;'''
            , ignoreOrder=True, profilingResponse= [('SELECT A, B, C FROM NATIVE.NUMBERS1 WHERE 2 < B', ),])
        self.createAndTestWithViews('''SELECT a, b, c FROM {v}.numbers1''', '''SELECT a, sum(b) x FROM NATIVE.FILTER_VIEW_{v} WHERE a > 3 GROUP BY a ORDER BY a, x;'''
            , ignoreOrder=True, profilingResponse= [('SELECT A, B, C FROM NATIVE.NUMBERS1', ),])
        self.createAndTestWithViews('''SELECT * FROM {v}.numbers1''', '''SELECT a, sum(b) x FROM NATIVE.FILTER_VIEW_{v} WHERE a > 3 GROUP BY a ORDER BY a, x;'''
            , ignoreOrder=True, profilingResponse= [('SELECT * FROM NATIVE.NUMBERS1', ),])

    def testDifferentJoins(self):
        self.createAndTestWithViews('''SELECT a, b, c, e, sum(f) i FROM {v}.numbers1 JOIN NATIVE.numbers2 on numbers1.d = numbers2.f WHERE b > 2 GROUP BY a, b, c, e ORDER BY b''', '''SELECT a, sum(b), sum(e), max(i) FROM NATIVE.FILTER_VIEW_{v} WHERE a > 3  and e < 8 GROUP BY a ORDER BY a;'''
            , ignoreOrder=True, profilingResponse= [('SELECT * FROM NATIVE.NUMBERS1 WHERE 2 < B', ),])
        self.query("DROP VIEW IF EXISTS NATIVE.FILTER_VIEW1_NATIVE")
        self.query("DROP VIEW IF EXISTS NATIVE.FILTER_VIEW1_VS1")
        self.query("CREATE VIEW NATIVE.FILTER_VIEW1_NATIVE AS SELECT E,F,G,H FROM NATIVE.numbers2 WHERE G > 4")
        self.query("CREATE VIEW NATIVE.FILTER_VIEW1_VS1 AS SELECT E,F,G,H FROM VS1.numbers2 WHERE G > 4")
        self.createAndTestWithViews('''SELECT a, b, c, e, sum(f) i FROM {v}.numbers1 JOIN NATIVE.FILTER_VIEW1_NATIVE on numbers1.d = FILTER_VIEW1_NATIVE.f WHERE b > 2 GROUP BY a, b, c, e ORDER BY b''', '''SELECT a, sum(b), sum(e), max(i) FROM NATIVE.FILTER_VIEW_{v} WHERE a > 3  and e < 8 GROUP BY a ORDER BY a;'''
            , ignoreOrder=True, profilingResponse= [('SELECT * FROM NATIVE.NUMBERS1 WHERE 2 < B', )])
        self.createAndTestWithViews('''SELECT a, b, c, e, sum(f) i FROM {v}.numbers1 JOIN NATIVE.FILTER_VIEW1_VS1 on numbers1.d = FILTER_VIEW1_VS1.f WHERE b > 2 GROUP BY a, b, c, e ORDER BY b''', '''SELECT a, sum(b), sum(e), max(i) FROM NATIVE.FILTER_VIEW_{v} WHERE a > 3  and e < 8 GROUP BY a ORDER BY a;'''
            , ignoreOrder=True, profilingResponse= [('SELECT * FROM NATIVE.NUMBERS1 WHERE 2 < B', ), ('SELECT * FROM NATIVE.NUMBERS2 WHERE 4 < G', )])



class ViewInPushdownTest(VSchemaTest):

    def simpleViewInPushdown(self):
        self.query("DROP VIEW IF EXISTS NATIVE.FILTER_VIEW3_NATIVE")
        self.query("DROP VIEW IF EXISTS NATIVE.FILTER_VIEW3_VS1")
        self.query("CREATE VIEW NATIVE.FILTER_VIEW3_NATIVE AS SELECT * FROM NATIVE.numbers1")
        self.query("COMMIT")
        self.query('''
            ALTER VIRTUAL SCHEMA VS1 REFRESH ''')
        self.compareWithProfilingExtended('''SELECT * FROM {v}.FILTER_VIEW3_NATIVE;'''
            , ignoreOrder=True, profilingResponse= [('SELECT * FROM NATIVE.FILTER_VIEW3_NATIVE', ), ])
        self.query("DROP VIEW IF EXISTS NATIVE.FILTER_VIEW3_NATIVE")
        self.query("DROP VIEW IF EXISTS NATIVE.FILTER_VIEW3_VS1")

    def virtualTableInPushdown(self):
        self.query("DROP VIEW IF EXISTS NATIVE.FILTER_VIEW3_NATIVE")
        self.query("DROP VIEW IF EXISTS NATIVE.FILTER_VIEW3_VS1")
        self.query("CREATE VIEW NATIVE.FILTER_VIEW3_NATIVE AS SELECT * FROM VS2.numbers1")
        self.query("CREATE VIEW NATIVE.FILTER_VIEW4_NATIVE AS SELECT * FROM NATIVE.FILTER_VIEW3_NATIVE")
        self.query("COMMIT")
        self.query('''
            ALTER VIRTUAL SCHEMA VS1 REFRESH ''')
        with self.assertRaisesRegexp(Exception, 'The pushdown query returned by the Adapter contains a virtual table \\(NUMBERS1\\). This is currently not supported.'):
            self.query('''
                SELECT * FROM VS1.FILTER_VIEW3_NATIVE
                ''')
        with self.assertRaisesRegexp(Exception, 'The pushdown query returned by the Adapter contains a virtual table \\(NUMBERS1\\). This is currently not supported.'):
            self.query('''
                SELECT * FROM VS1.FILTER_VIEW4_NATIVE
                ''')
        self.query("DROP VIEW IF EXISTS NATIVE.FILTER_VIEW3_NATIVE")
        self.query("DROP VIEW IF EXISTS NATIVE.FILTER_VIEW3_VS1")

class WithClauseTest(VSchemaTest):

    def createAndTestWithClause(self, view, query, ignoreOrder=False, partialOrder=-1, profilingResponse=None):
        self.compareWithProfilingExtended(view + " " + query, ignoreOrder, partialOrder, profilingResponse)

    def testDifferentOuterSelects(self):
        self.createAndTestWithClause('''WITH TMP_VIEW AS (SELECT a, b, c, e, sum(f) i FROM {v}.numbers1 JOIN {v}.numbers2 on numbers1.d = numbers2.f WHERE b > 2 GROUP BY a, b, c, e ORDER BY b)''', '''SELECT a, sum(b), sum(e), max(i) FROM TMP_VIEW WHERE a > 3  and e < 8 GROUP BY a ORDER BY a;'''
            , ignoreOrder=True, profilingResponse= [('SELECT * FROM NATIVE.NUMBERS1 WHERE 2 < B', ), ('SELECT E, F FROM NATIVE.NUMBERS2', ), ('SELECT * FROM NATIVE.NUMBERS1 WHERE 2 < B', ), ('SELECT E, F FROM NATIVE.NUMBERS2', )])
        self.createAndTestWithClause('''WITH TMP_VIEW AS (SELECT a, b, c, e, sum(f) i FROM {v}.numbers1 JOIN {v}.numbers2 on numbers1.d = numbers2.f WHERE b > 2 GROUP BY a, b, c, e ORDER BY b)''', '''SELECT a, sum(b), sum(e), max(i) FROM TMP_VIEW WHERE a > 3  and e < 8 GROUP BY a;'''
            , ignoreOrder=True, profilingResponse= [('SELECT * FROM NATIVE.NUMBERS1 WHERE 2 < B', ), ('SELECT E, F FROM NATIVE.NUMBERS2', ), ('SELECT * FROM NATIVE.NUMBERS1 WHERE 2 < B', ), ('SELECT E, F FROM NATIVE.NUMBERS2', )])
        self.createAndTestWithClause('''WITH TMP_VIEW AS (SELECT a, b, c, e, sum(f) i FROM {v}.numbers1 JOIN {v}.numbers2 on numbers1.d = numbers2.f WHERE b > 2 GROUP BY a, b, c, e ORDER BY b)''', '''SELECT a, b, e, i FROM TMP_VIEW WHERE a > 3  and e < 8 ORDER BY a, b, e, i;'''
            , profilingResponse= [('SELECT * FROM NATIVE.NUMBERS1 WHERE 2 < B', ), ('SELECT E, F FROM NATIVE.NUMBERS2', ), ('SELECT * FROM NATIVE.NUMBERS1 WHERE 2 < B', ), ('SELECT E, F FROM NATIVE.NUMBERS2', )])
        self.createAndTestWithClause('''WITH TMP_VIEW AS (SELECT a, b, c, e, sum(f) i FROM {v}.numbers1 JOIN {v}.numbers2 on numbers1.d = numbers2.f WHERE b > 2 GROUP BY a, b, c, e ORDER BY b)''', '''SELECT a, b, e, i FROM TMP_VIEW WHERE a > 3 ORDER BY a, b, e, i;'''
            , profilingResponse= [('SELECT * FROM NATIVE.NUMBERS1 WHERE 2 < B', ), ('SELECT E, F FROM NATIVE.NUMBERS2', ), ('SELECT * FROM NATIVE.NUMBERS1 WHERE 2 < B', ), ('SELECT E, F FROM NATIVE.NUMBERS2', )])
        self.createAndTestWithClause('''WITH TMP_VIEW AS (SELECT a, b, c, e, sum(f) i FROM {v}.numbers1 JOIN {v}.numbers2 on numbers1.d = numbers2.f WHERE b > 2 GROUP BY a, b, c, e ORDER BY b)''', '''SELECT a, b, e, i FROM TMP_VIEW ORDER BY a, b, e, i;'''
            , profilingResponse= [('SELECT * FROM NATIVE.NUMBERS1 WHERE 2 < B', ), ('SELECT E, F FROM NATIVE.NUMBERS2', ), ('SELECT * FROM NATIVE.NUMBERS1 WHERE 2 < B', ), ('SELECT E, F FROM NATIVE.NUMBERS2', )])
        self.createAndTestWithClause('''WITH TMP_VIEW AS (SELECT a, b, c, e, sum(f) i FROM {v}.numbers1 JOIN {v}.numbers2 on numbers1.d = numbers2.f WHERE b > 2 GROUP BY a, b, c, e ORDER BY b)''', '''SELECT a, b, e FROM TMP_VIEW ORDER BY a, b, e;'''
            , profilingResponse= [('SELECT * FROM NATIVE.NUMBERS1 WHERE 2 < B', ), ('SELECT E, F FROM NATIVE.NUMBERS2', ), ('SELECT * FROM NATIVE.NUMBERS1 WHERE 2 < B', ), ('SELECT E, F FROM NATIVE.NUMBERS2', )])
        self.createAndTestWithClause('''WITH TMP_VIEW AS (SELECT a, b, c, e, sum(f) i FROM {v}.numbers1 JOIN {v}.numbers2 on numbers1.d = numbers2.f WHERE b > 2 GROUP BY a, b, c, e ORDER BY b)''', '''SELECT a, b FROM TMP_VIEW ORDER BY a, b;'''
            , profilingResponse= [('SELECT * FROM NATIVE.NUMBERS1 WHERE 2 < B', ), ('SELECT E, F FROM NATIVE.NUMBERS2', ), ('SELECT * FROM NATIVE.NUMBERS1 WHERE 2 < B', ), ('SELECT E, F FROM NATIVE.NUMBERS2', )])
        self.createAndTestWithClause('''WITH TMP_VIEW AS (SELECT a, b, c, e, sum(f) i FROM {v}.numbers1 JOIN {v}.numbers2 on numbers1.d = numbers2.f WHERE b > 2 GROUP BY a, b, c, e ORDER BY b)''', '''SELECT a FROM TMP_VIEW ORDER BY a;'''
            , profilingResponse= [('SELECT * FROM NATIVE.NUMBERS1 WHERE 2 < B', ), ('SELECT E, F FROM NATIVE.NUMBERS2', ), ('SELECT * FROM NATIVE.NUMBERS1 WHERE 2 < B', ), ('SELECT E, F FROM NATIVE.NUMBERS2', )])
        self.createAndTestWithClause('''WITH TMP_VIEW AS (SELECT a, b, c, e, sum(f) i FROM {v}.numbers1 JOIN {v}.numbers2 on numbers1.d = numbers2.f WHERE b > 2 GROUP BY a, b, c, e ORDER BY b)''', '''SELECT * FROM TMP_VIEW ORDER BY a, b, c, e, i;'''
            , profilingResponse= [('SELECT * FROM NATIVE.NUMBERS1 WHERE 2 < B', ), ('SELECT E, F FROM NATIVE.NUMBERS2', ), ('SELECT * FROM NATIVE.NUMBERS1 WHERE 2 < B', ), ('SELECT E, F FROM NATIVE.NUMBERS2', )])

    def testDifferentWithClauses(self):
        self.createAndTestWithClause('''WITH TMP_VIEW AS (SELECT a, b, c FROM {v}.numbers1 WHERE b > 2 GROUP BY a, b, c ORDER BY b)''', '''SELECT a, sum(b) x FROM TMP_VIEW WHERE a > 3 GROUP BY a ORDER BY a, x;'''
        , ignoreOrder=True, profilingResponse= [('SELECT A, B, C FROM NATIVE.NUMBERS1 WHERE 2 < B GROUP BY A, B, C ORDER BY B', ),('SELECT A, B, C FROM NATIVE.NUMBERS1 WHERE 2 < B GROUP BY A, B, C ORDER BY B', )])
        self.createAndTestWithClause('''WITH TMP_VIEW AS (SELECT a, b, c FROM {v}.numbers1 WHERE b > 2 GROUP BY a, b, c ORDER BY b)''','''SELECT a, b FROM TMP_VIEW WHERE a > 3 ORDER BY a, b;'''
            , profilingResponse= [('SELECT A, B, C FROM NATIVE.NUMBERS1 WHERE 2 < B GROUP BY A, B, C ORDER BY B', ), ('SELECT A, B, C FROM NATIVE.NUMBERS1 WHERE 2 < B GROUP BY A, B, C ORDER BY B', )])
        self.createAndTestWithClause('''WITH TMP_VIEW AS (SELECT a, b, c FROM {v}.numbers1 WHERE b > 2 GROUP BY a, b, c ORDER BY b)''', '''SELECT a FROM TMP_VIEW WHERE a > 3 ORDER BY a;'''
            , profilingResponse= [('SELECT A, B, C FROM NATIVE.NUMBERS1 WHERE 2 < B GROUP BY A, B, C ORDER BY B', ), ('SELECT A, B, C FROM NATIVE.NUMBERS1 WHERE 2 < B GROUP BY A, B, C ORDER BY B', )])
        self.createAndTestWithClause('''WITH TMP_VIEW AS (SELECT a, b, c FROM {v}.numbers1 WHERE b > 2 GROUP BY a, b, c ORDER BY b)''', '''SELECT a FROM TMP_VIEW ORDER BY a;'''
            , profilingResponse= [('SELECT A, B, C FROM NATIVE.NUMBERS1 WHERE 2 < B GROUP BY A, B, C ORDER BY B', ), ('SELECT A, B, C FROM NATIVE.NUMBERS1 WHERE 2 < B GROUP BY A, B, C ORDER BY B', )])
        self.createAndTestWithClause('''WITH TMP_VIEW AS (SELECT a, b, c FROM {v}.numbers1 WHERE b > 2 GROUP BY a, b, c ORDER BY b)''', '''SELECT * FROM TMP_VIEW ORDER BY a, b, c;'''
            , profilingResponse= [('SELECT A, B, C FROM NATIVE.NUMBERS1 WHERE 2 < B GROUP BY A, B, C ORDER BY B', ), ('SELECT A, B, C FROM NATIVE.NUMBERS1 WHERE 2 < B GROUP BY A, B, C ORDER BY B', )])
        self.createAndTestWithClause('''WITH TMP_VIEW AS (SELECT a, b, c FROM {v}.numbers1 WHERE b > 2 GROUP BY a, b, c)''', '''SELECT a, sum(b) x FROM TMP_VIEW WHERE a > 3 GROUP BY a ORDER BY a, x;'''
            , ignoreOrder=True, profilingResponse= [('SELECT A, B, C FROM NATIVE.NUMBERS1 WHERE 2 < B GROUP BY A, B, C', ), ('SELECT A, B, C FROM NATIVE.NUMBERS1 WHERE 2 < B GROUP BY A, B, C', )])
        self.createAndTestWithClause('''WITH TMP_VIEW AS (SELECT a, b, c FROM {v}.numbers1 WHERE b > 2)''', '''SELECT a, sum(b) x FROM TMP_VIEW WHERE a > 3 GROUP BY a ORDER BY a, x;'''
            , ignoreOrder=True, profilingResponse= [('SELECT A, B, C FROM NATIVE.NUMBERS1 WHERE 2 < B', ), ('SELECT A, B, C FROM NATIVE.NUMBERS1 WHERE 2 < B', )])
        self.createAndTestWithClause('''WITH TMP_VIEW AS (SELECT a, b, c FROM {v}.numbers1)''', '''SELECT a, sum(b) x FROM TMP_VIEW WHERE a > 3 GROUP BY a ORDER BY a, x;'''
            , ignoreOrder=True, profilingResponse= [('SELECT A, B, C FROM NATIVE.NUMBERS1', ), ('SELECT A, B, C FROM NATIVE.NUMBERS1', )])
        self.createAndTestWithClause('''WITH TMP_VIEW AS (SELECT * FROM {v}.numbers1)''', '''SELECT a, sum(b) x FROM TMP_VIEW WHERE a > 3 GROUP BY a ORDER BY a, x;'''
            , ignoreOrder=True, profilingResponse= [('SELECT * FROM NATIVE.NUMBERS1', ), ('SELECT * FROM NATIVE.NUMBERS1', )])

    def testDifferentJoins(self):
        self.createAndTestWithClause('''WITH TMP_VIEW AS (SELECT a, b, c, e, sum(f) i FROM {v}.numbers1 JOIN NATIVE.numbers2 on numbers1.d = numbers2.f WHERE b > 2 GROUP BY a, b, c, e ORDER BY b)''', '''SELECT a, sum(b), sum(e), max(i) FROM TMP_VIEW WHERE a > 3  and e < 8 GROUP BY a ORDER BY a;'''
            , ignoreOrder=True, profilingResponse= [('SELECT * FROM NATIVE.NUMBERS1 WHERE 2 < B', ), ('SELECT * FROM NATIVE.NUMBERS1 WHERE 2 < B', )])
        self.query("DROP VIEW IF EXISTS NATIVE.FILTER_VIEW1_NATIVE")
        self.query("DROP VIEW IF EXISTS NATIVE.FILTER_VIEW1_VS1")
        self.query("CREATE VIEW NATIVE.FILTER_VIEW1_NATIVE AS SELECT E,F,G,H FROM NATIVE.numbers2 WHERE G > 4")
        self.query("CREATE VIEW NATIVE.FILTER_VIEW1_VS1 AS SELECT E,F,G,H FROM VS1.numbers2 WHERE G > 4")
        self.createAndTestWithClause('''WITH TMP_VIEW AS (SELECT a, b, c, e, sum(f) i FROM {v}.numbers1 JOIN NATIVE.FILTER_VIEW1_NATIVE on numbers1.d = FILTER_VIEW1_NATIVE.f WHERE b > 2 GROUP BY a, b, c, e ORDER BY b)''', '''SELECT a, sum(b), sum(e), max(i) FROM TMP_VIEW WHERE a > 3  and e < 8 GROUP BY a ORDER BY a;'''
            , ignoreOrder=True, profilingResponse= [('SELECT * FROM NATIVE.NUMBERS1', ), ('SELECT * FROM NATIVE.NUMBERS1 WHERE 2 < B', )])
        self.createAndTestWithClause('''WITH TMP_VIEW AS (SELECT a, b, c, e, sum(f) i FROM {v}.numbers1 JOIN NATIVE.FILTER_VIEW1_VS1 on numbers1.d = FILTER_VIEW1_VS1.f WHERE b > 2 GROUP BY a, b, c, e ORDER BY b)''', '''SELECT a, sum(b), sum(e), max(i) FROM TMP_VIEW WHERE a > 3  and e < 8 GROUP BY a ORDER BY a;'''
            , ignoreOrder=True, profilingResponse= [('SELECT * FROM NATIVE.NUMBERS1', ), ('SELECT E, F, G FROM NATIVE.NUMBERS2', ), ('SELECT * FROM NATIVE.NUMBERS1 WHERE 2 < B', ), ('SELECT * FROM NATIVE.NUMBERS2 WHERE 4 < G', )])

if __name__ == '__main__':
    udf.main()

