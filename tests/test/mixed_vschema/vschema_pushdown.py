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

#TODO Consolidate testAggregate, testScalar, etc. with explain tests

#@unittest.skip("skipped test")
class PushdownTest(VSchemaTest):
    
    setupDone = False

    def setUp(self):
        # TODO This is another ugly workaround for the problem that the framework doesn't offer us a query in classmethod setUpClass. Rewrite!
        if self.__class__.setupDone:
            self.query(''' CLOSE SCHEMA ''')
            return

        self.createJdbcAdapter()
        self.createNative()
        self.commit()  # We have to commit, otherwise the adapter won't see these tables
        self.createVirtualSchemaJdbc("VS1", "NATIVE", "ADAPTER.JDBC_ADAPTER", True)
        self.commit()
        self.query(''' CLOSE SCHEMA ''')
        self.__class__.setupDone = True

    def testProjection(self):
        # Single Group
        self.compareWithNativeExtended('''
            SELECT a, c FROM {v}.T;
        ''', ignoreOrder=True, explainResponse="SELECT A, C")
        self.compareWithNativeExtended('''
            SELECT b, c FROM {v}.T;
        ''', ignoreOrder=True, explainResponse="SELECT B, C")
        self.compareWithNativeExtended('''
            SELECT a FROM {v}.T;
        ''', ignoreOrder=True, explainResponse="SELECT A")
        # Column only in filter. To be executed with and without fiter pushdown.
        #  If no filter-pushdown: lookup a requires being updated, and a needs to be included.
        #  Otherwise no need for a in projection
        self.compareWithNativeExtended('''
            SELECT c FROM {v}.t WHERE a=1;
        ''', ignoreOrder=True, explainResponse="SELECT C FROM NATIVE.T WHERE A = 1")
        # Column only in order-by
        self.compareWithNativeExtended('''
            SELECT b, c, a FROM {v}.t ORDER BY a ;
        ''', partialOrder=2, explainResponse="SELECT * FROM NATIVE.T ORDER BY A")
        # Column only in analytical query
        self.compareWithNativeExtended('''
            SELECT k, v1, COUNT(*) OVER(PARTITION BY k ORDER BY v1) AS COUNT FROM {v}.g;
        ''', partialOrder=1, explainResponse="SELECT K, V1 FROM NATIVE.G")
        # Column only in on-clause of join
        self.compareWithNativeExtended('''
            SELECT vt.a FROM {v}.t vt JOIN {n}.t nt ON vt.c = nt.c;
        ''', ignoreOrder=True, explainResponse="SELECT A, C FROM NATIVE.T")
        # Empty Projection special case (no columns required)
        self.compareWithNativeExtended('''
            SELECT count(*) FROM {v}.t;
        ''', ignoreOrder=True, explainResponse="SELECT COUNT(*) FROM NATIVE.T")
        self.compareWithNativeExtended('''
            SELECT t1.a FROM {v}.t t1, {v}.t t2;
        ''', ignoreOrder=True, explainResponse=["SELECT A FROM NATIVE.T", "SELECT true FROM NATIVE.T"])
        self.compareWithNativeExtended('''
            SELECT true FROM {v}.t;
        ''', ignoreOrder=True, explainResponse="SELECT true FROM NATIVE.T")

    def testAliases(self):
        # Table Aliases
        self.compareWithNativeExtended('''
            SELECT a FROM {v}.t t1
        ''', ignoreOrder=True, explainResponse="SELECT A FROM NATIVE.T")
        self.compareWithNativeExtended('''
            SELECT t1.a FROM {v}.t t1, {v}.t t2
        ''', ignoreOrder=True, explainResponse=["SELECT A FROM NATIVE.T", "SELECT true FROM NATIVE.T"])
        self.compareWithNativeExtended('''
            SELECT t1.a FROM (SELECT * FROM {v}.t t1) as t1
        ''', ignoreOrder=True, explainResponse="SELECT A FROM NATIVE.T")
        self.compareWithNativeExtended('''
            SELECT t1.a FROM {v}.t t1, {v}.t t2
        ''', ignoreOrder=True, explainResponse=["SELECT A FROM NATIVE.T", "SELECT true FROM NATIVE.T"])
        
        # Column Aliases
        self.compareWithNativeExtended('''
            SELECT (a+1) a1 FROM {v}.t
        ''', ignoreOrder=True, explainResponse="SELECT (A + 1) FROM NATIVE.T")
        # in subselect: must work for selectlist and projection pushdown!
        self.compareWithNativeExtended('''
            SELECT a1 FROM (SELECT a a1 FROM {v}.t ORDER BY false)
        ''', ignoreOrder=True, explainResponse="SELECT A FROM NATIVE.T ORDER BY false")
        # with aggregation
        self.compareWithNativeExtended('''
            SELECT sum(a) AS suma FROM {v}.t
        ''', ignoreOrder=True, explainResponse="SELECT SUM(A) FROM NATIVE.T")
        
    def testJoins(self):
        self.compareWithNativeExtended('''
            select t1.a FROM {v}.t t1 join {v}.t t2 on t1.b=t2.b
        ''', ignoreOrder=True, explainResponse=["SELECT A, B FROM NATIVE.T", "SELECT B FROM NATIVE.T"])
        # with Projection
        self.compareWithNativeExtended('''
            select t1.a FROM {v}.t t1, {v}.t t2 where t1.c = t2.c
        ''', ignoreOrder=True, explainResponse=["SELECT A, C FROM NATIVE.T", "SELECT C FROM NATIVE.T"])
        # with local filter
        self.compareWithNativeExtended('''
            select t1.a FROM {v}.t t1 join {v}.t t2 on t1.b=t2.b and t1.a=2
        ''', ignoreOrder=True, explainResponse=["SELECT A, B FROM NATIVE.T", "SELECT B FROM NATIVE.T"])
        # with global filter
        self.compareWithNativeExtended('''
            select t1.a FROM {v}.t t1 join {v}.t t2 on t1.c=t2.c and t1.b<t2.b
        ''', ignoreOrder=True, explainResponse=["SELECT * FROM NATIVE.T","SELECT B, C FROM NATIVE.T"])
        # with non-sql92 syntax
        self.compareWithNativeExtended('''
            select t1.a FROM {v}.t t1, {v}.t t2 where t1.c=t2.c and t1.b<t2.b
        ''', ignoreOrder=True, explainResponse=["SELECT * FROM NATIVE.T", "SELECT B, C FROM NATIVE.T"])
        # equi join with using
        self.compareWithNativeExtended('''
            select * FROM {v}.t t1 join {v}.t t2 using (a)
        ''', ignoreOrder=True, explainResponse=["SELECT * FROM NATIVE.T", "SELECT * FROM NATIVE.T"])
        self.compareWithNativeExtended('''
            select t1.a FROM {v}.t t1 join {v}.g t2 on t1.a=t2.k join {v}.t_nulls t3 on t1.a = t3.a where t1.c < 3 and t3.a < 2 and t2.v1 < 4
        ''', ignoreOrder=True, explainResponse=["SELECT A FROM NATIVE.T WHERE C < 3","SELECT K FROM NATIVE.G WHERE V1 < 4","SELECT A FROM NATIVE.T_NULLS WHERE A < 2"])


        # Outer Join
        self.compareWithNativeExtended('''
            select * FROM {v}.t t1 left join (select * FROM {v}.t where a=1) t2 on t1.a=t2.a
        ''', ignoreOrder=True, explainResponse=["SELECT * FROM NATIVE.T", "SELECT * FROM NATIVE.T"])
        # outer join with coalesce (should not contain null rows, i.e. no filterpushdown and filter nulls after the join)
        self.compareWithNativeExtended('''
            select * FROM {v}.t t1 left join {v}.t t2 on t1.a=t2.a where coalesce(t2.a, 1) = 1
        ''', ignoreOrder=True, explainResponse=["SELECT * FROM NATIVE.T", "SELECT * FROM NATIVE.T"])
        # same with other syntax
        self.compareWithNativeExtended('''
        select * FROM {v}.t t1 left join {v}.t t2 on t1.a=t2.a and coalesce(t2.a, 1) = 1
        ''', ignoreOrder=True, explainResponse=["SELECT * FROM NATIVE.T", "SELECT * FROM NATIVE.T"])
        # This SHOULD contain null rows, which are produced by outer join
        self.compareWithNativeExtended('''
            select * FROM {v}.t t1 left join (select * FROM {v}.t where coalesce(a, 1) = 1) t2 on t1.a=t2.a
        ''', ignoreOrder=True, explainResponse=["SELECT * FROM NATIVE.T", "SELECT * FROM NATIVE.T"])
        
        # Cross Join
        self.compareWithNativeExtended('''
            select t1.a FROM {v}.t t1, {v}.t t2
        ''', ignoreOrder=True, explainResponse=["SELECT A FROM NATIVE.T", "SELECT true FROM NATIVE.T"])
        self.compareWithNativeExtended('''
            select * FROM {v}.t t1 cross join {v}.t t2 where t2.a!=1
        ''', ignoreOrder=True, explainResponse=["SELECT * FROM NATIVE.T", "SELECT * FROM NATIVE.T WHERE A != 1"])
        self.compareWithNativeExtended('''
            select * FROM {v}.t t1, {v}.t t2 where t2.a!=1
        ''', ignoreOrder=True, explainResponse=["SELECT * FROM NATIVE.T", "SELECT * FROM NATIVE.T WHERE A != 1"])
        # cross join via equi join syntax
        self.compareWithNativeExtended('''
            select * FROM {v}.t t1 inner join {v}.t t2 on t2.a!=1
        ''', ignoreOrder=True, explainResponse=["SELECT * FROM NATIVE.T", "SELECT * FROM NATIVE.T"])
        
        # Multi Join Conditions
        self.compareWithNativeExtended('''
            select a, t2.c+1 FROM {v}.t t1 join {v}.t t2 using (a, b)
        ''', ignoreOrder=True, explainResponse=["SELECT A, B FROM NATIVE.T", "SELECT * FROM NATIVE.T"])
        self.compareWithNativeExtended('''
            select t1.a FROM {v}.t t1 join {v}.t t2 on t1.a = t2.a and t1.b = t2.b
        ''', ignoreOrder=True, explainResponse=["SELECT A, B FROM NATIVE.T", "SELECT A, B FROM NATIVE.T"])
        # same with other syntax
        self.compareWithNativeExtended('''
            select t1.a FROM {v}.t t1, {v}.t t2 where t1.a = t2.a and t1.b = t2.b
        ''', ignoreOrder=True, explainResponse=["SELECT A, B FROM NATIVE.T", "SELECT A, B FROM NATIVE.T"])
        
        # Equi Join using strange syntax
        self.compareWithNativeExtended('''
            select t1.a FROM {v}.t t1 inner join {v}.t t2 on true where t1.b = t2.b
        ''', ignoreOrder=True, explainResponse=["SELECT A, B FROM NATIVE.T", "SELECT B FROM NATIVE.T"])
        
        # Join with native table
        self.compareWithNativeExtended('''
            select * from {v}.t vt join {n}.t nt on vt.a = nt.a where nt.a = 1
        ''', ignoreOrder=True, explainResponse="SELECT * FROM NATIVE.T")

    def testSelectListExpressions(self):
        # TODO Check partial ordering if possible
        self.compareWithNativeExtended('''
            select a+1, c from {v}.t order by c desc
        ''', partialOrder=1, explainResponse="SELECT (A + 1), C FROM NATIVE.T ORDER BY C DESC")
        self.compareWithNativeExtended('''
            select a+1, 3-c from {v}.t order by 3-c
        ''', partialOrder=1, explainResponse="SELECT (A + 1), (3 - C) FROM NATIVE.T ORDER BY (3 - C)")
        # with additional col in filter
        self.compareWithNativeExtended('''
            select a+1 from {v}.t where c=1.1
        ''', ignoreOrder=True, explainResponse="SELECT (A + 1) FROM NATIVE.T WHERE C = 1.1")
        # with additional col in ON-clause
        self.compareWithNativeExtended('''
            select t1.a+1 from {v}.t t1 join {v}.t t2 on t1.c = t2.c
        ''', ignoreOrder=True, explainResponse=["SELECT A, C FROM NATIVE.T", "SELECT C FROM NATIVE.T"])
        # With UDF
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SCALAR SCRIPT native.emit_dummy (a int) EMITS (a int, b varchar(100)) AS
            def run(ctx):
                ctx.emit(ctx[0],'a')
            /
        '''))
        self.compareWithNativeExtended('''
            select native.emit_dummy(a) from {v}.t
        ''', ignoreOrder=True, explainResponse="SELECT A FROM NATIVE.T")
        # With scalar-emit and select list expressions (map_script->getCachedExpressions())
        self.compareWithNativeExtended('''
            select native.emit_dummy(a), b from {v}.t
        ''', ignoreOrder=True, explainResponse="SELECT A, B FROM NATIVE.T")
        # With VarEmit
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SET SCRIPT native.var_emit_dummy (...) EMITS (...) AS
            def run(ctx):
                ctx.emit(ctx[0],'a')
            
            def default_output_columns(ctx):
                return ("a int, b varchar(100)")
            /
        '''))
        self.compareWithNativeExtended('''
            select native.var_emit_dummy(a) emits (a int, b varchar(100)) from {v}.t
        ''', ignoreOrder=True, explainResponse="SELECT A FROM NATIVE.T")
        # TODO Crashes! Extend getSingleCallEXScriptVMContainer to get types differently if this is a column lookup.
        # Add BUG in Jira if necessary
        # self.compareWithNativeSimple('''
        #    select native.var_emit_dummy(a) from t
        #''')
        
    def testPredicates(self):
        # =, !=, <, <=
        self.compareWithNativeExtended('''
            SELECT a=1, b FROM {v}.t WHERE a=1;
        ''', ignoreOrder=True, explainResponse="SELECT A = 1, B FROM NATIVE.T WHERE A = 1")
        self.compareWithNativeExtended('''
            SELECT a=1, b FROM {v}.t WHERE a=(a*2/2)
        ''', ignoreOrder=True, explainResponse="SELECT A = 1, B FROM NATIVE.T WHERE A = ((A * 2) / 2)")
        self.compareWithNativeExtended('''
            SELECT a=2, b FROM {v}.t WHERE a!=1
        ''', ignoreOrder=True, explainResponse="SELECT A = 2, B FROM NATIVE.T WHERE A != 1")
        self.compareWithNativeExtended('''
            SELECT a>2, b FROM {v}.t WHERE a>=2
        ''', ignoreOrder=True, explainResponse="SELECT 2 < A, B FROM NATIVE.T WHERE 2 <= A")
        self.compareWithNativeExtended('''
            SELECT a<2, b FROM {v}.t WHERE a<=2
        ''', ignoreOrder=True, explainResponse="SELECT A < 2, B FROM NATIVE.T WHERE A <= 2")
        # AND, OR, NOT
        self.compareWithNativeExtended('''
            SELECT a, c FROM {v}.t WHERE (a=1 AND c=1.1) OR (NOT a=2)
        ''', ignoreOrder=True, explainResponse="SELECT A, C FROM NATIVE.T WHERE ((A = 1 AND C = 1.1) OR NOT (A = 2))")
        # LIKE
        self.compareWithNativeExtended('''
            SELECT * FROM {v}.g WHERE v2 LIKE '%ne'
        ''', ignoreOrder=True, explainResponse="SELECT * FROM NATIVE.G WHERE V2 LIKE '%ne'")
        self.compareWithNativeExtended('''
            SELECT * FROM {v}.g WHERE v2 LIKE '%ne' ESCAPE '$'
        ''', ignoreOrder=True, explainResponse="SELECT * FROM NATIVE.G WHERE V2 LIKE '%ne' ESCAPE '$'")
        # REGEXP_LIKE
        self.compareWithNativeExtended('''
            SELECT * FROM {v}.g WHERE v2 REGEXP_LIKE '.*ne'
        ''', ignoreOrder=True, explainResponse="SELECT * FROM NATIVE.G WHERE V2 REGEXP_LIKE '.*ne'")
        # BETWEEN
        self.compareWithNativeExtended('''
            SELECT * FROM {v}.t WHERE a BETWEEN 2 AND 3
        ''', ignoreOrder=True, explainResponse="SELECT * FROM NATIVE.T WHERE A BETWEEN 2 AND 3")
        self.compareWithNativeExtended('''
            SELECT * FROM {v}.t WHERE c BETWEEN a AND a+1
        ''', ignoreOrder=True, explainResponse="SELECT * FROM NATIVE.T WHERE C BETWEEN A AND (A + 1)")
        # IN with const-list
        self.compareWithNativeExtended('''
            SELECT * FROM {v}.t WHERE c IN (1.1,3.3)
        ''', ignoreOrder=True, explainResponse="SELECT * FROM NATIVE.T WHERE C IN (1.1, 3.3)")
        # ... with const expressions
        self.compareWithNativeExtended('''
            SELECT * FROM {v}.t WHERE c IN (1.1*2, 1.1*3)
        ''', ignoreOrder=True, explainResponse="SELECT * FROM NATIVE.T WHERE C IN (2.2, 3.3)")
        # ... with non-const expression
        self.compareWithNativeExtended('''
            SELECT * FROM {v}.t WHERE c IN (1.1*a)
        ''', ignoreOrder=True, explainResponse="SELECT * FROM NATIVE.T WHERE C = (1.1 * A)")
        # ... with non-correlated subselect to native table
        self.compareWithNativeExtended('''
            SELECT * FROM {v}.t WHERE a IN (SELECT DISTINCT a FROM {n}.t ORDER BY a DESC LIMIT 2)
        ''', ignoreOrder=True, explainResponse="SELECT * FROM NATIVE.T")
        # ... with non-correlated subselect to virtual table
        self.compareWithNativeExtended('''
            SELECT * FROM {v}.t WHERE a IN (SELECT DISTINCT a FROM {v}.t ORDER BY a DESC LIMIT 2)
        ''', ignoreOrder=True, explainResponse=["SELECT * FROM NATIVE.T", "SELECT A FROM NATIVE.T GROUP BY A ORDER BY A DESC LIMIT 2"])
        # ... with correlated subselect (mixing what is native and what virtual)
        self.compareWithNativeExtended('''
            SELECT * FROM {v}.t t_out WHERE a IN (SELECT a FROM {n}.t WHERE t_out.c = a*1.1)
        ''', ignoreOrder=True, explainResponse="SELECT * FROM NATIVE.T")
        self.compareWithNativeExtended('''
            SELECT * FROM {v}.t t_out WHERE a IN (SELECT a FROM {v}.t WHERE t_out.c = a*1.1)
        ''', ignoreOrder=True, explainResponse=["SELECT * FROM NATIVE.T", "SELECT A FROM NATIVE.T"])
        self.compareWithNativeExtended('''
            SELECT * FROM {n}.t t_out WHERE a IN (SELECT a FROM {v}.t WHERE t_out.c = a*1.1)
        ''', ignoreOrder=True, explainResponse="SELECT A FROM NATIVE.T")
        
    def testOrderByLimit(self):
        # Order By Column
        # ... Column already in select list
        self.compareWithNativeExtended('''
            select b from {v}.t order by b
        ''', explainResponse="SELECT B FROM NATIVE.T ORDER BY B")
        # ... Column not in select list (should not be pushed down in projection, only in order-by)
        self.compareWithNativeExtended('''
            select b, c from {v}.t order by c
        ''', partialOrder=1, explainResponse="SELECT B, C FROM NATIVE.T ORDER BY C")
        # Order By Expression
        # ... also in selectlist
        self.compareWithNativeExtended('''
            select a+1 as a1, c from {v}.t order by a+1
        ''', partialOrder=0, explainResponse="SELECT (A + 1), C FROM NATIVE.T ORDER BY (A + 1)")
        # ... expressions not in selectlist (should not be pushed down in selectlist pushdown)
        self.compareWithNativeExtended('''
            select a+1, c+1 from {v}.t order by c+1
        ''', partialOrder=1, explainResponse="SELECT (A + 1), (C + 1) FROM NATIVE.T ORDER BY (C + 1)")
        # Order By Position
        self.compareWithNativeExtended('''
            select a+1, c from {v}.t order by 2 desc
        ''', partialOrder=1, explainResponse="SELECT (A + 1), C FROM NATIVE.T ORDER BY C DESC")
        # Order By Local
        self.compareWithNativeExtended('''
            select a+1 as a1, c from {v}.t order by local.a1
        ''', partialOrder=1, explainResponse="SELECT (A + 1), C FROM NATIVE.T ORDER BY (A + 1)")
        # Order By Options
        self.compareWithNativeExtended('''
            select b from {v}.t order by b asc
        ''', explainResponse="SELECT B FROM NATIVE.T ORDER BY B")
        self.compareWithNativeExtended('''
            select b from {v}.t order by b desc
        ''', explainResponse="SELECT B FROM NATIVE.T ORDER BY B DESC")
        self.compareWithNativeExtended('''
            select b from {v}.t order by b desc nulls first
        ''', explainResponse="SELECT B FROM NATIVE.T ORDER BY B DESC")
        self.compareWithNativeExtended('''
            select b from {v}.t order by b desc nulls last
        ''', explainResponse="SELECT B FROM NATIVE.T ORDER BY B DESC")
        # Limit
        self.compareWithNativeExtended('''
            select a from {v}.t order by b limit 2
        ''', ignoreOrder=True, explainResponse="SELECT A FROM NATIVE.T ORDER BY B LIMIT 2")
        self.compareWithNativeExtended('''
            select a from {v}.t limit 0
        ''', ignoreOrder=True, explainResponse="SELECT A FROM NATIVE.T")
        # ... with offset (2 syntax variants)
        self.compareWithNativeExtended('''
            select a from {v}.t order by c desc limit 1, 2
        ''', ignoreOrder=True, explainResponse="SELECT A FROM NATIVE.T ORDER BY C DESC LIMIT 2 OFFSET 1")
        self.compareWithNativeExtended('''
            select a from {v}.t order by c limit 2 offset 1
        ''', ignoreOrder=True, explainResponse="SELECT A FROM NATIVE.T ORDER BY C LIMIT 2 OFFSET 1")
        # With GroupBy
        self.compareWithNativeExtended('''
            select sum(a+2) from {v}.t group by c order by sum(a+2)
        ''', explainResponse="SELECT SUM((A + 2)) FROM NATIVE.T GROUP BY C ORDER BY SUM((A + 2))")
        # With Join
        self.compareWithNativeExtended('''
            select sum(t1.a+2) from {v}.t t1 join {n}.t t2 on t1.a=t2.a group by t2.c order by sum(t1.a+2)
        ''', explainResponse="SELECT A FROM NATIVE.T")
        self.compareWithNativeExtended('''
            select sum(t1.a+2) from {v}.t t1 join {v}.t t2 on t1.a=t2.a group by t2.c order by sum(t1.a+2)
        ''', explainResponse=["SELECT A FROM NATIVE.T", "SELECT A, C FROM NATIVE.T"])
        # NULLS FIRST/LAST combined with ASC/DESC
        self.compareWithNativeExtended('''
            select a from {v}.t_nulls order by b
        ''', explainResponse="SELECT A FROM NATIVE.T_NULLS ORDER BY B")
        self.compareWithNativeExtended('''
            select * from {v}.t_nulls order by b
        ''', explainResponse="SELECT * FROM NATIVE.T_NULLS ORDER BY B")
        self.compareWithNativeExtended('''
            select * from {v}.t_nulls order by b nulls last
        ''', explainResponse="SELECT * FROM NATIVE.T_NULLS ORDER BY B")
        self.compareWithNativeExtended('''
            select * from {v}.t_nulls order by b nulls first
        ''', explainResponse="SELECT * FROM NATIVE.T_NULLS ORDER BY B NULLS FIRST")
        self.compareWithNativeExtended('''
            select * from {v}.t_nulls order by b desc
        ''', explainResponse="SELECT * FROM NATIVE.T_NULLS ORDER BY B DESC")
        self.compareWithNativeExtended('''
            select * from {v}.t_nulls order by b desc nulls last
        ''', explainResponse="SELECT * FROM NATIVE.T_NULLS ORDER BY B DESC NULLS LAST")
        self.compareWithNativeExtended('''
            select * from {v}.t_nulls order by b desc nulls first
        ''', explainResponse="SELECT * FROM NATIVE.T_NULLS ORDER BY B DESC")

    def testFilter(self):
        # Filter in ON-clause
        # ... with projection
        self.compareWithNativeExtended('''
            select t1.a from {v}.t t1 join {v}.t t2 on t1.c=t2.c and t2.b='a'
        ''', explainResponse=["SELECT A, C FROM NATIVE.T", "SELECT B, C FROM NATIVE.T"])
        # ... with select list expression pushdown
        self.compareWithNativeExtended('''
            select t1.a+1 from {v}.t t1 join {v}.t t2 on t1.c=t2.c and t2.b='a'
        ''', explainResponse=["SELECT A, C FROM NATIVE.T", "SELECT B, C FROM NATIVE.T"])

    def testMixed(self):
        ''' Collection of queries mixing different sql features and special cases '''
        # Special Case: c*c is removed from select list, so only lookups in selectlist. Should still pushdown agg.
        self.compareWithNativeExtended('''
            SELECT count(a) FROM (
              SELECT a,c*c as x, sum(c) mysum FROM {v}.t GROUP BY a,c*c) subsel
            WHERE subsel.x<15 AND mysum>2;
        ''', explainResponse="SELECT A FROM NATIVE.T WHERE (C * C) < 15 GROUP BY (C * C), A HAVING 2 < SUM(C)")
        # ... same with b only in filter
        self.compareWithNativeExtended('''
            SELECT count(a) FROM (
              SELECT a,c*c as x, sum(c) mysum FROM {v}.t WHERE b!='f' GROUP BY a,c*c) subsel
            WHERE subsel.x<15 AND mysum>2;
        ''', explainResponse="SELECT A FROM NATIVE.T WHERE (B != 'f' AND (C * C) < 15) GROUP BY (C * C), A HAVING 2 < SUM(C)")
        # ... same with join
        self.compareWithNativeExtended('''
            SELECT count(a) FROM (
              SELECT t1.a,t1.c*t1.c as x, sum(t1.c) mysum FROM {v}.t t1 JOIN {v}.t t2 ON t1.b=t2.b GROUP BY t1.a,t1.c*t1.c) subsel
            WHERE subsel.x<15 AND mysum>2;
        ''', explainResponse=["SELECT * FROM NATIVE.T WHERE (C * C) < 15", "SELECT B FROM NATIVE.T"])
        self.compareWithNativeExtended('''
            SELECT * WITH INVALID PRIMARY KEY (a) from {v}.T;
        ''', ignoreOrder=True, explainResponse=["SELECT * FROM NATIVE.T", "SELECT A FROM NATIVE.T"])
        self.compareWithNativeExtended('''
            SELECT a WITH INVALID UNIQUE (a) from {v}.t;
        ''', ignoreOrder=True, explainResponse=["SELECT A FROM NATIVE.T", "SELECT A FROM NATIVE.T WHERE A IS NOT NULL"])

    def testAggregation(self):
        # Single Group
        self.compareWithNativeExtended('''
            select count(*), sum(a), min(c) from {v}.t;
        ''', explainResponse="SELECT COUNT(*), SUM(A), MIN(C) FROM NATIVE.T")

        # ... with filter
        self.compareWithNativeExtended('''
            select sum(a) from {v}.t where a>2;
        ''', explainResponse="SELECT SUM(A) FROM NATIVE.T WHERE 2 < A")

        # ... with having (one having expression in select list, the other not)
        self.compareWithNativeExtended('''
            select max(c) as maxa from {v}.t having sum(c) < 3.4 and max(c) = 3.3;
        ''', explainResponse="SELECT MAX(C) FROM NATIVE.T HAVING (SUM(C) < 3.4 AND MAX(C) = 3.3)")

        # Group By Column
        self.compareWithNativeExtended('''
            select a from {v}.t group by a;
        ''', explainResponse="SELECT A FROM NATIVE.T GROUP BY A")
        self.compareWithNativeExtended('''
            select c from {v}.t group by c;
        ''', explainResponse="SELECT C FROM NATIVE.T GROUP BY C")
        self.compareWithNativeExtended('''
            select a, sum(c) from {v}.t group by a;
        ''', explainResponse="SELECT A, SUM(C) FROM NATIVE.T GROUP BY A")
        self.compareWithNativeExtended('''
            select sum(c) from {v}.t group by a;
        ''', explainResponse="SELECT SUM(C) FROM NATIVE.T GROUP BY A")
        self.compareWithNativeExtended('''
            select sum(c) from {v}.t group by a having sum(c) != 1 or max(c) = 3.3;
        ''', explainResponse="SELECT SUM(C) FROM NATIVE.T GROUP BY A HAVING (SUM(C) != 1 OR MAX(C) = 3.3)")

        # Group By Columns
        self.compareWithNativeExtended('''
            select a+1, b, sum(c) from {v}.t group by (a+1,b);
        ''', explainResponse="SELECT (A + 1), B, SUM(C) FROM NATIVE.T GROUP BY (A + 1), B")

        # Group By Expression
        self.compareWithNativeExtended('''
            select a*2, count(*), max(b) from {v}.t group by a*2;
        ''', explainResponse="SELECT (A * 2), COUNT(*), MAX(B) FROM NATIVE.T GROUP BY (A * 2)")

        # Group By Empty
        self.compareWithNativeExtended('''
            select count(*) from {v}.t group by true;
        ''', explainResponse="SELECT COUNT(*) FROM NATIVE.T GROUP BY true")

        # Group By Tuple
        self.compareWithNativeExtended('''
            select a, b, count(*), sum(a) from {v}.t group by a, b;
        ''', explainResponse="SELECT A, B, COUNT(*), SUM(A) FROM NATIVE.T GROUP BY A, B")
        self.compareWithNativeExtended('''
            select a, b, count(*), sum(a) from {v}.t group by (a,b);
        ''', explainResponse="SELECT A, B, COUNT(*), SUM(A) FROM NATIVE.T GROUP BY A, B")
        self.compareWithNativeExtended('''
            select a, count(*), sum(a) from {v}.t group by (a);
        ''', explainResponse="SELECT A, COUNT(*), SUM(A) FROM NATIVE.T GROUP BY A")

        # Having
        self.compareWithNativeExtended('''
            select sum(a) from {v}.t having sum(a) > 1;
        ''', explainResponse="SELECT SUM(A) FROM NATIVE.T HAVING 1 < SUM(A)")
        self.compareWithNativeExtended('''
            select a, sum(a), count(*) from {v}.t group by a having sum(a) > 1;
        ''', explainResponse="SELECT A, SUM(A), COUNT(*) FROM NATIVE.T GROUP BY A HAVING 1 < SUM(A)")

        # ... with new column (to be considered in projection)
        self.compareWithNativeExtended('''
            select sum(a), sum(c) from {v}.t having sum(c) = 6.6;
        ''', explainResponse="SELECT SUM(A), SUM(C) FROM NATIVE.T HAVING SUM(C) = 6.6")

        # Aggregation On Join
        self.compareWithNativeExtended('''
            select sum(t1.a) from {v}.t t1, {v}.t t2 group by t1.a;
        ''', explainResponse=["SELECT A FROM NATIVE.T", "SELECT true FROM NATIVE.T"])

        # Nested Set Functions
        self.compareWithNativeExtended('''
            select sum(a+1) / count(a+1) from {v}.t;
        ''', explainResponse="SELECT (SUM((A + 1)) / COUNT((A + 1))) FROM NATIVE.T")

        # Set Functions
        self.compareWithNativeExtended('''
            select count(distinct a+1) from {v}.t;
        ''', explainResponse="SELECT COUNT(DISTINCT (A + 1)) FROM NATIVE.T")
        self.compareWithNativeExtended('''
            select sum(a+5) from {v}.t;
        ''', explainResponse="SELECT SUM((A + 5)) FROM NATIVE.T")
        self.compareWithNativeExtended('''
            select avg(a+1) from {v}.t;
        ''', explainResponse="SELECT AVG((A + 1)) FROM NATIVE.T")
        self.compareWithNativeExtended('''
            select percentile_cont(0.5) within group(order by a) from {v}.t;
        ''', explainResponse="SELECT A FROM NATIVE.T")

        # Multiple set functions
        self.compareWithNativeExtended('''
            select min(a), group_concat(b order by a desc), first_value(c) from {v}.t;
        ''', explainResponse="SELECT MIN(A), GROUP_CONCAT(B ORDER BY A DESC), FIRST_VALUE(C) FROM NATIVE.T")

        # Distinct: Gets translated to SELECT k FROM g GROUP BY k Should be pushed down!
        self.compareWithNativeExtended('''
            select distinct k from {v}.g;
        ''', explainResponse="SELECT K FROM NATIVE.G GROUP BY K")
        self.compareWithNativeExtended('''
            select distinct k+1 from {v}.g;
        ''', explainResponse="SELECT (K + 1) FROM NATIVE.G GROUP BY (K + 1)")

    def testScalarFunctions(self):
        # 1 arg
        self.compareWithNativeExtended('''
            select * from {v}.t where abs(a) = 1;
        ''', ignoreOrder=True, explainResponse="SELECT * FROM NATIVE.T WHERE ABS(A) = 1")

        # 2 arg
        self.compareWithNativeExtended('''
            select * from {v}.t where power(a,2) != 1;
        ''', ignoreOrder=True, explainResponse="SELECT * FROM NATIVE.T WHERE POWER(A, 2) != 1")
        self.compareWithNativeExtended('''
            select * from {v}.t where b||b||b = 'aaa';
        ''', ignoreOrder=True, explainResponse="SELECT * FROM NATIVE.T WHERE CONCAT(B, CONCAT(B, B)) = 'aaa'")
        self.compareWithNativeExtended('''
            select * from {v}.t where substr(b,0) = 'a';
        ''', ignoreOrder=True, explainResponse="SELECT * FROM NATIVE.T WHERE SUBSTR(B, 0) = 'a'")

        # 3 arg
        self.compareWithNativeExtended('''
            select * from {v}.t where substr(b,0,2) = 'a';
        ''', ignoreOrder=True, explainResponse="SELECT * FROM NATIVE.T WHERE SUBSTR(B, 0, 2) = 'a'")

        # 4 arg
        self.compareWithNativeExtended('''
            SELECT * from {v}.t_datetime where a != CONVERT_TZ(a,
                'Europe/Berlin',
                'UTC',
                'INVALID REJECT AMBIGUOUS REJECT');
        ''', ignoreOrder=True, explainResponse="SELECT * FROM NATIVE.T_DATETIME WHERE A != CONVERT_TZ(A, 'Europe/Berlin', 'UTC', 'INVALID REJECT AMBIGUOUS REJECT')")

        # variable arg
        self.compareWithNativeExtended('''
            select * from {v}.t where concat(b,b,b,b,b) = 'aaaaa';
        ''', ignoreOrder=True, explainResponse="SELECT * FROM NATIVE.T WHERE CONCAT(B, B, B, B, B) = 'aaaaa'")

        # cast
        self.compareWithNativeExtended('''
            select * from {v}.t where b != to_char(timestamp '2015-12-01 12:13:04.1234');
        ''', ignoreOrder=True, explainResponse="SELECT * FROM NATIVE.T WHERE B != '2015-12-01 12:13:04.123000'")

    def testMultiPushdown(self):
        self.createVirtualSchemaJdbc("VS2", "NATIVE", "ADAPTER.JDBC_ADAPTER", True)
        # Create an additional virtual schema using another adapter
        self.createJdbcAdapter(schemaName="ADAPTER2", adapterName="JDBC_ADAPTER")
        self.createVirtualSchemaJdbc("VS3", "NATIVE", "ADAPTER2.JDBC_ADAPTER", True)

        # 1 virtual schema, n virtual tables
        self.compareWithNativeExtended('''
            select * from {v}.t t1, {v}.t t2 where t1.a = t2.a;
        ''', ignoreOrder=True, explainResponse=["SELECT * FROM NATIVE.T", "SELECT * FROM NATIVE.T"])
        self.compareWithNativeExtended('''
            select * from {v}.t t1, {v}.t t2, {v}.t t3 where t1.a = t2.a and t2.a = t3.a;
        ''', ignoreOrder=True, explainResponse=["SELECT * FROM NATIVE.T", "SELECT * FROM NATIVE.T", "SELECT * FROM NATIVE.T"])

        # 1 adapter, n virtual schemas
        self.compareWithNativeExtended('''
            select * from {v}.t t1, {v2}.t t2 where t1.a = t2.a;
        ''', ignoreOrder=True, explainResponse=["SELECT * FROM NATIVE.T", "SELECT * FROM NATIVE.T"])
        self.compareWithNativeExtended('''
            select * from {v}.t t1, {v2}.t t2, {v}.t t3 where t1.a = t2.a and t2.a = t3.a;
        ''', ignoreOrder=True, explainResponse=["SELECT * FROM NATIVE.T", "SELECT * FROM NATIVE.T", "SELECT * FROM NATIVE.T"])

        # different adapters, different schemas
        self.compareWithNativeExtended('''
            select * from {v}.t t1, {v3}.t t2 where t1.a = t2.a;
        ''', ignoreOrder=True, explainResponse=["SELECT * FROM NATIVE.T", "SELECT * FROM NATIVE.T"])
        self.compareWithNativeExtended('''
            select * from {v}.t t1, (select a, b from {v3}.t) t2 where t1.a = t2.a;
        ''', ignoreOrder=True, explainResponse=["SELECT * FROM NATIVE.T", "SELECT A, B FROM NATIVE.T"])
        self.compareWithNativeExtended('''
            select * from {v}.t where a in (select distinct a from {v3}.t order by a desc limit 2);
        ''', ignoreOrder=True, explainResponse=["SELECT * FROM NATIVE.T", "SELECT A FROM NATIVE.T GROUP BY A ORDER BY A DESC LIMIT 2"])

    def testWithShortcuts(self):
        self.compareWithNativeExtended('''
            SELECT sum(c) suma, sum(c) sumb FROM {v}.t;
        ''', explainResponse="SELECT SUM(C), SUM(C) FROM NATIVE.T")
        self.compareWithNativeExtended('''
            SELECT (c+1) c1, b, (c+1) c2 FROM {v}.t;
        ''', explainResponse="SELECT (C + 1), B, (C + 1) FROM NATIVE.T")
        self.compareWithNativeExtended('''
            select sum(x.a), sum(x.a) from {v}.t x;
        ''', explainResponse=["SELECT SUM(A), SUM(A) FROM NATIVE.T"])
        self.compareWithNativeExtended('''
            select a * 2 from {v}.t WHERE (a * 2 > 0) AND (a * 2 > 1);
        ''', explainResponse=["SELECT (A * 2) FROM NATIVE.T WHERE (0 < (A * 2) AND 1 < (A * 2))"])
        self.compareWithNativeExtended('''
            select sum(x.a), sum(x.a) from {v}.t x, {v}.g y where x.a=y.v1 group by x.a;
        ''', explainResponse=["SELECT A FROM NATIVE.T", "SELECT V1 FROM NATIVE.G"])
        self.compareWithNativeExtended('''
            select x.a * 2, x.a * 2 from {v}.t x;
        ''', explainResponse=["SELECT (A * 2), (A * 2) FROM NATIVE.T"])
        self.compareWithNativeExtended('''
            select x.a * 2, x.a * 2 from {v}.t x, {v}.g y where x.a=y.v1 group by x.a;
        ''', explainResponse=["SELECT A FROM NATIVE.T", "SELECT V1 FROM NATIVE.G"])
        self.compareWithNativeExtended('''
            SELECT GROUP_CONCAT(A ORDER BY C), GROUP_CONCAT(A ORDER BY C) FROM {v}.t;
        ''', explainResponse="SELECT GROUP_CONCAT(A ORDER BY C), GROUP_CONCAT(A ORDER BY C) FROM NATIVE.T")
        self.compareWithNativeExtended('''
            select level, level from {v}.T_CONNECT start with val = 100 connect by val = prior parent;
        ''', explainResponse="SELECT * FROM NATIVE.T_CONNECT")
        self.compareWithNativeExtended('''
            select extract(hour from a-1) || ':' || extract(minute from a-1) from {v}.T_datetime;
        ''', explainResponse="SELECT CONCAT(EXTRACT(HOUR FROM (A - 1)), CONCAT(':', EXTRACT(MINUTE FROM (A - 1)))) FROM NATIVE.T_DATETIME")
        self.compareWithNativeExtended('''
            select concat(v2, v2), concat(v2, v2) from {v}.g;
        ''', explainResponse="SELECT CONCAT(V2, V2), CONCAT(V2, V2) FROM NATIVE.G")
        self.createFastAdapter(schemaName="ADAPTER", adapterName="FAST_ADAPTER")
        self.query('DROP VIRTUAL SCHEMA IF EXISTS FAST_VS1 CASCADE')
        self.query('CREATE VIRTUAL SCHEMA FAST_VS1 USING ADAPTER.FAST_ADAPTER')
        rows = self.query('''
            SELECT CONCAT(KEY, KEY), CONCAT(KEY,KEY) FROM FAST_VS1.DUMMY;
        ''')
        self.assertRowsEqual([('FOOFOO', 'FOOFOO')],rows)
        rows = self.query('''
            SELECT PUSHDOWN_SQL FROM (EXPLAIN VIRTUAL SELECT CONCAT(KEY, KEY), CONCAT(KEY,KEY) FROM FAST_VS1.DUMMY);
        ''')
        self.assertRowsEqual([("SELECT * FROM (VALUES ('FOO', 'BAR')) t",)],rows)

    def testWithAnalytical(self):
        self.compareWithNativeExtended('''
            SELECT k, v1, count(*) over (PARTITION BY k ORDER BY v1) AS COUNT FROM {v}.g;
        ''', explainResponse="SELECT K, V1 FROM NATIVE.G")
        self.compareWithNativeExtended('''
            SELECT k, v1, sum(v1) over (PARTITION BY k ORDER BY v1) AS SUM FROM {v}.g;
        ''', explainResponse="SELECT K, V1 FROM NATIVE.G")

        # and with order-by
        self.compareWithNativeExtended('''
            SELECT k, v1, sum(v1) over (PARTITION BY k ORDER BY v1) AS SUM FROM {v}.g order by k desc, sum;
        ''', explainResponse="SELECT K, V1 FROM NATIVE.G")

    def testWithConnectBy(self):
        self.compareWithNativeExtended('''
            SELECT val,
            PRIOR val PARENT_NAME,
            CONNECT_BY_ROOT val ROOT_NAME,
            SYS_CONNECT_BY_PATH(val, '/') "PATH"
            FROM {v}.T_CONNECT
            CONNECT BY PRIOR val = parent
            START WITH val = 100;
        ''', ignoreOrder=True, explainResponse="SELECT * FROM NATIVE.T_CONNECT")
        self.compareWithNativeExtended('''
            WITH tmp_view(x) AS (SELECT a+1 from {v}.T) SELECT x FROM tmp_view;
        ''', explainResponse="SELECT (A + 1) FROM NATIVE.T")

    def testWithPreferring(self):
        self.compareWithNativeExtended('''
            SELECT * FROM {v}.t
            PREFERRING (LOW ROUND(a/1000) PLUS HIGH ROUND(c/10))
            PRIOR TO (b = 'a');
        ''', explainResponse="SELECT * FROM NATIVE.T")

    def testWithWithClause(self):
        self.compareWithNativeExtended('''
            WITH tmp_view(x) AS (SELECT a+1 from {v}.T) SELECT x FROM tmp_view;
        ''', explainResponse="SELECT (A + 1) FROM NATIVE.T")
        self.compareWithNativeExtended('''
            WITH tmp_view2(x) AS (SELECT a+1 from {v}.T),
            tmp_view(y) AS (select a*a from {v}.T) SELECT x,y FROM tmp_view, tmp_view2 order by x, y;
        ''', explainResponse=["SELECT (A + 1) FROM NATIVE.T", "SELECT (A * A) FROM NATIVE.T"])
        self.compareWithNativeExtended('''
            With tmp_view(x) AS (SELECT a+1 from {v}.T),
            tmp_view2(y) as (SELECT x*x+c FROM tmp_view, {v}.T) SELECT y FROM tmp_view2 order by y;
        ''', explainResponse=["SELECT (A + 1) FROM NATIVE.T", "SELECT C FROM NATIVE.T"])
        self.compareWithNativeExtended('''
            (With tmp_view(x) AS (SELECT a+1 from {v}.T) select * FROM tmp_view)
            union (with tmp_view2(y) as (SELECT c FROM {v}.T) select * from tmp_view2);
        ''', explainResponse=["SELECT (A + 1) FROM NATIVE.T", "SELECT C FROM NATIVE.T"])
        self.query("DROP VIEW IF EXISTS NATIVE.PUSHDOWN_VIEW5_NATIVE")
        self.query("DROP VIEW IF EXISTS NATIVE.PUSHDOWN_VIEW5_VS1")
        self.query('''CREATE VIEW NATIVE.PUSHDOWN_VIEW5_NATIVE as (With tmp_view(x) AS (SELECT a+1 from NATIVE.T),
            tmp_view2(y) as (SELECT x*x+c FROM tmp_view, NATIVE.T)
            SELECT y FROM tmp_view2 order by y);''')
        self.query('''CREATE VIEW NATIVE.PUSHDOWN_VIEW5_VS1 as (With tmp_view(x) AS (SELECT a+1 from VS1.T),
            tmp_view2(y) as (SELECT x*x+c FROM tmp_view, VS1.T)
            SELECT y FROM tmp_view2 order by y);''')
        self.compareWithNativeExtended('''
            select * from NATIVE.PUSHDOWN_VIEW5_{v};
        ''', ignoreOrder=True, explainResponse=["SELECT (A + 1) FROM NATIVE.T", "SELECT C FROM NATIVE.T"])
        self.query("DROP VIEW IF EXISTS NATIVE.PUSHDOWN_VIEW5_NATIVE")
        self.query("DROP VIEW IF EXISTS NATIVE.PUSHDOWN_VIEW5_VS1")

    def testWithViews(self):
        self.query("DROP VIEW IF EXISTS NATIVE.PUSHDOWN_VIEW1_NATIVE")
        self.query("DROP VIEW IF EXISTS NATIVE.PUSHDOWN_VIEW1_VS1")
        self.query("DROP VIEW IF EXISTS NATIVE.PUSHDOWN_VIEW2_NATIVE")
        self.query("DROP VIEW IF EXISTS NATIVE.PUSHDOWN_VIEW2_VS1")
        self.query("DROP VIEW IF EXISTS NATIVE.PUSHDOWN_VIEW3_NATIVE")
        self.query("DROP VIEW IF EXISTS NATIVE.PUSHDOWN_VIEW3_VS1")
        self.query("DROP VIEW IF EXISTS NATIVE.PUSHDOWN_VIEW4_NATIVE")
        self.query("DROP VIEW IF EXISTS NATIVE.PUSHDOWN_VIEW4_VS1")
        self.query("CREATE VIEW NATIVE.PUSHDOWN_VIEW1_NATIVE AS SELECT A, K FROM NATIVE.T, NATIVE.G")
        self.query("CREATE VIEW NATIVE.PUSHDOWN_VIEW1_VS1 AS SELECT A, K FROM VS1.T, VS1.G")
        self.query("CREATE VIEW NATIVE.PUSHDOWN_VIEW2_NATIVE AS SELECT * FROM NATIVE.T")
        self.query("CREATE VIEW NATIVE.PUSHDOWN_VIEW2_VS1 AS SELECT * FROM VS1.T")
        self.query("CREATE VIEW NATIVE.PUSHDOWN_VIEW3_NATIVE AS SELECT SUM(A) X FROM NATIVE.T WHERE A < 4")
        self.query("CREATE VIEW NATIVE.PUSHDOWN_VIEW3_VS1 AS SELECT SUM(A) X FROM VS1.T  WHERE A < 4")
        self.query("CREATE VIEW NATIVE.PUSHDOWN_VIEW4_NATIVE AS SELECT SUM(A) X, AVG(C) Y FROM NATIVE.PUSHDOWN_VIEW2_NATIVE")
        self.query("CREATE VIEW NATIVE.PUSHDOWN_VIEW4_VS1 AS SELECT SUM(A) X, AVG(C) Y FROM NATIVE.PUSHDOWN_VIEW2_VS1")
        self.compareWithNativeExtended('''
            select * from NATIVE.PUSHDOWN_VIEW1_{v};
        ''', ignoreOrder=True, explainResponse=["SELECT A FROM NATIVE.T", "SELECT K FROM NATIVE.G"])
        self.compareWithNativeExtended('''
            select * from NATIVE.PUSHDOWN_VIEW2_{v};
        ''', ignoreOrder=True, explainResponse=["SELECT * FROM NATIVE.T"])
        self.compareWithNativeExtended('''
            select SUM(A) from NATIVE.PUSHDOWN_VIEW2_{v};
        ''', ignoreOrder=True, explainResponse=["SELECT * FROM NATIVE.T"])
        self.compareWithNativeExtended('''
            select * from NATIVE.PUSHDOWN_VIEW3_{v};
        ''', ignoreOrder=True, explainResponse=["SELECT SUM(A) FROM NATIVE.T WHERE A < 4"])
        self.compareWithNativeExtended('''
            select * from NATIVE.PUSHDOWN_VIEW4_{v};
        ''', ignoreOrder=True, explainResponse=["SELECT * FROM NATIVE.T"])
        # test view materialization
        self.compareWithNativeExtended('''
            SELECT * FROM (SELECT a, rownum FROM  NATIVE.PUSHDOWN_VIEW2_{v}) t1 join (SELECT * FROM  NATIVE.PUSHDOWN_VIEW2_{v}) t2 on t1.a = t2.a;
        ''', ignoreOrder=True)
        # test view elimination
        self.compareWithNativeExtended('''
            SELECT * FROM (SELECT * FROM  NATIVE.PUSHDOWN_VIEW2_{v}) t1 join (SELECT * FROM  NATIVE.PUSHDOWN_VIEW2_{v}) t2 on t1.a = t2.a;
        ''', ignoreOrder=True)
        self.query("DROP VIEW IF EXISTS NATIVE.PUSHDOWN_VIEW1_NATIVE")
        self.query("DROP VIEW IF EXISTS NATIVE.PUSHDOWN_VIEW1_VS1")
        self.query("DROP VIEW IF EXISTS NATIVE.PUSHDOWN_VIEW2_NATIVE")
        self.query("DROP VIEW IF EXISTS NATIVE.PUSHDOWN_VIEW2_VS1")
        self.query("DROP VIEW IF EXISTS NATIVE.PUSHDOWN_VIEW3_NATIVE")
        self.query("DROP VIEW IF EXISTS NATIVE.PUSHDOWN_VIEW3_VS1")
        self.query("DROP VIEW IF EXISTS NATIVE.PUSHDOWN_VIEW4_NATIVE")
        self.query("DROP VIEW IF EXISTS NATIVE.PUSHDOWN_VIEW4_VS1")

    def testWithSetOperations(self):
        self.compareWithNativeExtended('''
            (SELECT c from {v}.T) UNION (SELECT v1 FROM {v}.G);
        ''', explainResponse=["SELECT C FROM NATIVE.T", "SELECT V1 FROM NATIVE.G"])
        self.compareWithNativeExtended('''
            (SELECT * FROM VS1.T WHERE C in (SELECT c FROM (SELECT c from VS1.T) UNION ALL (SELECT v1 FROM VS1.G)));
        ''', explainResponse=["SELECT * FROM NATIVE.T", "SELECT C FROM NATIVE.T", "SELECT V1 FROM NATIVE.G"])
        self.compareWithNativeExtended('''
            (SELECT c from {v}.T) INTERSECT (SELECT v1 FROM {v}.G);
        ''', explainResponse=["SELECT C FROM NATIVE.T GROUP BY C", "SELECT V1 FROM NATIVE.G"])
        self.compareWithNativeExtended('''
            (SELECT c from {v}.T) MINUS (SELECT v1 FROM {v}.G);
        ''', explainResponse=["SELECT C FROM NATIVE.T GROUP BY C", "SELECT V1 FROM NATIVE.G"])
        self.compareWithNativeExtended('''
            (SELECT c from {v}.T) EXCEPT (SELECT v1 FROM {v}.G);
        ''', explainResponse=["SELECT C FROM NATIVE.T GROUP BY C","SELECT V1 FROM NATIVE.G"])

    def testWithExists(self):
        self.compareWithNativeExtended('''
            SELECT a FROM {v}.t WHERE EXISTS (SELECT * FROM {v}.g WHERE t.a=g.v1);
        ''', explainResponse=["SELECT A FROM NATIVE.T", "SELECT * FROM NATIVE.G"])
        self.compareWithNativeExtended('''
            SELECT a FROM {v}.t WHERE EXISTS (SELECT * FROM {v}.t);
        ''', explainResponse=["SELECT A FROM NATIVE.T", "SELECT * FROM NATIVE.T"])

class ExtendedFunctionTest(VSchemaTest):
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
        self.query(''' CLOSE SCHEMA ''')
        self.__class__.setupDone = True
    
    # def testOperatorsPar(self):
    #     procs = []
    #     procs.append(self.executor.submit(target=self.completeFunctionTestPar, args=('b||b', 't', 'CONCAT(B, B)')))
    #     procs.append(self.executor.submit(target=self.completeFunctionTestPar, args=('a + a', 't', 'A + A')))
    #     procs.append(self.executor.submit(target=self.completeFunctionTestPar, args=('a + 2', 't_datetime', 'A + 2')))
    #     procs.append(self.executor.submit(target=self.completeFunctionTestPar, args=('a - a', 't', 'A - A')))
    #     procs.append(self.executor.submit(target=self.completeFunctionTestPar, args=('a - a', 't_datetime', 'A - A')))
    #     procs.append(self.executor.submit(target=self.completeFunctionTestPar, args=('a - 2', 't_datetime', 'A - 2')))
    #     procs.append(self.executor.submit(target=self.completeFunctionTestPar, args=('2 * a', 't', '2 * A')))
    #     procs.append(self.executor.submit(target=self.completeFunctionTestPar, args=('a / 2', 't', 'A / 2')))
    #     concurrent.futures.wait(procs, NONE, ALL_COMPLETED)

    def testOperators(self):
        self.completeFunctionTest('+A', 't', 'A')
        self.completeFunctionTest('-A', 't', '-A')
        self.completeFunctionTest('B||B', 't', 'CONCAT(B, B)')
        self.completeFunctionTest('A + A', 't', 'A + A')
        self.completeFunctionTest('A + A', 't_interval', 'A + A')
        self.completeFunctionTest('A + 2', 't_datetime', 'A + 2')
        self.completeFunctionTest("A + INTERVAL '5' MONTH", 't_interval', "A + INTERVAL '+0-05' YEAR (2) TO MONTH")
        self.completeFunctionTest("A + INTERVAL '130' MONTH (3)", 't_interval', "A + INTERVAL '+10-10' YEAR (3) TO MONTH")
        self.completeFunctionTest("A + INTERVAL '27' YEAR", 't_interval', "A + INTERVAL '+27-00' YEAR (2) TO MONTH")
        self.completeFunctionTest("A + INTERVAL '2-1' YEAR TO MONTH", 't_interval', "A + INTERVAL '+2-01' YEAR (2) TO MONTH")
        self.completeFunctionTest("A + INTERVAL '100-1' YEAR(3) TO MONTH", 't_interval', "A + INTERVAL '+100-01' YEAR (3) TO MONTH")
        self.completeFunctionTest("B + INTERVAL '5' DAY", 't_interval', "B + INTERVAL '+5 00:00:00.000' DAY (2) TO SECOND (3)")
        self.completeFunctionTest("B + INTERVAL '100' HOUR(3)", 't_interval', "B + INTERVAL '+4 04:00:00.000' DAY (3) TO SECOND (3)")
        self.completeFunctionTest("B + INTERVAL '6' MINUTE", 't_interval', "B + INTERVAL '+0 00:06:00.000' DAY (2) TO SECOND (3)")
        self.completeFunctionTest("B + INTERVAL '1.99999' SECOND(2,2)", 't_interval', "B + INTERVAL '+0 00:00:02.000' DAY (2) TO SECOND (2)")
        self.completeFunctionTest("B + INTERVAL '10:20' HOUR TO MINUTE", 't_interval', "B + INTERVAL '+0 10:20:00.000' DAY (2) TO SECOND (3)")
        self.completeFunctionTest("B + INTERVAL '2 23:10:59' DAY TO SECOND", 't_interval', "B + INTERVAL '+2 23:10:59.000' DAY (2) TO SECOND (3)")
        self.completeFunctionTest("B + INTERVAL '23:10:59.123' HOUR(2) TO SECOND(3)", 't_interval', "B + INTERVAL '+0 23:10:59.123' DAY (2) TO SECOND (3)")
        self.completeFunctionTest('A - A', 't', 'A - A')
        self.completeFunctionTest('A - A', 't_interval', 'A - A')
        self.completeFunctionTest('A - A', 't_datetime', 'A - A')
        self.completeFunctionTest('A - 2', 't_datetime', 'A - 2')
        self.completeFunctionTest("A - INTERVAL '5' MONTH", 't_interval', "A - INTERVAL '+0-05' YEAR (2) TO MONTH")
        self.completeFunctionTest("B - INTERVAL '2 23:10:59' DAY TO SECOND", 't_interval', "B - INTERVAL '+2 23:10:59.000' DAY (2) TO SECOND (3)")
        self.completeFunctionTest('2 * A', 't', '2 * A')
        self.completeFunctionTest('2 * A', 't_interval', '2 * A')
        self.completeFunctionTest('A / 2', 't', 'A / 2')
        self.completeFunctionTest('A / 2', 't_interval', 'A / 2')

    def testPredicates(self):
        self.completeFunctionTest('(a = 1)', 't', 'A = 1')
        self.completeFunctionTest('(a = a)', 't', 'A = A')
        self.completeFunctionTest('(a != 1)', 't', 'A != 1')
        self.completeFunctionTest('(a < 1)', 't', 'A < 1')
        self.completeFunctionTest('(a <= 1)', 't', 'A <= 1')
        self.completeFunctionTest('(a > 1)', 't', '1 < A')
        self.completeFunctionTest('(a >= 1)', 't', '1 <= A')
        self.completeFunctionTest('NOT (a = 1)', 't', 'NOT (A = 1)')
        self.completeFunctionTest('(a > 1) AND (a / 2 > 3)', 't', '(1 < A AND 3 < (A / 2))')
        self.completeFunctionTest('(a = 1) OR (a < 1)', 't', '(A = 1 OR A < 1)')
        self.completeFunctionTest('a BETWEEN 1 AND 3', 't', 'A BETWEEN 1 AND 3')
        self.completeFunctionTest('a NOT BETWEEN 1 AND 3', 't', 'NOT (A BETWEEN 1 AND 3)')
        self.completeFunctionTest('a', 't', 'A IN (1, 2)', 'WHERE a IN (1,2)')
        self.completeFunctionTest('a', 't', 'NOT (A IN (1, 2))', 'WHERE a NOT IN (1,2)')
        self.completeFunctionTest('a', 't', 'WHERE (A = POWER(A, A) OR A = POWER(A, 2))', 'WHERE a IN (POWER(a,a), POWER(a,2))')
        self.completeFunctionTest('a', 't', 'NOT ((A = POWER(A, A) OR A = POWER(A, 2)))', 'WHERE a NOT IN (POWER(a,a), POWER(a,2))')
        self.completeFunctionTest('a', 't', '(A = POWER(A, A) OR A IN (1, 2))', 'WHERE a IN (POWER(a,a), 1, 2)')
        self.completeFunctionTest('a', 't', 'SELECT A FROM', 'WHERE a IN (SELECT 2 FROM (VALUES 2))')
        self.completeFunctionTest('a', 't', 'SELECT A FROM', 'WHERE a NOT IN (SELECT 2 FROM (VALUES 2))')
        self.completeFunctionTest('a', 't', 'A IS NULL', 'WHERE a IS NULL')
        self.completeFunctionTest('a', 't', 'A IS NOT NULL', 'WHERE a IS NOT NULL')
        self.completeFunctionTest('v2', 'g', "REGEXP_LIKE '.*t.*'", '''WHERE v2 REGEXP_LIKE '.*t.*' ''')
        self.completeFunctionTest('v2', 'g', "NOT (V2 REGEXP_LIKE '.*t.*')", '''WHERE v2 NOT REGEXP_LIKE '.*t.*' ''')
        self.completeFunctionTest('v2', 'g', "LIKE '%t%'", '''WHERE v2 LIKE '%t%' ''')
        self.completeFunctionTest('v2', 'g', "NOT (V2 LIKE '%t%')", '''WHERE v2 NOT LIKE '%t%' ''')
        self.completeFunctionTest('v2', 'g', "LIKE 'a%t%' ESCAPE 'a'", '''WHERE '%'||v2 LIKE 'a%t%' ESCAPE 'a' ''')
        self.completeFunctionTest('v2', 'g', "NOT (V2 LIKE 'a%t%' ESCAPE 'a')", '''WHERE v2 NOT LIKE 'a%t%' ESCAPE 'a' ''')
        self.completeFunctionTest('(a > 1) AND (a < 3)', 't', '1 < A AND A < 3')
        self.completeFunctionTest('(a >= 1) AND (a < 3)', 't', '1 <= A AND A < 3')
        self.completeFunctionTest('(a > 1) AND (a <= 3)', 't', '1 < A AND A <= 3')
        self.completeFunctionTest('(a >= 1) AND (a <= 3)', 't', 'A BETWEEN 1 AND 3')

    def testNumericFunctions(self):
        self.completeFunctionTest('ABS(a)', 't', 'ABS(A)')
        self.completeFunctionTest('ACOS(a)', 't', 'ACOS(A)', 'WHERE a = 1')
        self.completeFunctionTest('ASIN(a)', 't', 'ASIN(A)', 'WHERE a = 1')
        self.completeFunctionTest('ATAN(a)', 't', 'ATAN(A)')
        self.completeFunctionTest('ATAN2(a,a)', 't', 'ATAN2(A, A)')
        self.completeFunctionTest('CEIL(c)', 't', 'CEIL(C)')
        self.completeFunctionTest('CEILING(c)', 't', 'CEIL(C)')
        self.completeFunctionTest('COS(a)', 't', 'COS(A)')
        self.completeFunctionTest('COSH(a)', 't', 'COSH(A)')
        self.completeFunctionTest('COT(a)', 't', 'COT(A)')
        self.completeFunctionTest('DEGREES(a)', 't', 'DEGREES(A)')
        self.completeFunctionTest('DIV(A, C)', 't', 'DIV(A, C)')
        self.completeFunctionTest('EXP(a)', 't', 'EXP(A)')
        self.completeFunctionTest('FLOOR(c)', 't', 'FLOOR(C)')
        self.completeFunctionTest('GREATEST(a,a)', 't', 'GREATEST(A, A)')
        self.completeFunctionTest('LEAST(a,a)', 't', 'LEAST(A, A)')
        self.completeFunctionTest('LN(a)', 't', 'LN(A)')
        self.completeFunctionTest('LOG(3,a)', 't', 'LOG(3, A)')
        self.completeFunctionTest('LOG10(a)', 't', 'LOG(10, A)')
        self.completeFunctionTest('LOG2(a)', 't', 'LOG(2, A)')
        self.completeFunctionTest('MOD(a,a)', 't', 'MOD(A, A)')
        self.completeFunctionTest('c', 't', '3.141592653589793', 'WHERE c = PI()')
        self.completeFunctionTest('POWER(a,a)', 't', 'POWER(A, A)')
        self.completeFunctionTest('RADIANS(a)', 't', 'RADIANS(A)')
        self.completeFunctionTest('a', 't', 'RAND()', 'WHERE a = SIGN(RAND())')
        self.completeFunctionTest('a', 't', 'RAND()', 'WHERE a = SIGN(RANDOM())')
        self.completeFunctionTest('a', 't', 'RAND(1, 10)', 'WHERE a = SIGN(RAND(1, 10))')
        self.completeFunctionTest('a', 't', 'RAND(1, 10)', 'WHERE a = SIGN(RANDOM(1, 10))')
        self.completeFunctionTest('ROUND(c)', 't', 'ROUND(C)')
        self.completeFunctionTest('ROUND(c,1)', 't', 'ROUND(C, 1)')
        self.completeFunctionTest('SIGN(a)', 't', 'SIGN(A)')
        self.completeFunctionTest('SIN(a)', 't', 'SIN(A)')
        self.completeFunctionTest('SINH(a)', 't', 'SINH(A)')
        self.completeFunctionTest('SQRT(a)', 't', 'SQRT(A)')
        self.completeFunctionTest('TAN(a)', 't', 'TAN(A)')
        self.completeFunctionTest('TANH(a)', 't', 'TANH(A)')
        self.completeFunctionTest('TO_CHAR(c)', 't', 'TO_CHAR(C)')
        self.completeFunctionTest('''TO_CHAR(c, '000G000G000D000000MI')''', 't', '''TO_CHAR(C, '000G000G000D000000MI')''')
        self.completeFunctionTest('TO_NUMBER(TO_CHAR(c))', 't', 'TO_NUMBER(TO_CHAR(C))')
        self.completeFunctionTest('''TO_NUMBER(TO_CHAR(c, '99999.999'))''', 't', '''TO_NUMBER(TO_CHAR(C, '99999.999'))''')
        self.completeFunctionTest('TRUNC(c)', 't', 'TRUNC(C)')
        self.completeFunctionTest('TRUNCATE(c)', 't', 'TRUNC(C)')
        self.completeFunctionTest('TRUNC(c,1)', 't', 'TRUNC(C, 1)')
        self.completeFunctionTest('TRUNCATE(c,1)', 't', 'TRUNC(C, 1)')

    def testStringFunctions(self):
        self.completeFunctionTest('ASCII(b)', 't', 'ASCII(B)')
        self.completeFunctionTest('BIT_LENGTH(v2)', 'g', 'BIT_LENGTH(V2)')
        self.completeFunctionTest('CHARACTER_LENGTH(v2)', 'g', 'LENGTH(V2)')
        self.completeFunctionTest('CHR(a)', 't', 'CHR(A)')
        self.completeFunctionTest('CHAR(a)', 't', 'CHR(A)')
        self.completeFunctionTest('COLOGNE_PHONETIC(V2)', 'g', 'COLOGNE_PHONETIC(V2)')
        self.completeFunctionTest('CONCAT(V2)', 'g', 'CONCAT(V2)')
        self.completeFunctionTest('CONCAT(V2, V2)', 'g', 'CONCAT(V2, V2)')
        self.completeFunctionTest('DUMP(V2)', 'g', 'DUMP(V2)')
        self.completeFunctionTest('DUMP(V2, 16, 1, 1)', 'g', 'DUMP(V2, 16, 1, 1)')
        self.completeFunctionTest('EDIT_DISTANCE(V2, V2||V2)', 'g', 'EDIT_DISTANCE(V2, CONCAT(V2, V2))')
        self.completeFunctionTest("INSERT(V2, 1, 2, 'foo')", 'g', "INSERT(V2, 1, 2, 'foo')")
        self.completeFunctionTest("INSTR(V2, 't')", 'g', "INSTR(V2, 't')")
        self.completeFunctionTest("INSTR(V2||V2, 't', 2, 2)", 'g', "INSTR(CONCAT(V2, V2), 't', 2, 2)")
        self.completeFunctionTest("LCASE(V2)", 'g', "LOWER(V2)")
        self.completeFunctionTest("LEFT(V2, 1)", 'g', "SUBSTR(V2, 1, 1)")
        self.completeFunctionTest("LEFT(V2, NULL)", 'g', "SUBSTR(V2, 1, NULL)")
        self.completeFunctionTest("LEFT(V2, 100)", 'g', "SUBSTR(V2, 1, 100)")
        self.completeFunctionTest("LENGTH(V2)", 'g', "LENGTH(V2)")
        self.completeFunctionTest("LOCATE('t', V2)", 'g', "LOCATE('t', V2)")
        self.completeFunctionTest("LOCATE('t', V2||V2, 1)", 'g', "LOCATE('t', CONCAT(V2, V2), 1)")
        self.completeFunctionTest("LOWER(V2)", 'g', "LOWER(V2)")
        self.completeFunctionTest("LPAD(V2, 4000000)", 'g', "LPAD(V2, 4000000)")
        self.completeFunctionTest("LPAD(V2, 4000000, 'A')", 'g', "LPAD(V2, 4000000, 'A')")
        self.completeFunctionTest("LTRIM(' '||V2||' ')", 'g', "LTRIM(CONCAT(' ', CONCAT(V2, ' ')))")
        self.completeFunctionTest("LTRIM(' '||V2||'\n', ' \n')", 'g', "LTRIM(CONCAT(' ', CONCAT(V2, '\n')), ' \n')")
        self.completeFunctionTest("MID(V2, 2)", 'g', "SUBSTR(V2, 2)")
        self.completeFunctionTest("MID(V2, 2, 1)", 'g', "SUBSTR(V2, 2, 1)")
        self.completeFunctionTest("OCTET_LENGTH(V2)", 'g', "OCTET_LENGTH(V2)")
        self.completeFunctionTest("POSITION('t' in V2)", 'g', "INSTR(V2, 't')")
        self.completeFunctionTest("REGEXP_INSTR(V2, '%t')", 'g', "REGEXP_INSTR(V2, '%t')")
        self.completeFunctionTest("REGEXP_INSTR(V2, '%t', 1, 1)", 'g', "REGEXP_INSTR(V2, '%t', 1, 1)")
        self.completeFunctionTest("REGEXP_REPLACE(V2, '%t')", 'g', "REGEXP_REPLACE(V2, '%t')")
        self.completeFunctionTest("REGEXP_REPLACE(V2, '%t', 'X', 1, 1)", 'g', "REGEXP_REPLACE(V2, '%t', 'X', 1, 1)")
        self.completeFunctionTest("REGEXP_SUBSTR(V2, '%t')", 'g', "REGEXP_SUBSTR(V2, '%t')")
        self.completeFunctionTest("REGEXP_SUBSTR(V2, '%t', 1, 1)", 'g', "REGEXP_SUBSTR(V2, '%t', 1, 1)")
        self.completeFunctionTest("REPEAT(V2, 999)", 'g', "REPEAT(V2, 999)")
        self.completeFunctionTest("REPEAT(V2, NULL)", 'g', "REPEAT(V2, NULL)")
        self.completeFunctionTest("REPLACE(V2, 't')", 'g', "REPLACE(V2, 't', NULL)")
        self.completeFunctionTest("REPLACE(V2, 't', 'X')", 'g', "REPLACE(V2, 't', 'X')")
        self.completeFunctionTest("REVERSE(V2)", 'g', "REVERSE(V2)")
        self.completeFunctionTest("RIGHT(V2, 1)", 'g', "RIGHT(V2, 1)")
        self.completeFunctionTest("RIGHT(V2, NULL)", 'g', "RIGHT(V2, NULL)")
        self.completeFunctionTest("RPAD(V2, 4000000)", 'g', "RPAD(V2, 4000000)")
        self.completeFunctionTest("RPAD(V2, 4000000, 'A')", 'g', "RPAD(V2, 4000000, 'A')")
        self.completeFunctionTest("RTRIM(' '||V2||' ')", 'g', "RTRIM(CONCAT(' ', CONCAT(V2, ' ')))")
        self.completeFunctionTest("RTRIM(' '||V2||'\n', ' \n')", 'g', "RTRIM(CONCAT(' ', CONCAT(V2, '\n')), ' \n')")
        self.completeFunctionTest("SOUNDEX(V2)", 'g', "SOUNDEX(V2)")
        self.completeFunctionTest("V2", 'g', "WHERE V2 = '          '", 'WHERE v2=SPACE(10)')
        self.completeFunctionTest("V2", 'g', "WHERE V2 = '          ", 'WHERE v2=SPACE(2000000)')
        self.completeFunctionTest("SUBSTR(V2, 2)", 'g', "SUBSTR(V2, 2)")
        self.completeFunctionTest("SUBSTR(V2, 2, 1)", 'g', "SUBSTR(V2, 2, 1)")
        self.completeFunctionTest("SUBSTRING(V2 FROM 2)", 'g', "SUBSTR(V2, 2)")
        self.completeFunctionTest("SUBSTRING(V2 FROM 2 FOR 1)", 'g', "SUBSTR(V2, 2, 1)")
        self.completeFunctionTest('TO_CHAR(A)', 't_datetime', 'TO_CHAR(A)')
        self.completeFunctionTest("TO_CHAR(A, 'VW', 'NLS_DATE_LANGUAGE=GERMAN')", 't_datetime', "TO_CHAR(A, 'VW', 'NLS_DATE_LANGUAGE=GERMAN')")
        self.completeFunctionTest('TO_CHAR(a)', 't_interval', 'TO_CHAR(A)')
        self.completeFunctionTest('TO_NUMBER(TO_CHAR(c))', 't', 'TO_NUMBER(TO_CHAR(C))')
        self.completeFunctionTest('''TO_NUMBER(TO_CHAR(c, '99999.999'))''', 't', '''TO_NUMBER(TO_CHAR(C, '99999.999'))''')
        self.completeFunctionTest("TRANSLATE(V2, 't', 'X')", 'g', "TRANSLATE(V2, 't', 'X')")
        self.completeFunctionTest("TRIM(' '||V2||' ')", 'g', "TRIM(CONCAT(' ', CONCAT(V2, ' ')))")
        self.completeFunctionTest("TRIM(' '||V2||'\n', ' \n')", 'g', "TRIM(CONCAT(' ', CONCAT(V2, '\n')), ' \n')")
        self.completeFunctionTest("UCASE(V2)", 'g', "UPPER(V2)")
        self.completeFunctionTest("UNICODE(B)", 't', "UNICODE(B)")
        self.completeFunctionTest("UNICODECHR(A)", 't', "UNICODECHR(A)")
        self.completeFunctionTest("UPPER(V2)", 'g', "UPPER(V2)")

    def testDateTimeFunctions(self):
        self.completeFunctionTest("ADD_DAYS(A, 42)", 't_datetime', "ADD_DAYS(A, 42)")
        self.completeFunctionTest("ADD_HOURS(A, 42)", 't_datetime', "ADD_HOURS(A, 42)")
        self.completeFunctionTest("ADD_MINUTES(A, 42)", 't_datetime', "ADD_MINUTES(A, 42)")
        self.completeFunctionTest("ADD_MONTHS(A, 42)", 't_datetime', "ADD_MONTHS(A, 42)")
        self.completeFunctionTest("ADD_SECONDS(A, 42)", 't_datetime', "ADD_SECONDS(A, 42)")
        self.completeFunctionTest("ADD_WEEKS(A, 42)", 't_datetime', "ADD_WEEKS(A, 42)")
        self.completeFunctionTest("ADD_YEARS(A, 42)", 't_datetime', "ADD_YEARS(A, 42)")
        self.completeFunctionTest("CONVERT_TZ(A, 'NZ', 'EUROPE/BERLIN')", 't_datetime', "CONVERT_TZ(A, 'NZ', 'EUROPE/BERLIN')")
        self.completeFunctionTest("CONVERT_TZ(A, 'NZ', 'EUROPE/BERLIN', 'ENSURE REVERSIBILITY')", 't_datetime', "CONVERT_TZ(A, 'NZ', 'EUROPE/BERLIN', 'ENSURE REVERSIBILITY')")
        self.completeFunctionTest("A", 't_datetime', "CURRENT_DATE()", 'WHERE A < CURDATE()')
        self.completeFunctionTest("A", 't_datetime', "CURRENT_DATE()", 'WHERE A < CURRENT_DATE()')
        self.completeFunctionTest("A", 't_datetime', "CURRENT_TIMESTAMP()", 'WHERE A < CURRENT_TIMESTAMP()')
        self.completeFunctionTest("DATE_TRUNC('month', A)", 't_datetime', "DATE_TRUNC('month', A)")
        self.completeFunctionTest("DAY(A)", 't_datetime', "DAY(A)")
        self.completeFunctionTest("DAYS_BETWEEN(A, A)", 't_datetime', "DAYS_BETWEEN(A, A)")
        self.completeFunctionTest("V2", 'g', "DBTIMEZONE()", 'WHERE V2 > DBTIMEZONE()')
        self.completeFunctionTest("EXTRACT(MONTH FROM A)", 't_datetime', "EXTRACT(MONTH FROM A)")
        self.completeFunctionTest("EXTRACT(MONTH FROM A)", 't_interval', "EXTRACT(MONTH FROM A)")
        self.completeFunctionTest("HOURS_BETWEEN(A, A)", 't_datetime', "HOURS_BETWEEN(A, A)")
        self.completeFunctionTest("A", 't_datetime', "LOCALTIMESTAMP()", 'WHERE A < LOCALTIMESTAMP()')
        self.completeFunctionTest("MINUTE(A)", 't_datetime', "MINUTE(A)")
        self.completeFunctionTest("MINUTES_BETWEEN(A, A)", 't_datetime', "MINUTES_BETWEEN(A, A)")
        self.completeFunctionTest("MONTH(A)", 't_datetime', "MONTH(A)")
        self.completeFunctionTest("MONTHS_BETWEEN(A, A)", 't_datetime', "MONTHS_BETWEEN(A, A)")
        self.completeFunctionTest("A", 't_datetime', "CURRENT_TIMESTAMP()", 'WHERE A < NOW()')
        self.completeFunctionTest("NUMTODSINTERVAL(A, 'HOUR')", 't', "NUMTODSINTERVAL(A, 'HOUR')")
        self.completeFunctionTest("NUMTOYMINTERVAL(A, 'MONTH')", 't', "NUMTOYMINTERVAL(A, 'MONTH')")
        self.completeFunctionTest("POSIX_TIME(A)", 't_datetime', "POSIX_TIME(A)")
        self.completeFunctionTest("ROUND(A)", 't_datetime', "ROUND(A)")
        self.completeFunctionTest("ROUND(A, 'CC')", 't_datetime', "ROUND(A, 'CC')")
        self.completeFunctionTest("SECOND(A)", 't_datetime', "SECOND(A)")
        self.completeFunctionTest("SECOND(A, 2)", 't_datetime', "SECOND(A, 2)")
        self.completeFunctionTest("SECONDS_BETWEEN(A, A)", 't_datetime', "SECONDS_BETWEEN(A, A)")
        self.completeFunctionTest("V2", 'g', "SESSIONTIMEZONE()", 'WHERE V2 > SESSIONTIMEZONE()')
        self.completeFunctionTest("A", 't_datetime', "SYSDATE", 'WHERE A < SYSDATE')
        self.completeFunctionTest("A", 't_datetime', "SYSTIMESTAMP", 'WHERE A < SYSTIMESTAMP')
        self.completeFunctionTest('TO_CHAR(A)', 't_datetime', 'TO_CHAR(A)')
        self.completeFunctionTest("TO_CHAR(A, 'VW', 'NLS_DATE_LANGUAGE=GERMAN')", 't_datetime', "TO_CHAR(A, 'VW', 'NLS_DATE_LANGUAGE=GERMAN')")
        self.completeFunctionTest("A", 't_datetime', "TO_DATE(TO_CHAR(A)", 'WHERE A < TO_DATE(TO_CHAR(A))')
        self.completeFunctionTest("A", 't_datetime', "TO_DATE(TO_CHAR(A), 'YYYY-MM-DD')", "WHERE A < TO_DATE(TO_CHAR(A), 'YYYY-MM-DD')")
        self.completeFunctionTest("TO_DSINTERVAL(A||' 10:59:59')", 't', "TO_DSINTERVAL(CONCAT(A, ' 10:59:59'))", "WHERE A IS NOT NULL")
        self.completeFunctionTest("A", 't_datetime', "TO_TIMESTAMP(TO_CHAR(A, 'HH24:MI:SS DDD-YYYY'), 'HH24:MI:SS DDD-YYYY')", "WHERE A < TO_TIMESTAMP(TO_CHAR(A, 'HH24:MI:SS DDD-YYYY'), 'HH24:MI:SS DDD-YYYY')")
        self.completeFunctionTest("TO_YMINTERVAL(A || '-' || A)", 't', "TO_YMINTERVAL(CONCAT(A, CONCAT('-', A)))", "WHERE A IS NOT NULL")
        self.completeFunctionTest("TRUNC(A)", 't_datetime', "TRUNC(A)")
        self.completeFunctionTest("TRUNC(A, 'W')", 't_datetime', "TRUNC(A, 'W')")
        self.completeFunctionTest("TRUNCATE(A)", 't_datetime', "TRUNC(A)")
        self.completeFunctionTest("TRUNCATE(A, 'W')", 't_datetime', "TRUNC(A, 'W')")
        self.completeFunctionTest("WEEK(A)", 't_datetime', "WEEK(A)")
        self.completeFunctionTest("YEAR(A)", 't_datetime', "YEAR(A)")
        self.completeFunctionTest("YEARS_BETWEEN(A, A)", 't_datetime', "YEARS_BETWEEN(A, A)")

    def testGeospatialFunctions(self):
        self.completeFunctionTest("ST_AREA(A)", 't_geometry', "ST_AREA(A)")
        self.completeFunctionTest("ST_BOUNDARY(A)", 't_geometry', "ST_BOUNDARY(A)", "WHERE id < 3")
        self.completeFunctionTest("ST_BUFFER(A, 1)", 't_geometry', "ST_BUFFER(A, 1)")
        self.completeFunctionTest("ST_CENTROID(A)", 't_geometry', "ST_CENTROID(A)")
        self.completeFunctionTest("ST_CONTAINS(A, A)", 't_geometry', "ST_CONTAINS(A, A)")
        self.completeFunctionTest("ST_CONVEXHULL(A)", 't_geometry', "ST_CONVEXHULL(A)")
        self.completeFunctionTest("ST_CROSSES(A, A)", 't_geometry', "ST_CROSSES(A, A)")
        self.completeFunctionTest("ST_DIFFERENCE(A, A)", 't_geometry', "ST_DIFFERENCE(A, A)")
        self.completeFunctionTest("ST_DIMENSION(A)", 't_geometry', "ST_DIMENSION(A)")
        self.completeFunctionTest("ST_DISJOINT(A, A)", 't_geometry', "ST_DISJOINT(A, A)")
        self.completeFunctionTest("ST_DISTANCE(A, A)", 't_geometry', "ST_DISTANCE(A, A)")
        self.completeFunctionTest("ST_ENDPOINT(A)", 't_geometry', "ST_ENDPOINT(A)", "WHERE id = 2")
        self.completeFunctionTest("ST_ENVELOPE(A)", 't_geometry', "ST_ENVELOPE(A)")
        self.completeFunctionTest("ST_EQUALS(A, A)", 't_geometry', "ST_EQUALS(A, A)")
        self.completeFunctionTest("ST_EXTERIORRING(A)", 't_geometry', "ST_EXTERIORRING(A)", "WHERE id = 1")
        self.completeFunctionTest("ST_FORCE2D('POINT(1 2 '||a||')')", 't', "ST_FORCE2D(CONCAT('POINT(1 2 ', CONCAT(A, ')')))")
        self.completeFunctionTest("ST_GEOMETRYN(A, 1)", 't_geometry', "ST_GEOMETRYN(A, 1)", "WHERE id = 3")
        self.completeFunctionTest("ST_GEOMETRYTYPE(A)", 't_geometry', "ST_GEOMETRYTYPE(A)")
        self.completeFunctionTest("ST_INTERIORRINGN(A, 1)", 't_geometry', "ST_INTERIORRINGN(A, 1)", "WHERE id = 1")
        self.completeFunctionTest("ST_INTERSECTION(A)", 't_geometry', "ST_INTERSECTION(A)")
        self.completeFunctionTest("ST_INTERSECTION(A, A)", 't_geometry', "ST_INTERSECTION(A, A)")
        self.completeFunctionTest("ST_INTERSECTS(A, A)", 't_geometry', "ST_INTERSECTS(A, A)")
        self.completeFunctionTest("ST_ISCLOSED(A)", 't_geometry', "ST_ISCLOSED(A)", "WHERE id = 2")
        self.completeFunctionTest("ST_ISEMPTY(A)", 't_geometry', "ST_ISEMPTY(A)")
        self.completeFunctionTest("ST_ISRING(A)", 't_geometry', "ST_ISRING(A)", "WHERE id = 2")
        self.completeFunctionTest("ST_ISSIMPLE(A)", 't_geometry', "ST_ISSIMPLE(A)", "WHERE id < 3")
        self.completeFunctionTest("ST_LENGTH(A)", 't_geometry', "ST_LENGTH(A)")
        self.completeFunctionTest("ST_NUMGEOMETRIES(A)", 't_geometry', "ST_NUMGEOMETRIES(A)", "WHERE id = 3")
        self.completeFunctionTest("ST_NUMINTERIORRINGS(A)", 't_geometry', "ST_NUMINTERIORRINGS(A)", "WHERE id = 1")
        self.completeFunctionTest("ST_NUMPOINTS(A)", 't_geometry', "ST_NUMPOINTS(A)", "WHERE id = 2")
        self.completeFunctionTest("ST_OVERLAPS(A, A)", 't_geometry', "ST_OVERLAPS(A, A)")
        self.completeFunctionTest("ST_POINTN(A, 1)", 't_geometry', "ST_POINTN(A, 1)", "WHERE id = 2")
        self.completeFunctionTest("ST_SETSRID(A, 5)", 't_geometry', "ST_SETSRID(A, 5)")
        self.completeFunctionTest("ST_STARTPOINT(A)", 't_geometry', "ST_STARTPOINT(A)", "WHERE id = 2")
        self.completeFunctionTest("ST_SYMDIFFERENCE(A, A)", 't_geometry', "ST_SYMDIFFERENCE(A, A)")
        self.completeFunctionTest("ST_TRANSFORM(ST_SETSRID(A, 2000), 2001)", 't_geometry', "ST_TRANSFORM(ST_SETSRID(A, 2000), 2001)")
        self.completeFunctionTest("ST_TOUCHES(A, A)", 't_geometry', "ST_TOUCHES(A, A)")
        self.completeFunctionTest("ST_UNION(A)", 't_geometry', "ST_UNION(A)")
        self.completeFunctionTest("ST_UNION(A, A)", 't_geometry', "ST_UNION(A, A)")
        self.completeFunctionTest("ST_WITHIN(A, A)", 't_geometry', "ST_WITHIN(A, A)")
        self.completeFunctionTest("ST_X(ST_STARTPOINT(A))", 't_geometry', "ST_X(ST_STARTPOINT(A))", "WHERE id = 2")
        self.completeFunctionTest("ST_Y(ST_STARTPOINT(A))", 't_geometry', "ST_Y(ST_STARTPOINT(A))", "WHERE id = 2")

    def testBitwiseFunctions(self):
        self.completeFunctionTest('BIT_AND(A, A)', 't', 'BIT_AND(A, A)')
        self.completeFunctionTest('BIT_CHECK(A, 0)', 't', 'BIT_CHECK(A, 0)')
        self.completeFunctionTest('BIT_NOT(A)', 't', 'BIT_NOT(A)')
        self.completeFunctionTest('BIT_OR(A, A)', 't', 'BIT_OR(A, A)')
        self.completeFunctionTest('BIT_SET(A, 0)', 't', 'BIT_SET(A, 0)')
        self.completeFunctionTest('A + BIT_TO_NUM(A,0,0)', 't', 'BIT_TO_NUM(A, 0, 0)', "WHERE A=1")
        self.completeFunctionTest('BIT_XOR(A, A)', 't', 'BIT_XOR(A, A)')

    def testConversionFunctions(self):
        self.completeFunctionTest('CAST(A as CHAR(15))', 't', 'CAST(A AS CHAR(15) UTF8)')
        self.completeFunctionTest('CAST(CAST(A > 0 as VARCHAR(15)) as BOOLEAN)', 't', 'CAST(CAST(0 < A AS VARCHAR(15) UTF8) AS BOOLEAN)')
        self.completeFunctionTest('CAST(CAST(A as VARCHAR(30)) as DATE)', 't_datetime', 'CAST(CAST(A AS VARCHAR(30) UTF8) AS DATE)')
        self.completeFunctionTest('CAST(CAST(A as VARCHAR(15)) as DECIMAL(8,1))', 't', 'CAST(CAST(A AS VARCHAR(15) UTF8) AS DECIMAL(8, 1))')
        self.completeFunctionTest('CAST(CAST(C as VARCHAR(15)) as DOUBLE)', 't', 'CAST(CAST(C AS VARCHAR(15) UTF8) AS DOUBLE)')
        self.completeFunctionTest('CAST(CAST(A as VARCHAR(100)) as GEOMETRY(5))', 't_geometry', 'CAST(CAST(A AS VARCHAR(100) UTF8) AS GEOMETRY(5))')
        self.completeFunctionTest('CAST(CAST(B as VARCHAR(100)) as INTERVAL DAY(5) TO SECOND(2))', 't_interval', 'CAST(CAST(B AS VARCHAR(100) UTF8) AS INTERVAL DAY (5) TO SECOND (2))')
        self.completeFunctionTest('CAST(CAST(A as VARCHAR(100)) as INTERVAL YEAR (5) TO MONTH)', 't_interval', 'CAST(CAST(A AS VARCHAR(100) UTF8) AS INTERVAL YEAR (5) TO MONTH)')
        self.completeFunctionTest('CAST(CAST(A as VARCHAR(100)) as TIMESTAMP)', 't_datetime', 'CAST(CAST(A AS VARCHAR(100) UTF8) AS TIMESTAMP)')
        self.completeFunctionTest('CAST(CAST(A as VARCHAR(100)) as TIMESTAMP WITH LOCAL TIME ZONE)', 't_datetime', 'CAST(CAST(A AS VARCHAR(100) UTF8) AS TIMESTAMP WITH LOCAL TIME ZONE)')
        self.completeFunctionTest('CAST(A as VARCHAR(15))', 't', 'CAST(A AS VARCHAR(15) UTF8)')
        self.completeFunctionTest('CONVERT(CHAR(15), A)', 't', 'CAST(A AS CHAR(15) UTF8)')
        self.completeFunctionTest('IS_NUMBER(TO_CHAR(A))', 't', 'IS_NUMBER(TO_CHAR(A))')
        self.completeFunctionTest("IS_NUMBER(TO_CHAR(A), '99999.999')", 't', "IS_NUMBER(TO_CHAR(A), '99999.999')")
        self.completeFunctionTest('IS_DATE(TO_CHAR(A))', 't_datetime', 'IS_DATE(TO_CHAR(A))')
        self.completeFunctionTest("IS_DATE(TO_CHAR(A), 'YYYY-MM-DD')", 't_datetime', "IS_DATE(TO_CHAR(A), 'YYYY-MM-DD')")
        self.completeFunctionTest('IS_TIMESTAMP(TO_CHAR(A))', 't_datetime', 'IS_TIMESTAMP(TO_CHAR(A))')
        self.completeFunctionTest("IS_TIMESTAMP(TO_CHAR(A), 'HH24:MI:SS DDD-YYYY')", 't_datetime', "IS_TIMESTAMP(TO_CHAR(A), 'HH24:MI:SS DDD-YYYY')")
        self.completeFunctionTest('IS_BOOLEAN(B)', 't', 'IS_BOOLEAN(B)')
        self.completeFunctionTest('IS_DSINTERVAL(B)', 't', 'IS_DSINTERVAL(B)')
        self.completeFunctionTest('IS_YMINTERVAL(B)', 't', 'IS_YMINTERVAL(B)')

    def testOtherFunctions(self):
        self.completeFunctionTest("CASE A WHEN 1 THEN 'YES' WHEN 2 THEN 'FOO' ELSE 'NO' END", 't', "CASE A WHEN 1 THEN 'YES' WHEN 2 THEN 'FOO' ELSE 'NO' END")
        self.completeFunctionTest("CASE WHEN A > 1 THEN 'YES' ELSE 'NO' END", 't', "CASE WHEN 1 < A THEN 'YES' ELSE 'NO' END")
        self.completeFunctionTest('COALESCE(A, B, C)', 't', 'WHEN A IS NOT NULL THEN A WHEN B IS NOT NULL THEN B ELSE C')
        self.completeFunctionTest('a', 't', 'CURRENT_SCHEMA', "WHERE A > CURRENT_SCHEMA")
        self.completeFunctionTest('a', 't', 'CURRENT_SESSION', "WHERE A < CURRENT_SESSION")
        self.completeFunctionTest('a', 't', 'CURRENT_STATEMENT', "WHERE A > CURRENT_STATEMENT")
        self.completeFunctionTest('v2', 'g', 'CURRENT_USER', "WHERE v2 > CURRENT_USER")
        self.completeFunctionTest("DECODE(v2, 'xyz', 1, 'abc', 2, 3)", 'g', "CASE V2 WHEN 'xyz' THEN 1 WHEN 'abc' THEN 2 ELSE 3 END")
        self.completeFunctionTest('GREATEST(A, C)', 't', 'GREATEST(A, C)')
        self.completeFunctionTest('HASH_MD5(A)', 't', 'HASH_MD5(A)')
        self.completeFunctionTest('HASH_SHA(A)', 't', 'HASH_SHA1(A)')
        self.completeFunctionTest('HASH_SHA1(A)', 't', 'HASH_SHA1(A)')
        self.completeFunctionTest('HASH_TIGER(A)', 't', 'HASH_TIGER(A)')
        self.completeFunctionTest('LEAST(A, C)', 't', 'LEAST(A, C)')
        self.completeFunctionTest('NULLIF(A, 1)', 't', 'CASE A WHEN 1 THEN NULL ELSE A')
        self.completeFunctionTest('NULLIFZERO(A)', 't', 'NULLIFZERO(A)')
        self.completeFunctionTest('NVL(A, C)', 't', 'CASE WHEN A IS NOT NULL THEN A ELSE C')
        self.completeFunctionTest('NVL2(A, C, NULL)', 't', 'CASE WHEN A IS NOT NULL THEN C ELSE NULL')
        self.completeFunctionTest('A', 't', 'SELECT A FROM NATIVE.T', 'WHERE ROWNUM < 3')
        self.completeFunctionTest('*', 't', 'SELECT * FROM NATIVE.T', 'WHERE ROWNUM < 3')
        self.completeFunctionTest('A, C, ROWNUM', 't', 'SELECT A, C FROM NATIVE.T', 'WHERE ROWNUM < 3')
        self.completeFunctionTest('A, ROW_NUMBER() OVER (ORDER BY C DESC) ROW_NUMBER', 't', 'SELECT A, C FROM NATIVE.T')
        with self.assertRaisesRegexp(Exception, 'ROWID is invalid for virtual tables.'):
            self.completeFunctionTest('ROWID', 't', 'SELECT true')
        self.completeFunctionTest('v2', 'g', 'SYS_GUID()', "WHERE v2 > SYS_GUID()")
        self.completeFunctionTest('v2', 'g', 'USER', "WHERE v2 > USER")
        self.completeFunctionTest('ZEROIFNULL(A)', 't', 'ZEROIFNULL(A)')

    def testAggregateFunctions(self):
        self.completeFunctionTest('APPROXIMATE_COUNT_DISTINCT(A)', 't', 'APPROXIMATE_COUNT_DISTINCT(A)')
        self.completeFunctionTest('AVG(A)', 't', 'AVG(A)')
        self.completeFunctionTest('AVG(ALL A)', 't', 'AVG(A)')
        self.completeFunctionTest('AVG(DISTINCT A)', 't', 'AVG(DISTINCT A)')
        self.completeFunctionTest('CORR(A, C)', 't', 'SELECT A, C FROM')
        self.completeFunctionTest('COUNT(A)', 't', 'COUNT(A)')
        self.completeFunctionTest('COUNT(*)', 't', 'COUNT(*)')
        self.completeFunctionTest('COUNT(DISTINCT A)', 't', 'COUNT(DISTINCT A)')
        self.completeFunctionTest('COUNT(ALL (A, C))', 't', 'SELECT A, C FROM')
        self.completeFunctionTest('COVAR_POP(A, C)', 't', 'SELECT A, C FROM')
        self.completeFunctionTest('COVAR_SAMP(A, C)', 't', 'SELECT A, C FROM')
        self.completeFunctionTest('FIRST_VALUE(A)', 't', 'FIRST_VALUE(A)')
        self.completeFunctionTest('GROUP_CONCAT(A)', 't', 'GROUP_CONCAT(A)')
        self.completeFunctionTest('GROUP_CONCAT(DISTINCT A)', 't', 'GROUP_CONCAT(DISTINCT A)')
        self.completeFunctionTest('GROUP_CONCAT(A ORDER BY C)', 't', 'GROUP_CONCAT(A ORDER BY C)')
        self.completeFunctionTest('GROUP_CONCAT(A ORDER BY C DESC)', 't', 'GROUP_CONCAT(A ORDER BY C DESC)')
        self.completeFunctionTest('GROUP_CONCAT(A ORDER BY C DESC NULLS LAST)', 't', 'GROUP_CONCAT(A ORDER BY C DESC NULLS LAST)')
        self.completeFunctionTest("GROUP_CONCAT(A SEPARATOR ';'||' ')", 't', "GROUP_CONCAT(A SEPARATOR '; ')")
        self.completeFunctionTest('GROUPING(A)', 't', 'SELECT A FROM', 'GROUP BY A')
        self.completeFunctionTest('GROUPING(A, C)', 't', 'SELECT A, C FROM', 'GROUP BY A, C')
        self.completeFunctionTest('GROUPING_ID(A)', 't', 'SELECT A FROM', 'GROUP BY A')
        self.completeFunctionTest('GROUPING_ID(A, C)', 't', 'SELECT A, C FROM', 'GROUP BY A, C')
        self.completeFunctionTest('LAST_VALUE(A)', 't', 'LAST_VALUE(A)')
        self.completeFunctionTest('MAX(A)', 't', 'MAX(A)')
        self.completeFunctionTest('MAX(ALL A)', 't', 'MAX(A)')
        self.completeFunctionTest('MAX(DISTINCT A)', 't', 'MAX(A)')
        self.completeFunctionTest('MEDIAN(A)', 't', 'MEDIAN(A)')
        self.completeFunctionTest('MIN(A)', 't', 'MIN(A)')
        self.completeFunctionTest('MIN(ALL A)', 't', 'MIN(A)')
        self.completeFunctionTest('MIN(DISTINCT A)', 't', 'MIN(A)')
        self.completeFunctionTest('PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY A)', 't', 'SELECT A FROM')
        self.completeFunctionTest('PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY A)', 't', 'SELECT A FROM')
        self.completeFunctionTest('REGR_AVGX(A, C)', 't', 'SELECT A, C FROM')
        self.completeFunctionTest('REGR_AVGY(A, C)', 't', 'SELECT A, C FROM')
        self.completeFunctionTest('REGR_COUNT(A, C)', 't', 'SELECT A, C FROM')
        self.completeFunctionTest('REGR_INTERCEPT(A, C)', 't', 'SELECT A, C FROM')
        self.completeFunctionTest('REGR_R2(A, C)', 't', 'SELECT A, C FROM')
        self.completeFunctionTest('REGR_SLOPE(A, C)', 't', 'SELECT A, C FROM')
        self.completeFunctionTest('REGR_SXX(A, C)', 't', 'SELECT A, C FROM')
        self.completeFunctionTest('REGR_SXY(A, C)', 't', 'SELECT A, C FROM')
        self.completeFunctionTest('REGR_SYY(A, C)', 't', 'SELECT A, C FROM')
        self.completeFunctionTest('STDDEV(A)', 't', 'STDDEV(A)')
        self.completeFunctionTest('STDDEV(ALL A)', 't', 'STDDEV(A)')
        self.completeFunctionTest('STDDEV(DISTINCT A)', 't', 'STDDEV(DISTINCT A)')
        self.completeFunctionTest('STDDEV_POP(A)', 't', 'STDDEV_POP(A)')
        self.completeFunctionTest('STDDEV_POP(ALL A)', 't', 'STDDEV_POP(A)')
        self.completeFunctionTest('STDDEV_POP(DISTINCT A)', 't', 'STDDEV_POP(DISTINCT A)')
        self.completeFunctionTest('STDDEV_SAMP(A)', 't', 'STDDEV_SAMP(A)')
        self.completeFunctionTest('STDDEV_SAMP(ALL A)', 't', 'STDDEV_SAMP(A)')
        self.completeFunctionTest('STDDEV_SAMP(DISTINCT A)', 't', 'STDDEV_SAMP(DISTINCT A)')
        self.completeFunctionTest('SUM(A)', 't', 'SUM(A)')
        self.completeFunctionTest('SUM(ALL A)', 't', 'SUM(A)')
        self.completeFunctionTest('SUM(DISTINCT A)', 't', 'SUM(DISTINCT A)')
        self.completeFunctionTest('VAR_POP(A)', 't', 'VAR_POP(A)')
        self.completeFunctionTest('VAR_POP(ALL A)', 't', 'VAR_POP(A)')
        self.completeFunctionTest('VAR_POP(DISTINCT A)', 't', 'VAR_POP(DISTINCT A)')
        self.completeFunctionTest('VAR_SAMP(A)', 't', 'VAR_SAMP(A)')
        self.completeFunctionTest('VAR_SAMP(ALL A)', 't', 'VAR_SAMP(A)')
        self.completeFunctionTest('VAR_SAMP(DISTINCT A)', 't', 'VAR_SAMP(DISTINCT A)')
        self.completeFunctionTest('VARIANCE(A)', 't', 'VARIANCE(A)')
        self.completeFunctionTest('VARIANCE(ALL A)', 't', 'VARIANCE(A)')
        self.completeFunctionTest('VARIANCE(DISTINCT A)', 't', 'VARIANCE(DISTINCT A)')

class IllegalResponses(VSchemaTest):

    def setUp(self):
        self.query('DROP SCHEMA IF EXISTS NATIVE CASCADE')
        self.query('CREATE SCHEMA NATIVE')
        self.query('CREATE VIEW my_view as select 1 c1, 2 c2 from dual')

    def testViewInPushdownResponse(self):
        self.createViewAdapter(schemaName="ADAPTER", adapterName="VIEW_ADAPTER")
        self.query('DROP FORCE VIRTUAL SCHEMA IF EXISTS VS1 CASCADE')
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.VIEW_ADAPTER')
        self.query('''SELECT * FROM VS1.DUMMY''')

    def testVirtualTableInPushdownResponse(self):
        self.createRecursiveAdapter(schemaName="ADAPTER", adapterName="RECURSIVE_ADAPTER")
        self.query('DROP FORCE VIRTUAL SCHEMA IF EXISTS VS1 CASCADE')
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.RECURSIVE_ADAPTER')
        with self.assertRaisesRegexp(Exception, 'The pushdown query returned by the Adapter contains a virtual table \\(DUMMY\\). This is currently not supported.'):
            self.query('''
                SELECT * FROM VS1.DUMMY
                ''')

    def testViewInPushdownResponseWithSubSelect(self):
        self.createExtendedViewAdapter(schemaName="ADAPTER", adapterName="EXTENDED_VIEW_ADAPTER")
        self.query('DROP FORCE VIRTUAL SCHEMA IF EXISTS VS1 CASCADE')
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.EXTENDED_VIEW_ADAPTER')
        self.query('''SELECT * FROM VS1.DUMMY''')

    def testVirtualTableInPushdownResponseWithSubSelect(self):
        self.createExtendedRecursiveAdapter(schemaName="ADAPTER", adapterName="EXTENDED_RECURSIVE_ADAPTER")
        self.query('DROP FORCE VIRTUAL SCHEMA IF EXISTS VS1 CASCADE')
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.EXTENDED_RECURSIVE_ADAPTER')
        with self.assertRaisesRegexp(Exception, 'The pushdown query returned by the Adapter contains a virtual table \\(DUMMY\\). This is currently not supported.'):
            self.query('''
                SELECT * FROM VS1.DUMMY
                ''')

    def createViewAdapter(self, schemaName="ADAPTER", adapterName="VIEW_ADAPTER"):
        self.dropOldAdapter(schemaName, adapterName)
        self.query('CREATE SCHEMA {schema}'.format(schema=schemaName))
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON ADAPTER SCRIPT {schema}.{adapter} AS
            import json
            import string
            def adapter_call(request):
                # database expects utf-8 encoded string of type str. unicode not yet supported
                root = json.loads(request)
                if root["type"] == "createVirtualSchema":
                    res = {{
                        "type": "createVirtualSchema",
                        "schemaMetadata": {{
                            "tables": [
                            {{
                                "name": "DUMMY",
                                "columns": [{{
                                    "name": "KEY",
                                    "dataType": {{"type": "VARCHAR", "size": 2000000}}
                                }},{{
                                    "name": "VALUE",
                                    "dataType": {{"type": "VARCHAR", "size": 2000000}}
                                }}]
                            }}]
                        }}
                    }}
                    return json.dumps(res).encode('utf-8')
                if root["type"] == "getCapabilities":
                    return json.dumps({{
                        "type": "getCapabilities",
                        "capabilities": []
                        }}).encode('utf-8') # database expects utf-8 encoded string of type str. unicode not yet supported.
                elif root["type"] == "pushdown":
                    res = {{
                        "type": "pushdown",
                        "sql": "SELECT * FROM NATIVE.my_view"
                    }}
                    return json.dumps(res).encode('utf-8')
                else:
                    raise ValueError('Unsupported callback')
            /
            ''').format(schema = schemaName, adapter = adapterName))

    def createRecursiveAdapter(self, schemaName="ADAPTER", adapterName="RECURSIVE_ADAPTER"):
        self.dropOldAdapter(schemaName, adapterName)
        self.query('CREATE SCHEMA {schema}'.format(schema=schemaName))
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON ADAPTER SCRIPT {schema}.{adapter} AS
            import json
            import string
            def adapter_call(request):
                # database expects utf-8 encoded string of type str. unicode not yet supported
                root = json.loads(request)
                if root["type"] == "createVirtualSchema":
                    res = {{
                        "type": "createVirtualSchema",
                        "schemaMetadata": {{
                            "tables": [
                            {{
                                "name": "DUMMY",
                                "columns": [{{
                                    "name": "KEY",
                                    "dataType": {{"type": "VARCHAR", "size": 2000000}}
                                }},{{
                                    "name": "VALUE",
                                    "dataType": {{"type": "VARCHAR", "size": 2000000}}
                                }}]
                            }}]
                        }}
                    }}
                    return json.dumps(res).encode('utf-8')
                if root["type"] == "getCapabilities":
                    return json.dumps({{
                        "type": "getCapabilities",
                        "capabilities": []
                        }}).encode('utf-8') # database expects utf-8 encoded string of type str. unicode not yet supported.
                elif root["type"] == "pushdown":
                    res = {{
                        "type": "pushdown",
                        "sql": "SELECT * FROM DUMMY"
                    }}
                    return json.dumps(res).encode('utf-8')
                else:
                    raise ValueError('Unsupported callback')
            /
            ''').format(schema = schemaName, adapter = adapterName))

    def createExtendedViewAdapter(self, schemaName="ADAPTER", adapterName="EXTENDED_VIEW_ADAPTER"):
        self.dropOldAdapter(schemaName, adapterName)
        self.query('CREATE SCHEMA {schema}'.format(schema=schemaName))
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON ADAPTER SCRIPT {schema}.{adapter} AS
            import json
            import string
            def adapter_call(request):
                # database expects utf-8 encoded string of type str. unicode not yet supported
                root = json.loads(request)
                if root["type"] == "createVirtualSchema":
                    res = {{
                        "type": "createVirtualSchema",
                        "schemaMetadata": {{
                            "tables": [
                            {{
                                "name": "DUMMY",
                                "columns": [{{
                                    "name": "KEY",
                                    "dataType": {{"type": "VARCHAR", "size": 2000000}}
                                }},{{
                                    "name": "VALUE",
                                    "dataType": {{"type": "VARCHAR", "size": 2000000}}
                                }}]
                            }}]
                        }}
                    }}
                    return json.dumps(res).encode('utf-8')
                if root["type"] == "getCapabilities":
                    return json.dumps({{
                        "type": "getCapabilities",
                        "capabilities": []
                        }}).encode('utf-8') # database expects utf-8 encoded string of type str. unicode not yet supported.
                elif root["type"] == "pushdown":
                    res = {{
                        "type": "pushdown",
                        "sql": "SELECT count(*), count(*) FROM (SELECT * FROM NATIVE.my_view)"
                    }}
                    return json.dumps(res).encode('utf-8')
                else:
                    raise ValueError('Unsupported callback')
            /
            ''').format(schema = schemaName, adapter = adapterName))

    def createExtendedRecursiveAdapter(self, schemaName="ADAPTER", adapterName="EXTENDED_RECURSIVE_ADAPTER"):
        self.dropOldAdapter(schemaName, adapterName)
        self.query('CREATE SCHEMA {schema}'.format(schema=schemaName))
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON ADAPTER SCRIPT {schema}.{adapter} AS
            import json
            import string
            def adapter_call(request):
                # database expects utf-8 encoded string of type str. unicode not yet supported
                root = json.loads(request)
                if root["type"] == "createVirtualSchema":
                    res = {{
                        "type": "createVirtualSchema",
                        "schemaMetadata": {{
                            "tables": [
                            {{
                                "name": "DUMMY",
                                "columns": [{{
                                    "name": "KEY",
                                    "dataType": {{"type": "VARCHAR", "size": 2000000}}
                                }},{{
                                    "name": "VALUE",
                                    "dataType": {{"type": "VARCHAR", "size": 2000000}}
                                }}]
                            }}]
                        }}
                    }}
                    return json.dumps(res).encode('utf-8')
                if root["type"] == "getCapabilities":
                    return json.dumps({{
                        "type": "getCapabilities",
                        "capabilities": []
                        }}).encode('utf-8') # database expects utf-8 encoded string of type str. unicode not yet supported.
                elif root["type"] == "pushdown":
                    res = {{
                        "type": "pushdown",
                        "sql": "SELECT count(*), count(*) FROM (SELECT * FROM DUMMY)"
                    }}
                    return json.dumps(res).encode('utf-8')
                else:
                    raise ValueError('Unsupported callback')
            /
            ''').format(schema = schemaName, adapter = adapterName))

class MiscQueriesTest(VSchemaTest):

    setupDone = False

    def setUp(self):
        if self.__class__.setupDone:
            self.query(''' CLOSE SCHEMA ''')
            return
        self.createJdbcAdapter()
        self.createNative()
        self.commit()  # We have to commit, otherwise the adapter won't see these tables
        self.createVirtualSchemaJdbc("VS1", "NATIVE", "ADAPTER.JDBC_ADAPTER", False)
        self.query(''' CLOSE SCHEMA ''')
        self.commit()
        self.__class__.setupDone = True

    def testWithInsertInto(self):
        self.query('DROP SCHEMA target CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA target')
        self.query('CREATE TABLE target.t like native.t')
        self.query('INSERT INTO target.t SELECT * FROM vs1.t')
        self.assertRowsEqualIgnoreOrder(
            self.query('SELECT * FROM native.t'),
            self.query('SELECT * FROM target.t'))
        self.query('TRUNCATE TABLE target.t')
        self.query('INSERT INTO target.t SELECT a, b, 1 FROM (SELECT * FROM vs1.t)')
        self.assertRowsEqualIgnoreOrder(
            self.query('SELECT a, b, 1 FROM native.t'),
            self.query('SELECT * FROM target.t'))

    def testCreateTableLike(self):
        self.query('DROP SCHEMA target CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA target')
        self.query('CREATE TABLE target.t LIKE vs1.t')
        self.assertRowsEqualIgnoreOrder(
            self.query('DESCRIBE native.t'),
            self.query('DESCRIBE target.t'))
        self.query('CREATE OR REPLACE TABLE target.t LIKE VS1.t INCLUDING DEFAULTS INCLUDING IDENTITY INCLUDING COMMENTS')
        self.assertRowsEqualIgnoreOrder(
            self.query('DESCRIBE native.t'),
            self.query('DESCRIBE target.t'))

    def testCreateTableAs(self):
        self.query('DROP SCHEMA target CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA target')
        self.query('CREATE TABLE target.t LIKE vs1.t')
        self.assertRowsEqualIgnoreOrder(
            self.query('DESCRIBE native.t'),
            self.query('DESCRIBE target.t'))
        self.query('CREATE OR REPLACE TABLE target.t_from_virtual AS SELECT * FROM vs1.t')
        self.query('CREATE OR REPLACE TABLE target.t_from_native  AS SELECT * FROM native.t')
        # TODO This comparison currently fails, because the IMPORT FROM JDBC subselect returns the wrong data types
        self.assertRowsEqualIgnoreOrder(
            self.query('DESCRIBE target.t_from_virtual'),
            self.query('DESCRIBE target.t_from_native'))
        self.assertRowsEqualIgnoreOrder(
            self.query('SELECT * FROM target.t_from_virtual'),
            self.query('SELECT * FROM target.t_from_native'))
        # TODO Remove this when the IMPORT FROM JDBC returns correct types
        self.assertRowsEqualIgnoreOrder(
            self.query('SELECT * FROM target.t_from_virtual'),
            self.query('SELECT cast(a AS INT), b, c FROM target.t_from_native'))

#@unittest.skip("skipped test")
class PreparedTest(VSchemaTest):

    def setUp(self):
        self.createJdbcAdapter()
        self.createNative()
        self.commit()  # We have to commit, otherwise the adapter won't see these tables
        self.createVirtualSchemaJdbc("VS1", "NATIVE", "ADAPTER.JDBC_ADAPTER", True)
        self.query(''' CLOSE SCHEMA ''')

    def testPrepared(self):
        rows = self.query('''
            SELECT a, b FROM vs1.t WHERE a = ?
        ''', 1)
        rows_native = self.query('''
            SELECT a, b FROM native.t WHERE a = ?
        ''', 1)
        self.assertEqual(rows, rows_native)

if __name__ == '__main__':
    udf.main()

