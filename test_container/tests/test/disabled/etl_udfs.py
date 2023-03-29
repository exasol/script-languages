#!/usr/opt/bs-python-2.7/bin/python

import os
import sys
import re

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf
from udf import useData, expectedFailure

from decimal import *

def get_profile_for(testcase,the_query,profile_table = "EXA_USER_PROFILE_LAST_DAY"):
    col_rs = testcase.query("desc "+profile_table)
    cols = []
    for c in col_rs:
        cols.append(c[0])
    testcase.query("ALTER SESSION SET profile = 'ON'")
    testcase.query(the_query)
    testcase.query("alter session set profile = 'OFF'")
    testcase.query("commit")
    testcase.query("FLUSH STATISTICS")
    testcase.query("commit")
    profile = testcase.query("select * from "+profile_table+" where CURRENT_STATEMENT-5 = stmt_id and SESSION_ID = CURRENT_SESSION")
    result = []
    for p in profile:
        r = {}
        for i in range(len(cols)):
            r[cols[i]] = p[i]
        result.append(r)
    return result

class LocalGroupByWhenIProc(udf.TestCase):

    def moreThanOneNode(self):
        x = self.query('select nproc()')
        return x[0][0] > 1

    def setUp(self):
        self.query('CREATE SCHEMA LOCAL_GROUPBY_WHEN_IPROC', ignore_errors=True)
        self.query('OPEN SCHEMA LOCAL_GROUPBY_WHEN_IPROC', ignore_errors=True)
        self.query('create or replace table t(x int, y int)')
        self.query('insert into t values (1,2), (1,2), (2,3), (2,3)')
        self.query("alter session set QUERY_CACHE='OFF'")
        self.query("create or replace table T_idx (id int, idx int, c dec(1), distribute by idx)")
        self.query("insert into T_idx values (1,1,1), (2,2,2), (3,3,3), (4,4,4), (5,5,5)")
        self.query("insert into T_idx select f.id, f.idx, f.c from T_idx f, T_idx") 
        self.query("enforce local index on T_idx (idx)")
        self.query(udf.fixindent('''
                 CREATE OR REPLACE python SET SCRIPT exporter_sr(x int, y int order by x desc)
                 returns int AS
                 def run(ctx):
                     c = 0
                     while True:
                         c = c +1
                         if not ctx.next(): break
                     return c
                 /
                '''))
        self.query(udf.fixindent('''
                 CREATE OR REPLACE python SET SCRIPT exporter_se(x int, y int order by x desc)
                 EMITS (x int, y int) AS
                 def run(ctx):
                     while True:
                         ctx.emit(ctx.x, ctx.y)
                         if not ctx.next(): break
                 /
                '''))


    def run_query(self,q,isLocal,profile_len,part_num,isSorted=False):
       p = get_profile_for(self,q)
       self.assertEqual(len(p), profile_len)
       if isSorted:
           self.assertIsNone(re.search('GLOBAL',p[part_num]["PART_INFO"])) and self.assertIsNone(re.search('LOCAL',p[part_num]["PART_INFO"]))
       else:
           if isLocal:
               self.assertIsNone(re.search('GLOBAL',p[part_num]["PART_INFO"]))
           else:
               self.assertIsNotNone(re.search('GLOBAL',p[part_num]["PART_INFO"]))
 
    def test_local_opt_when_iproc(self):
        if self.moreThanOneNode():
            self.run_query("select sum(x) from t group by y", False, 3, 2)
            self.run_query("select sum(x) from t group by y,iproc()", True, 3, 2)


    def test_set_emits_exporter(self):
        if self.moreThanOneNode():
            global_query = "select exporter_se(x,y) from t group by floor(random(0,5))"
            local_query = "select exporter_se(x,y) from t group by floor(random(0,5)),iproc()"
            self.run_query(global_query, False, 3, 2)
            self.run_query(local_query, True, 3, 2)
            local_table = "exporter_se_xxx_l"
            global_table = "exporter_se_xxx_g"
            self.query("create or replace table "+global_table+" as ("+global_query+")")
            self.query("create or replace table "+local_table+" as ("+local_query+")")
            q = self.query("select * from "+global_table+" MINUS select * from "+local_table)
            self.assertEqual(len(q),0)
            q = self.query("select * from "+local_table+" MINUS select * from "+global_table)
            self.assertEqual(len(q),0)
            q = self.query("select * from t MINUS select * from "+local_table)
            self.assertEqual(len(q),0)
            q = self.query("select * from "+local_table+" MINUS select * from t")
            self.assertEqual(len(q),0)

    def test_set_returns_exporter(self):
        if self.moreThanOneNode():
            self.run_query("select exporter_sr(x,y) from t group by y", False, 3, 2)
            self.run_query("select exporter_sr(x,y) from t group by y,iproc()", True, 3, 2)
            print get_profile_for(self,"select exporter_sr(x,y) from t group by y,iproc()")
        

    def test_only_toplevel_iproc(self):
        if self.moreThanOneNode():
            self.run_query("select exporter_sr(x,y) from t group by y,iproc()/2", False, 3, 2)
            self.run_query("select sum(x) from t group by y,2*iproc()", False, 3, 2)
            self.run_query("select sum(x) from t group by y,iproc()=iproc()", False, 3, 2)

    def test_sorted_group_by(self):
        if self.moreThanOneNode():
            # Not sorted:
            self.run_query("select id i, count(*) c from T_idx group by id", False,3,2, False)
            self.run_query("select id i, count(*) c from T_idx group by id,iproc()", True,3,2, False)
            # potentially sorted:
            self.run_query("select idx k, count(*) c from T_idx group by idx", False,3,2, True)
            self.run_query("select idx k, count(*) c from T_idx group by idx,iproc()", False, 3,2, True)


class TargetPredefinitionForInsertsWithUDFs(udf.TestCase):

    def assertRowsEqual(self, left, right):
        udf.TestCase.assertRowsEqual(self,sorted([tuple(x) for x in left]), sorted([tuple(x) for x in right]))

    def setUp(self):
        self.query('CREATE SCHEMA TARGET_PREDEFINITION_FOR_INSERTS_WITH_UDFs', ignore_errors=True)
        self.query('OPEN SCHEMA TARGET_PREDEFINITION_FOR_INSERTS_WITH_UDFs', ignore_errors=False)
        self.query('create or replace table t1(s int)')
        self.query('create or replace table t2(s int, t int)')
        self.query('create or replace table t3(s int, t int, u int)')
        self.query('create or replace table t4(s int, t int, u int, v int)')
        self.query('create or replace table t5(s int, t int, u int, v int, w int)')
        self.query('create or replace table t6(s int, t int, u int, v int, w int, x int)')
        self.query('create or replace table t7(s int, t int, u int, v int, w int, x int, y int)')
        self.query('create or replace table t8(s int, t int, u int, v int, w int, x int, y int, z int)')
        self.query('create or replace table dt1(s int default 666)')
        self.query('create or replace table dt2(s int default 666, t int default 666)')
        self.query('create or replace table dt3(s int default 666, t int default 666, u int default 666)')
        self.query('create or replace table dt4(s int default 666, t int default 666, u int default 666, v int default 666)')
        self.query('create or replace table dt5(s int default 666, t int default 666, u int default 666, v int default 666, w int default 666)')
        self.query('create or replace table it1(s int identity 665)')
        self.query('create or replace table it2(s int default 666, t int identity 665)')
        self.query('create or replace table it3(s int default 666, t int default 666, u int identity 665)')
        self.query('create or replace table it4(s int default 666, t int default 666, u int default 666, v int identity 665)')
        self.query('create or replace table it5(s int default 666, t int default 666, u int default 666, v int default 666, w int identity 665)')
        self.query('create or replace table vct(s varchar(2000000), t varchar(2000000))')
        self.query('create or replace table ct(s char(2000), t char(2000))')
        self.query('create or replace table s1(a int)')
        self.query('insert into s1 values (1)')
        self.query('create or replace table s2(a int, b int)')
        self.query('insert into s2 values (2,2)')
        self.query('create or replace table s3(a int, b int, c int)')
        self.query('insert into s3 values (3,3,3)')
        self.query('create or replace table s4(a int, b int, c int, d int)')
        self.query('insert into s4 values (4,4,4,4)')
        self.query('create or replace table s5(a int, b int, c int, d int, e int)')
        self.query('insert into s5 values (5,5,5,5,5)')
        self.query("alter session set QUERY_CACHE='OFF'")
        self.query(udf.fixindent('''
                 CREATE OR REPLACE LUA SCALAR SCRIPT ScR(x double)
                 RETURNS int AS
                 function run(ctx)
                     return decimal(ctx.x)
                 end
                 /
                '''))
        self.query(udf.fixindent('''
                 CREATE OR REPLACE LUA SET SCRIPT SeR(x double)
                 RETURNS int AS
                 function run(ctx)
                     return decimal(ctx.x)
                 end
                 /
                '''))
        self.query(udf.fixindent('''
                 CREATE OR REPLACE LUA SCALAR SCRIPT ScE1(x double)
                 EMITS (m int) AS
                 function run(ctx)
                     for i=1,ctx.x do
                         ctx.emit(decimal(ctx.x))
                     end
                 end
                 /
                '''))
        self.query(udf.fixindent('''
                 CREATE OR REPLACE LUA SCALAR SCRIPT ScE2(x double)
                 EMITS (m int, n int) AS
                 function run(ctx)
                     for i=1,ctx.x do
                         ctx.emit(decimal(ctx.x),decimal(ctx.x))
                     end
                 end
                 /
                '''))
        self.query(udf.fixindent('''
                 CREATE OR REPLACE LUA SCALAR SCRIPT ScE2p(x double)
                 EMITS (m int, n int) AS
                 function run(ctx)
                     for i=1,ctx.x do
                         ctx.emit(decimal(ctx.x),decimal(ctx.x+1))
                     end
                 end
                 /
                '''))
        self.query(udf.fixindent('''
                 CREATE OR REPLACE LUA SCALAR SCRIPT ScE3(x double)
                 EMITS (m int, n int, o int) AS
                 function run(ctx)
                     for i=1,ctx.x do
                         ctx.emit(decimal(ctx.x), decimal(ctx.x), decimal(ctx.x))
                     end
                 end
                 /
                '''))
        self.query(udf.fixindent('''
                 CREATE OR REPLACE LUA SCALAR SCRIPT ScE3p(x double)
                 EMITS (m int, n int, o int) AS
                 function run(ctx)
                     for i=1,ctx.x do
                         ctx.emit(decimal(ctx.x), decimal(ctx.x+1), decimal(ctx.x+2))
                     end
                 end
                 /
                '''))
        self.query(udf.fixindent('''
                 CREATE OR REPLACE LUA SET SCRIPT SeE1(x double)
                 EMITS (m int) AS
                 function run(ctx)
                     repeat
                         for i=1,ctx.x do
                             ctx.emit(decimal(ctx.x))
                         end
                     until not ctx.next()
                 end
                 /
                '''))
        self.query(udf.fixindent('''
                 CREATE OR REPLACE LUA SET SCRIPT SeE2(x double)
                 EMITS (m int, n int) AS
                 function run(ctx)
                     repeat
                         for i=1,ctx.x do
                             ctx.emit(decimal(ctx.x), decimal(ctx.x))
                         end
                     until not ctx.next()
                 end
                 /
                '''))
        self.query(udf.fixindent('''
                 CREATE OR REPLACE LUA SET SCRIPT SeE2p(x double)
                 EMITS (m int, n int) AS
                 function run(ctx)
                     repeat
                         for i=1,ctx.x do
                             ctx.emit(decimal(ctx.x), decimal(ctx.x+1))
                         end
                     until not ctx.next()
                 end
                 /
                '''))
        self.query(udf.fixindent('''
                 CREATE OR REPLACE LUA SET SCRIPT SeE3(x double)
                 EMITS (m int, n int, o int) AS
                 function run(ctx)
                     repeat
                         for i=1,ctx.x do
                             ctx.emit(decimal(ctx.x), decimal(ctx.x), decimal(ctx.x))
                         end
                     until not ctx.next()
                 end
                 /
                '''))
        self.query(udf.fixindent('''
                 CREATE OR REPLACE LUA SET SCRIPT SeE3p(x double)
                 EMITS (m int, n int, o int) AS
                 function run(ctx)
                     repeat
                         for i=1,ctx.x do
                             ctx.emit(decimal(ctx.x), decimal(ctx.x+1), decimal(ctx.x+2))
                         end
                     until not ctx.next()
                 end
                 /
                '''))
        self.query(udf.fixindent('''
                 CREATE OR REPLACE LUA SCALAR SCRIPT VCScE1_1000(num double)
                 EMITS (m varchar(1000)) AS
                 function run(ctx)
                     local s = string.rep('x',1000)
                     for i=1,ctx.num do
                         ctx.emit(s)
                     end
                 end
                 /
                '''))
        self.query(udf.fixindent('''
                 CREATE OR REPLACE LUA SET SCRIPT VCSeE1_1000(num double)
                 EMITS (m varchar(1000)) AS
                 function run(ctx)
                     local s = string.rep('x',1000)
                     for i=1,ctx.num do
                         ctx.emit(s)
                     end
                 end
                 /
                '''))
        self.query(udf.fixindent('''
                 CREATE OR REPLACE LUA SET SCRIPT VCSeE2_1000(num double)
                 EMITS (m varchar(1000), n varchar(1000)) AS
                 function run(ctx)
                     local s = string.rep('x',1000)
                     for i=1,ctx.num do
                         ctx.emit(s,s)
                     end
                 end
                 /
                '''))

        self.query(udf.fixindent('''
                 CREATE OR REPLACE LUA SCALAR SCRIPT mte(num double)
                 EMITS (i int, d double, c char(100), v varchar(100), b bool) AS
                 function run(ctx)
                     local s = string.rep('x',1000)
                     for i=1,ctx.num do
                         ctx.emit(decimal(1),1,'1','1',1)
                     end
                 end
                 /
                '''))
        self.query(udf.fixindent('''
                 CREATE OR REPLACE LUA SET SCRIPT mtee(num double)
                 EMITS (i int, d double, c char(100), v varchar(100), b bool) AS
                 function run(ctx)
                     local s = string.rep('x',1000)
                     for i=1,ctx.num do
                         ctx.emit(decimal(1),1,'1','1',1)
                     end
                 end
                 /
                '''))

    def run_query(self,q,isDirect,profile_len=False,part_num=False):
       #print '*************'
       p = get_profile_for(self,q)
       if profile_len:
           self.assertEqual(len(p), profile_len)
       if part_num: 
           if isDirect:
               self.assertIsNone(re.search('TEMPORARY',p[part_num]["PART_INFO"]))
           else:
               self.assertIsNotNone(re.search('TEMPORARY',p[part_num]["PART_INFO"]))
       else:
           foundTemp = False
           for y in range(len(p)):
               #print "Checking PI: ",p[y]["PART_INFO"]
               if p[y]["PART_INFO"]:
                   if re.search('TEMPORARY',p[y]["PART_INFO"]):
                       foundTemp = True
                       break
           if isDirect:
               self.assertFalse(foundTemp)
           else:
               self.assertTrue(foundTemp)


    ####

    def test_simple_scalar_returns(self):
        self.run_query('insert into t1 select * from s1',True)
        self.run_query('insert into t1 select ScR(1)',True)
        self.run_query('insert into t1 select ScR(a) from s1',True)
        rows = self.query('select * from t1')
        expected = [(1,), (1,), (1,)]
        self.assertRowsEqual(expected, rows)
        self.run_query('insert into t2 select ScR(a),ScR(a) from s1',True)
        self.assertRowsEqual([(1,1,)], self.query('select * from t2'))


    def test_simple_scalar_emits(self):
        self.run_query('insert into t1 select ScE1(1)', True)
        self.assertRowsEqual([(1,)], self.query('select * from t1'))
        self.run_query('insert into t1 select ScE1(a) from s1', True)
        self.assertRowsEqual([(1,),(1,)], self.query('select * from t1'))
        #       
        self.run_query('insert into t2 select ScE2(1)', True)
        self.assertRowsEqual([(1,1,)], self.query('select * from t2'))
        #       
        self.run_query('insert into t3 select ScE3(1)', True)
        self.assertRowsEqual([(1,1,1,)], self.query('select * from t3'))
        #

    def test_simple_set_returns(self):
        self.run_query('insert into t1 select * from s1',True)
        self.run_query('insert into t1 select SeR(1)',True)
        self.run_query('insert into t1 select SeR(a) from s1',True)
        rows = self.query('select * from t1')
        expected = [(1,), (1,), (1,)]
        self.assertRowsEqual(expected, rows)
        self.run_query('insert into t2 select SeR(a),SeR(a) from s1',True)
        self.assertRowsEqual([(1,1,)], self.query('select * from t2'))

    def test_simple_set_emits(self):
        self.run_query('insert into t1 select SeE1(1)', True)
        self.assertRowsEqual([(1,)], self.query('select * from t1'))
        self.run_query('insert into t1 select ScE1(a) from s1', True)
        self.assertRowsEqual([(1,),(1,)], self.query('select * from t1'))
        #       
        self.run_query('insert into t2 select SeE2(1)', True)
        self.assertRowsEqual([(1,1,)], self.query('select * from t2'))
        #       
        self.run_query('insert into t3 select SeE3(1)', True)
        self.assertRowsEqual([(1,1,1,)], self.query('select * from t3'))

    ####

    def test_scalar_returns_with_missing_and_no_default(self):
        with self.assertRaisesRegex(Exception, 'not enough value'):
            self.query('insert into t2 select ScR(1)')
        with self.assertRaisesRegex(Exception, 'too many values'):
            self.query('insert into t1 select ScR(1),ScR(1)')

        self.run_query('insert into t2(s) select ScR(a) from s1',True)
        self.assertRowsEqual([(1,None)], self.query('select * from t2'))
        self.run_query('insert into t2(t) select ScR(a) from s1',True)
        self.assertRowsEqual([(1,None),(None,1)], self.query('select * from t2'))

    def test_scalar_emits_with_missing_and_no_default(self):
        #
        with self.assertRaisesRegex(Exception, 'not enough value'):
            self.query('insert into t2 select ScE1(1)')
        with self.assertRaisesRegex(Exception, 'too many values'):
            self.query('insert into t1 select ScE2(1)')
        #
        self.run_query('insert into t1 select ScE1(1)', True)
        self.assertRowsEqual([(1,)], self.query('select * from t1'))
        #       
        self.run_query('insert into t2(s) select ScE1(1)', True)
        self.assertRowsEqual([(1,None,)], self.query('select * from t2'))
        self.run_query('insert into t2(t) select ScE1(1)', True)
        self.assertRowsEqual([(1,None,),(None,1,)], self.query('select * from t2'))
        #       
        self.run_query('insert into t3(s,t,u) select ScE3(1)', True)
        self.assertRowsEqual([(1,1,1,)], self.query('select * from t3'))
        self.run_query('insert into t3(s,t) select 1,ScE1(1)', True)
        self.assertRowsEqual([(1,1,1,),(1,1,None,)], self.query('select * from t3'))
        self.run_query('insert into t3(s,t) select ScE1(1),1', True)
        self.assertRowsEqual([(1,1,1,),(1,1,None,),(1,1,None,)], self.query('select * from t3'))
        self.run_query('insert into t3(t,u) select ScE1(1),1', True)
        self.assertRowsEqual([(1,1,1,),(1,1,None,),(1,1,None,),(None,1,1)], self.query('select * from t3'))

    def test_set_returns_with_missing_and_no_default(self):
        with self.assertRaisesRegex(Exception, 'not enough value'):
            self.query('insert into t2 select SeR(1)')
        with self.assertRaisesRegex(Exception, 'too many values'):
            self.query('insert into t1 select SeR(1),SeR(1)')

        self.run_query('insert into t2(s) select SeR(a) from s1',True)
        self.assertRowsEqual([(1,None)], self.query('select * from t2'))
        self.run_query('insert into t2(t) select SeR(a) from s1',True)
        self.assertRowsEqual([(1,None),(None,1)], self.query('select * from t2'))

    def test_set_emits_with_missing_and_no_default(self):
        #
        with self.assertRaisesRegex(Exception, 'not enough value'):
            self.query('insert into t2 select SeE1(1)')
        with self.assertRaisesRegex(Exception, 'too many values'):
            self.query('insert into t1 select SeE2(1)')
        #
        self.run_query('insert into t1 select SeE1(1)', True)
        self.assertRowsEqual([(1,)], self.query('select * from t1'))
        #       
        self.run_query('insert into t2(s) select SeE1(1)', False)
        self.assertRowsEqual([(1,None,)], self.query('select * from t2'))
        self.run_query('insert into t2(t) select SeE1(1)', False)
        self.assertRowsEqual([(1,None,),(None,1,)], self.query('select * from t2'))
        #       
        self.run_query('insert into t3(s,t,u) select SeE3(1)', True)
        self.assertRowsEqual([(1,1,1,)], self.query('select * from t3'))
        self.run_query('insert into t3(s,t) select SeE2(1)', False)
        self.assertRowsEqual([(1,1,1,),(1,1,None,)], self.query('select * from t3'))
        self.run_query('insert into t3(s,t) select SeE2(1)', False)
        self.assertRowsEqual([(1,1,1,),(1,1,None,),(1,1,None,)], self.query('select * from t3'))
        self.run_query('insert into t3(t,u) select SeE2(1)', False)
        self.assertRowsEqual([(1,1,1,),(1,1,None,),(1,1,None,),(None,1,1)], self.query('select * from t3'))

        ####

    def test_scalar_returns_with_missing_and_default(self):
        with self.assertRaisesRegex(Exception, 'not enough value'):
            self.query('insert into dt2 select ScR(1)')
        with self.assertRaisesRegex(Exception, 'too many values'):
            self.query('insert into dt1 select ScR(1),ScR(1)')

        self.run_query('insert into dt2(s) select ScR(a) from s1',True)
        self.assertRowsEqual([(1,666)], self.query('select * from dt2'))
        self.run_query('insert into dt2(t) select ScR(a) from s1',True)
        self.assertRowsEqual([(1,666),(666,1)], self.query('select * from dt2'))

    def test_scalar_emits_with_missing_and_default(self):
        #
        with self.assertRaisesRegex(Exception, 'not enough value'):
            self.query('insert into dt2 select ScE1(1)')
        with self.assertRaisesRegex(Exception, 'too many values'):
            self.query('insert into dt1 select ScE2(1)')
        #
        self.run_query('insert into dt1 select ScE1(1)', True)
        self.assertRowsEqual([(1,)], self.query('select * from dt1'))
        #       
        self.run_query('insert into dt2(s) select ScE1(1)', True)
        self.assertRowsEqual([(1,666,)], self.query('select * from dt2'))
        self.run_query('insert into dt2(t) select ScE1(1)', True)
        self.assertRowsEqual([(1,666,),(666,1,)], self.query('select * from dt2'))
        #       
        self.run_query('insert into dt3(s,t,u) select ScE3(1)', True)
        self.assertRowsEqual([(1,1,1,)], self.query('select * from dt3'))
        self.run_query('insert into dt3(s,t) select 1,ScE1(1)', True)
        self.assertRowsEqual([(1,1,1,),(1,1,666,)], self.query('select * from dt3'))
        self.run_query('insert into dt3(s,t) select ScE1(1),1', True)
        self.assertRowsEqual([(1,1,1,),(1,1,666,),(1,1,666,)], self.query('select * from dt3'))
        self.run_query('insert into dt3(t,u) select ScE1(1),1', True)
        self.assertRowsEqual([(1,1,1,),(1,1,666,),(1,1,666,),(666,1,1)], self.query('select * from dt3'))

    def test_set_returns_with_missing_and_default(self):
        with self.assertRaisesRegex(Exception, 'not enough value'):
            self.query('insert into dt2 select SeR(1)')
        with self.assertRaisesRegex(Exception, 'too many values'):
            self.query('insert into dt1 select SeR(1),SeR(1)')

        self.run_query('insert into dt2(s) select SeR(a) from s1',True)
        self.assertRowsEqual([(1,666)], self.query('select * from dt2'))
        self.run_query('insert into dt2(t) select SeR(a) from s1',True)
        self.assertRowsEqual([(1,666),(666,1)], self.query('select * from dt2'))

    def test_set_emits_with_missing_and_default(self):
        #
        with self.assertRaisesRegex(Exception, 'not enough value'):
            self.query('insert into dt2 select SeE1(1)')
        with self.assertRaisesRegex(Exception, 'too many values'):
            self.query('insert into dt1 select SeE2(1)')
        #
        self.run_query('insert into dt1 select SeE1(1)', True)
        self.assertRowsEqual([(1,)], self.query('select * from dt1'))
        #       
        self.run_query('insert into dt2(s) select SeE1(1)', False)
        self.assertRowsEqual([(1,666,)], self.query('select * from dt2'))
        self.run_query('insert into dt2(t) select SeE1(1)', False)
        self.assertRowsEqual([(1,666,),(666,1,)], self.query('select * from dt2'))
        #       
        self.run_query('insert into dt3(s,t,u) select SeE3(1)', True)
        self.assertRowsEqual([(1,1,1,)], self.query('select * from dt3'))
        ## Attention, for set-emits, we cannot add default/identity columns, therefore we cannot predefine target here.
        self.run_query('insert into dt3(s,t) select SeE2(1)', False)
        self.assertRowsEqual([(1,1,1,),(1,1,666,)], self.query('select * from dt3'))
        self.run_query('insert into dt3(s,t) select SeE2(1)', False)
        self.assertRowsEqual([(1,1,1,),(1,1,666,),(1,1,666,)], self.query('select * from dt3'))
        self.run_query('insert into dt3(t,u) select SeE2(1)', False)
        self.assertRowsEqual([(1,1,1,),(1,1,666,),(1,1,666,),(666,1,1)], self.query('select * from dt3'))

    def assertRowsAmong(self, left, right):
         rrows = [tuple(x) for x in right]
         rrows.sort()
         foundOne = False
         for l in left:
             lrows = [tuple(x) for x in l]
             lrows.sort()
             if lrows == rrows: foundOne = True
         self.assertTrue(foundOne, "Found no '%s' among '%s'" % (right, left))


    def test_scalar_returns_with_missing_and_identity(self):
        with self.assertRaisesRegex(Exception, 'not enough value'):
            self.query('insert into it2 select ScR(1)')
        with self.assertRaisesRegex(Exception, 'too many values'):
            self.query('insert into it1 select ScR(1),ScR(1)')

        self.run_query('insert into it2(s) select ScR(a) from s1',True)
        cand1 = [[(1,x)] for x in [666,667,668,669]]
        self.assertRowsAmong(cand1, self.query('select * from it2'))
        self.run_query('insert into it2(t) select ScR(a) from s1',True)
        cand2 = [[(1,x),(666,1)] for x in [666,667,668,669]]
        self.assertRowsAmong(cand2, self.query('select * from it2'))

        self.run_query('insert into it3(s) select ScR(a) from s1',True)
        cand3 = [[(1,666,x)] for x in [666,667,668,669]]
        self.assertRowsAmong(cand3, self.query('select * from it3'))
        self.run_query('insert into it3(t) select ScR(a) from s1',True)
        cand4 = [[(1,666,x),(666,1,y)] for x in range(666,674) for y in range(666,674)]
        self.assertRowsAmong(cand4, self.query('select * from it3'))

    def test_scalar_emits_with_missing_and_identity(self):
        #
        with self.assertRaisesRegex(Exception, 'not enough value'):
            self.query('insert into it2 select ScE1(1)')
        with self.assertRaisesRegex(Exception, 'too many values'):
            self.query('insert into it1 select ScE2(1)')
        #
        self.run_query('insert into it1 select ScE1(1)', True)
        self.assertRowsEqual([(1,)], self.query('select * from it1'))
        #       
        self.run_query('insert into it2(s) select ScE1(1)', True)
        self.assertRowsEqual([(1,666,)], self.query('select * from it2'))
        self.run_query('insert into it2(t) select ScE1(1)', True)
        self.assertRowsEqual([(1,666,),(666,1,)], self.query('select * from it2'))
        #       
        self.run_query('insert into it3(s,t,u) select ScE3(1)', True)
        self.assertRowsEqual([(1,1,1,)], self.query('select * from it3'))
        self.run_query('insert into it3(s,t) select 1,ScE1(1)', True)
        self.assertRowsEqual([(1,1,1,),(1,1,666,)], self.query('select * from it3'))
        self.run_query('insert into it3(s,t) select ScE1(1),1', True)
        self.assertRowsEqual([(1,1,1,),(1,1,666,),(1,1,667,)], self.query('select * from it3'))
        self.run_query('insert into it3(t,u) select ScE1(1),1', True)
        self.assertRowsEqual([(1,1,1,),(1,1,666,),(1,1,667,),(666,1,1)], self.query('select * from it3'))

    def test_set_returns_with_missing_and_identity(self):
        with self.assertRaisesRegex(Exception, 'not enough value'):
            self.query('insert into it2 select SeR(1)')
        with self.assertRaisesRegex(Exception, 'too many values'):
            self.query('insert into it1 select SeR(1),SeR(1)')

        self.run_query('insert into it2(s) select SeR(a) from s1',True)
        self.assertRowsEqual([(1,666)], self.query('select * from it2'))
        self.run_query('insert into it2(t) select SeR(a) from s1',True)
        self.assertRowsEqual([(1,666),(666,1)], self.query('select * from it2'))

        self.run_query('insert into it3(s) select SeR(a) from s1',True)
        self.assertRowsEqual([(1,666,666)], self.query('select * from it3'))
        self.run_query('insert into it3(t) select SeR(a) from s1',True)
        self.assertRowsEqual([(1,666,666),(666,1,667)], self.query('select * from it3'))

    def test_set_emits_with_missing_and_identity(self):
        #
        with self.assertRaisesRegex(Exception, 'not enough value'):
            self.query('insert into it2 select SeE1(1)')
        with self.assertRaisesRegex(Exception, 'too many values'):
            self.query('insert into it1 select SeE2(1)')
        #
        self.run_query('insert into it1 select SeE1(1)', True)
        self.assertRowsEqual([(1,)], self.query('select * from it1'))
        #       
        self.run_query('insert into it2(s) select SeE1(1)', False)
        self.assertRowsEqual([(1,666,)], self.query('select * from it2'))
        self.run_query('insert into it2(t) select SeE1(1)', False)
        self.assertRowsEqual([(1,666,),(666,1,)], self.query('select * from it2'))
        #       
        self.run_query('insert into it3(s,t,u) select SeE3(1)', True)
        self.assertRowsEqual([(1,1,1,)], self.query('select * from it3'))
        self.run_query('insert into it3(s,t) select SeE2(1)', False)                        
        self.assertRowsEqual([(1,1,1,),(1,1,666,)], self.query('select * from it3'))
        self.run_query('insert into it3(s,t) select SeE2(1)', False)
        self.assertRowsEqual([(1,1,1,),(1,1,666,),(1,1,667,)], self.query('select * from it3'))
        self.run_query('insert into it3(t,u) select SeE2(1)', False)
        self.assertRowsEqual([(1,1,1,),(1,1,666,),(1,1,667,),(666,1,1)], self.query('select * from it3'))

    ##############################################################################################################
        # The following tests are basically the same as above but we also change the order of the target columns
        # ... and don't use 1 as insert value all the time as we want to check for the correct insert ordering.
    ##############################################################################################################


    def test_permuted_scalar_returns(self):
        self.run_query('insert into t2(t,s) select ScR(a),ScR(a)+1 from s1',True)
        self.assertRowsEqual([(2,1,)], self.query('select * from t2'))
     

    def test_permuted_scalar_emits(self): 
        self.run_query('insert into t2(t,s) select ScE2p(1)', False)
        self.assertRowsEqual([(2,1,)], self.query('select * from t2'))
        #       
        self.run_query('insert into t3(s,u,t) select ScE3p(1)', False)
        self.assertRowsEqual([(1,3,2,)], self.query('select * from t3'))
        #
        self.run_query('insert into t3(u,s,t) select ScE2(1),2', False)
        self.assertRowsEqual([(1,3,2,),(1,2,1,)], self.query('select * from t3'))
        #
        self.run_query('insert into t3(s,u,t) select ScE2(1),2', False)
        self.assertRowsEqual([(1,3,2,),(1,2,1,),(1,2,1,)], self.query('select * from t3'))
        #
        self.run_query('insert into t3(s,t,u) select ScE2(1),2', True)
        self.assertRowsEqual([(1,3,2,),(1,2,1,),(1,2,1,),(1,1,2,)], self.query('select * from t3'))
        #
        self.run_query('insert into t3(u,s,t) select 2,ScE2(1)', True)
        self.assertRowsEqual([(1,3,2,),(1,2,1,),(1,2,1,),(1,1,2,),(1,1,2,)], self.query('select * from t3'))
        #
        self.run_query('insert into t3(s,u,t) select 2,ScE2(1)', False)
        self.assertRowsEqual([(1,3,2,),(1,2,1,),(1,2,1,),(1,1,2,),(1,1,2,),(2,1,1,)], self.query('select * from t3'))
        #
        self.run_query('insert into t3(s,t,u) select 2,ScE2(1)', True)
        self.assertRowsEqual([(1,3,2,),(1,2,1,),(1,2,1,),(1,1,2,),(1,1,2,),(2,1,1,),(2,1,1)], self.query('select * from t3'))

    def test_permuted_set_returns(self): 
        self.run_query('insert into t2(t,s) select SeR(a),SeR(a)+1 from s1',True)
        self.assertRowsEqual([(2,1,)], self.query('select * from t2'))

    def test_permuted_set_emits(self): 
        self.run_query('insert into t2(t,s) select SeE2p(1)', False)
        self.assertRowsEqual([(2,1,)], self.query('select * from t2'))
        #       
        self.run_query('insert into t3(s,u,t) select SeE3p(1)', False)
        self.assertRowsEqual([(1,3,2,)], self.query('select * from t3'))

    ####
    ####
    ####

    def test_permuted_scalar_returns_with_missing_and_no_default(self): 
        with self.assertRaisesRegex(Exception, 'not enough value'):
            self.query('insert into t2(t,s) select ScR(1)')

    def test_permuted_scalar_emits_with_missing_and_no_default(self):
        #
        with self.assertRaisesRegex(Exception, 'not enough value'):
            self.query('insert into t2(t,s) select ScE1(1)')
        #
        self.run_query('insert into t3(t,s,u) select ScE3p(1)', False)
        self.assertRowsEqual([(2,1,3,)], self.query('select * from t3'))
        self.run_query('insert into t3(t,s) select 2,ScE1(1)', True)
        self.assertRowsEqual([(2,1,3,),(1,2,None,)], self.query('select * from t3'))
        self.run_query('insert into t3(t,s) select ScE1(1),2', True)
        self.assertRowsEqual([(2,1,3,),(1,2,None,),(2,1,None,)], self.query('select * from t3'))
        self.run_query('insert into t3(u,t) select ScE1(1),2', True)
        self.assertRowsEqual([(2,1,3,),(1,2,None,),(2,1,None,),(None,2,1)], self.query('select * from t3'))

    def test_permuted_set_returns_with_missing_and_no_default(self): 
        pass

    def test_permuted_set_emits_with_missing_and_no_default(self):
        self.run_query('insert into t3(s,t,u) select SeE3p(1)', True)
        self.assertRowsEqual([(1,2,3)], self.query('select * from t3'))
        self.run_query('insert into t3(t,s,u) select SeE3p(1)', False)
        self.assertRowsEqual([(1,2,3),(2,1,3,)], self.query('select * from t3'))
        self.run_query('insert into t3(t,s) select SeE2p(1)', False)
        self.assertRowsEqual([(1,2,3),(2,1,3,),(2,1,None,)], self.query('select * from t3'))
        self.run_query('insert into t3(u,t) select SeE2p(1)', False)
        self.assertRowsEqual([(1,2,3),(2,1,3,),(2,1,None,),(None,2,1,)], self.query('select * from t3'))
        self.run_query('insert into t3(t,u) select SeE2p(1)', False)
        self.assertRowsEqual([(1,2,3),(2,1,3,),(2,1,None,),(None,2,1,),(None,1,2)], self.query('select * from t3'))


    ####
    ####
    ####
    ####

    def test_permuted_scalar_returns_with_missing_and_default(self): pass

    def test_permuted_scalar_emits_with_missing_and_default(self):
        #
        with self.assertRaisesRegex(Exception, 'not enough value'):
            self.query('insert into dt2(t,s) select ScE1(1)')
        #
        self.run_query('insert into dt3(t,s,u) select ScE3p(1)', False)
        self.assertRowsEqual([(2,1,3,)], self.query('select * from dt3'))
        self.run_query('insert into dt3(t,s) select 2,ScE1(1)', True)
        self.assertRowsEqual([(2,1,3,),(1,2,666,)], self.query('select * from dt3'))
        self.run_query('insert into dt3(t,s) select ScE1(1),2', True)
        self.assertRowsEqual([(2,1,3,),(1,2,666,),(2,1,666,)], self.query('select * from dt3'))
        self.run_query('insert into dt3(u,t) select ScE1(1),2', True)
        self.assertRowsEqual([(2,1,3,),(1,2,666,),(2,1,666,),(666,2,1)], self.query('select * from dt3'))

    def test_permuted_set_returns_with_missing_and_default(self): pass

    def test_permuted_set_emits_with_missing_and_default(self): 
        self.run_query('insert into dt3(s,t,u) select SeE3p(1)', True)
        self.assertRowsEqual([(1,2,3)], self.query('select * from dt3'))
        self.run_query('insert into dt3(t,s,u) select SeE3p(1)', False)
        self.assertRowsEqual([(1,2,3),(2,1,3,)], self.query('select * from dt3'))
        self.run_query('insert into dt3(t,s) select SeE2p(1)', False)
        self.assertRowsEqual([(1,2,3),(2,1,3,),(2,1,666,)], self.query('select * from dt3'))
        self.run_query('insert into dt3(u,t) select SeE2p(1)', False)
        self.assertRowsEqual([(1,2,3),(2,1,3,),(2,1,666,),(666,2,1,)], self.query('select * from dt3'))
        self.run_query('insert into dt3(t,u) select SeE2p(1)', False)
        self.assertRowsEqual([(1,2,3),(2,1,3,),(2,1,666,),(666,2,1,),(666,1,2)], self.query('select * from dt3'))

    ####
    ####
    ####
    ####

    def test_permuted_scalar_returns_with_missing_and_identity(self): pass

    def test_permuted_scalar_emits_with_missing_and_identity(self):
        #
        with self.assertRaisesRegex(Exception, 'not enough value'):
            self.query('insert into it2(t,s) select ScE1(1)')
        #
        self.run_query('insert into it3(t,s,u) select ScE3p(1)', False)
        self.assertRowsEqual([(2,1,3,)], self.query('select * from it3'))
        self.run_query('insert into it3(t,s) select 2,ScE1(1)', True)
        self.assertRowsEqual([(2,1,3,),(1,2,666,)], self.query('select * from it3'))
        self.run_query('insert into it3(t,s) select ScE1(1),2', True)
        self.assertRowsEqual([(2,1,3,),(1,2,666,),(2,1,667,)], self.query('select * from it3'))
        self.run_query('insert into it3(u,t) select ScE1(1),2', True)
        self.assertRowsEqual([(2,1,3,),(1,2,666,),(2,1,667,),(666,2,1)], self.query('select * from it3'))

    def test_permuted_set_returns_with_missing_and_identity(self): pass

    def test_permuted_set_emits_with_missing_and_identity(self):
        self.run_query('insert into it3(s,t,u) select SeE3p(1)', True)
        self.assertRowsEqual([(1,2,3)], self.query('select * from it3'))
        self.run_query('insert into it3(t,s,u) select SeE3p(1)', False)
        self.assertRowsEqual([(1,2,3),(2,1,3,)], self.query('select * from it3'))
        self.run_query('insert into it3(t,s) select SeE2p(1)', False)
        self.assertRowsEqual([(1,2,3),(2,1,3,),(2,1,666,)], self.query('select * from it3'))
        self.run_query('insert into it3(u,t) select SeE2p(1)', False)
        self.assertRowsEqual([(1,2,3),(2,1,3,),(2,1,666,),(666,2,1,)], self.query('select * from it3'))
        self.run_query('insert into it3(t,u) select SeE2p(1)', False)
        self.assertRowsEqual([(1,2,3),(2,1,3,),(2,1,666,),(666,2,1,),(666,1,2)], self.query('select * from it3'))


    ####
    ####
    ####
    ####

    def test_scalar_emits_with_large_varchars(self):
        self.run_query('insert into vct(s) select vcsce1_1000(1000)',True)
        self.assertRowsEqual([(1000000,)], self.query('select sum(len(s)) from vct'))
        self.query('create or replace table vct2 like vct')
        self.run_query('insert into vct2 select vcsce1_1000(1),s from vct',True)
        self.assertRowsEqual([(1000000,1000000)], self.query('select sum(len(s)),sum(len(t)) from vct2'))
        ## Target table is referenced by subselect: no predefinition
        self.run_query('insert into vct select vcsce1_1000(1),s from vct',False)                 
        self.assertRowsEqual([(2000000,1000000)], self.query('select sum(len(s)),sum(len(t)) from vct'))

    def test_set_emits_with_large_varchars(self):
        self.run_query('insert into vct(s) select vcsee1_1000(1000)',False)         # More emitted values than needed => no predefine target
        self.assertRowsEqual([(1000000,)], self.query('select sum(len(s)) from vct'))
        self.query('create or replace table vct2 like vct')
        self.run_query('insert into vct2 select vcsee2_1000(1) from vct',True)
        self.assertRowsEqual([(1000,1000)], self.query('select sum(len(s)),sum(len(t)) from vct2'))
        self.run_query('insert into vct select vcsee2_1000(1) from vct',False)
        self.assertRowsEqual([(1001000,1000)], self.query('select sum(len(s)),sum(len(t)) from vct'))

    ####
    ####
    def test_scalar_emits_with_large_chars(self):
        self.run_query('insert into ct(s) select vcsce1_1000(1000)',True)
        self.assertRowsEqual([(1000000,)], self.query('select sum(len(trim(s))) from ct'))
    ####
    ####
    ####
 
    def test_scalar_emits_with_various_types(self):
        self.run_query('create or replace table mt as select mte(10)', True)
        self.assertRowsEqual([(1,1,'1','1',True)], self.query('select i,d,trim(c),v,b from mt where i=1 and d=1 and c=1 and v=1 and b=1 limit 1'))

    def test_set_emits_with_various_types(self):
        self.run_query('create or replace table mt as select mtee(10)', True)
        self.assertRowsEqual([(1,1,'1','1',True)], self.query('select i,d,trim(c),v,b from mt where i=1 and d=1 and c=1 and v=1 and b=1 limit 1'))


    def test_too_many_values_exception(self):
        with self.assertRaisesRegex(Exception, 'too many values'):
            self.query('insert into t3(s) select ScE2(1)')
        with self.assertRaisesRegex(Exception, 'too many values'):
            self.query('insert into t3(s,t) select 2,ScE2(1)')
        with self.assertRaisesRegex(Exception, 'too many values'):
            self.query('insert into t3(u,s,t) select 2,ScE2(1),2')


if __name__ == '__main__':
    udf.main()

        
