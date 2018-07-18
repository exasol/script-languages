#!/usr/opt/bs-python-2.7/bin/python

import os
import sys

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf

class LuaNilsAndNulls(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA NilsAndNulls CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA NilsAndNulls')

    def test_sql_null_equals_lua_null_scalar_returns(self):
        self.query('''
                CREATE lua SCALAR SCRIPT
                sql_null_equals_lua_null_scalar_returns(x varchar(200)) RETURNS boolean AS
                function run(ctx)
                    if ctx.x == null then
                        return true
                    else
                        return false
                    end
                end
                ''')
        rows = self.query('''select sql_null_equals_lua_null_scalar_returns(null) from dual''')
        self.assertRowEqual((True,), rows[0])
        rows = self.query('''select sql_null_equals_lua_null_scalar_returns(12) from dual''')
        self.assertRowEqual((False,), rows[0])


    ############################################################################################

    def test_sql_null_equals_lua_null_scalar_emits(self):
        self.query('''
                CREATE or replace lua SCALAR SCRIPT
                sql_null_equals_lua_null_scalar_emits(x varchar(200)) emits (y boolean) AS
                function run(ctx)
                    if ctx.x == null then
                        ctx.emit(true)
                        ctx.emit(true)
                        ctx.emit(true)
                    else
                        ctx.emit(false)
                        ctx.emit(false)
                        ctx.emit(false)
                    end
                end
                ''')
        rows = self.query('''select sql_null_equals_lua_null_scalar_emits(null) from dual''')
        self.assertRowsEqual([(True, ), (True, ), (True, )], rows)
        rows = self.query('''select sql_null_equals_lua_null_scalar_emits(12) from dual''')
        self.assertRowsEqual([(False,), (False,), (False,)], rows)

    ############################################################################################

    def test_sql_null_equals_lua_null_set_returns(self):
        self.query('''
                CREATE or replace lua SET SCRIPT
                sql_null_equals_lua_null_set_returns(x double, y double) RETURNS double AS
                function run(ctx)
                    local res = 0;
                    repeat
                        if ctx.y == null then
                            res = ctx.x;
                        end
                   until not ctx.next()
                   return res;
                end
                ''')
        rows = self.query('''select sql_null_equals_lua_null_set_returns(x,y) from (values (1,1),(2,2),(3,3),(4,cast(NULL as number))) as t(x,y) where y is null''')
        self.assertRowEqual((4,), rows[0])


    def test_sql_null_equals_lua_null_set_returns2_1(self):
        self.query('''
                CREATE or replace lua SET SCRIPT
                sql_null_equals_lua_null_set_returns2_1(i double, x double) RETURNS boolean AS
                function run(ctx)
                    local res = false;
                    repeat
                        if ctx.x == null then
                            res = true;
                          else
                            res = false;
                        end
                   until not ctx.next()
                   return res;
                end
                ''')
        rows = self.query('''select sql_null_equals_lua_null_set_returns2_1(i, x order by i asc) from (values (1,1),(2,2),(3,3),(4,cast(null as number))) as t(i,x)''')
        self.assertRowEqual((True,), rows[0])

    def test_sql_null_equals_lua_null_set_returns2_2(self):
        self.query('''
                CREATE or replace lua SET SCRIPT
                sql_null_equals_lua_null_set_returns2_2(i double, x double) RETURNS boolean AS
                function run(ctx)
                    local res = false;
                    repeat
                        if ctx.x == null then
                            res = true;
                          else
                            res = false;
                        end
                   until not ctx.next()
                   return res;
                end
                ''')
        rows = self.query('''select sql_null_equals_lua_null_set_returns2_2(i, x order by i desc) from (values (1,1),(2,2),(3,3),(4,cast(null as number))) as t(i,x)''')
        self.assertRowEqual((False,), rows[0])

    ############################################################################################

    def test_sql_null_equals_lua_null_set_emits(self):
        self.query('''
                CREATE or replace lua SET SCRIPT
                sql_null_equals_lua_null_set_emits(x double, y double) emits (r double) AS
                function run(ctx)
                    local res = 0;
                    repeat
                        if ctx.y == null then
                            res = ctx.x;
                        end
                   until not ctx.next()
                   ctx.emit(res);
                end
                ''')
        rows = self.query('''select * from (select sql_null_equals_lua_null_set_emits(x,y) from (values (1,1),(2,2),(3,3),(4,cast(NULL as number))) as t(x,y) where y is null)''')
        self.assertRowEqual((4,), rows[0])


    def test_sql_null_equals_lua_null_set_emits2_1(self):
        self.query('''
                CREATE or replace lua SET SCRIPT
                sql_null_equals_lua_null_set_emits2(i double, x double) EMITS (r boolean) AS
                function run(ctx)
                    local res = false;
                    repeat
                        if ctx.x == null then
                            res = true;
                          else
                            res = false;
                        end
                   until not ctx.next()
                   ctx.emit(res);
                end
                ''')
        rows = self.query('''select * from (select sql_null_equals_lua_null_set_emits2(i, x order by i asc) from (values (1,1),(2,2),(3,3),(4,cast(null as number))) as t(i,x))''')
        self.assertRowEqual((True,), rows[0])

    def test_sql_null_equals_lua_null_set_emits2_2(self):
        self.query('''
                CREATE or replace lua SET SCRIPT
                sql_null_equals_lua_null_set_emits2(i double,x double) EMITS (r boolean) AS
                function run(ctx)
                    local res = false;
                    repeat
                        if ctx.x == null then
                            res = true;
                          else
                            res = false;
                        end
                   until not ctx.next()
                   ctx.emit(res);
                end
                ''')
        rows = self.query('''select * from (select sql_null_equals_lua_null_set_emits2(i, x order by i desc) from (values (1,1),(2,2),(3,3),(4,cast(null as number))) as t(i,x))''')
        self.assertRowEqual((False,), rows[0])


        #####################################################
        #####################################################
        #####################################################


    def test_sql_null_equals_lua_nil_scalar_returns(self):
        self.query('''
                CREATE lua SCALAR SCRIPT
                sql_null_equals_lua_nil_scalar_returns(x varchar(200)) RETURNS boolean AS
                function run(ctx)
                    if ctx.x == nil then
                        return true
                    else
                        return false
                    end
                end
                ''')
        rows = self.query('''select sql_null_equals_lua_nil_scalar_returns(null) from dual''')
        self.assertRowEqual((False,), rows[0])
        rows = self.query('''select sql_null_equals_lua_nil_scalar_returns(12) from dual''')
        self.assertRowEqual((False,), rows[0])


    ############################################################################################

    def test_sql_null_equals_lua_nil_scalar_emits(self):
        self.query('''
                CREATE or replace lua SCALAR SCRIPT
                sql_null_equals_lua_nil_scalar_emits(x varchar(200)) emits (y boolean) AS
                function run(ctx)
                    if ctx.x == nil then
                        ctx.emit(true)
                        ctx.emit(true)
                        ctx.emit(true)
                    else
                        ctx.emit(false)
                        ctx.emit(false)
                        ctx.emit(false)
                    end
                end
                ''')
        rows = self.query('''select sql_null_equals_lua_nil_scalar_emits(null) from dual''')
        self.assertRowsEqual([(False, ), (False, ), (False, )], rows)
        rows = self.query('''select sql_null_equals_lua_nil_scalar_emits(12) from dual''')
        self.assertRowsEqual([(False,), (False,), (False,)], rows)

    ############################################################################################

    def test_sql_null_equals_lua_nil_set_returns(self):
        self.query('''
                CREATE or replace lua SET SCRIPT
                sql_null_equals_lua_nil_set_returns(x double, y double) RETURNS double AS
                function run(ctx)
                    local res = 0;
                    repeat
                        if ctx.y == null then
                            res = ctx.x;
                        end
                   until not ctx.next()
                   return res;
                end
                ''')
        rows = self.query('''select sql_null_equals_lua_nil_set_returns(x,y) from (values (1,1),(2,2),(3,3),(4,cast(NULL as number))) as t(x,y) where y is null''')
        self.assertRowEqual((4,), rows[0]) 


    def test_sql_null_equals_lua_nil_set_returns2_1(self):
        self.query('''
                CREATE or replace lua SET SCRIPT
                sql_null_equals_lua_nil_set_returns2_1(i double, x double) RETURNS boolean AS
                function run(ctx)
                    local res = false;
                    repeat
                        if ctx.x == nil then
                            res = true;
                          else
                            res = false;
                        end
                   until not ctx.next()
                   return res;
                end
                ''')
        rows = self.query('''select sql_null_equals_lua_nil_set_returns2_1(i, x order by i asc) from (values (1,1),(2,2),(3,3),(4,cast(null as number))) as t(i,x)''')
        self.assertRowEqual((False,), rows[0])

    def test_sql_null_equals_lua_nil_set_returns2_2(self):
        self.query('''
                CREATE or replace lua SET SCRIPT
                sql_null_equals_lua_nil_set_returns2_2(i double, x double) RETURNS boolean AS
                function run(ctx)
                    local res = false;
                    repeat
                        if ctx.x == nil then
                            res = true;
                          else
                            res = false;
                        end
                   until not ctx.next()
                   return res;
                end
                ''')
        rows = self.query('''select sql_null_equals_lua_nil_set_returns2_2(i, x order by i desc) from (values (1,1),(2,2),(3,3),(4,cast(null as number))) as t(i,x)''')
        self.assertRowEqual((False,), rows[0])

    ############################################################################################

    def test_sql_null_equals_lua_nil_set_emits(self):
        self.query('''
                CREATE or replace lua SET SCRIPT
                sql_null_equals_lua_nil_set_emits(x double, y double) emits (r double) AS
                function run(ctx)
                    local res = 0;
                    repeat
                        if ctx.y == null then
                            res = ctx.x;
                        end
                   until not ctx.next()
                   ctx.emit(res);
                end
                ''')
        rows = self.query('''select * from (select sql_null_equals_lua_nil_set_emits(x,y) from (values (1,1),(2,2),(3,3),(4,cast(NULL as number))) as t(x,y) where y is null)''')
        self.assertRowEqual((4,), rows[0]) 


    def test_sql_null_equals_lua_nil_set_emits2_1(self):
        self.query('''
                CREATE or replace lua SET SCRIPT
                sql_null_equals_lua_nil_set_emits2_1(i double, x double) EMITS (r boolean) AS
                function run(ctx)
                    local res = false;
                    repeat
                        if ctx.x == nil then
                            res = true;
                          else
                            res = false;
                        end
                   until not ctx.next()
                   ctx.emit(res);
                end
                ''')
        rows = self.query('''select * from (select sql_null_equals_lua_nil_set_emits2_1(i, x order by i asc) from (values (1,1),(2,2),(3,3),(4,cast(null as number))) as t(i,x))''')
        self.assertRowEqual((False,), rows[0])

    def test_sql_null_equals_lua_nil_set_emits2_2(self):
        self.query('''
                CREATE or replace lua SET SCRIPT
                sql_null_equals_lua_nil_set_emits2_2(i double, x double) EMITS (r boolean) AS
                function run(ctx)
                    local res = false;
                    repeat
                        if ctx.x == nil then
                            res = true;
                          else
                            res = false;
                        end
                   until not ctx.next()
                   ctx.emit(res);
                end
                ''')
        rows = self.query('''select * from (select sql_null_equals_lua_nil_set_emits2_2(i, x order by i desc) from (values (1,1),(2,2),(3,3),(4,cast(null as number))) as t(i,x))''')
        self.assertRowEqual((False,), rows[0])




        #####################################################
        #####################################################
        #####################################################



    def test_lua_null_equals_lua_nil_scalar_returns(self):
        self.query('''
                CREATE or replace lua scalar SCRIPT
                lua_null_equals_lua_nil_scalar_returns_scalar_returns(x double) returns boolean AS


                function startsWith(String, Start)
                   return string.sub(String, 1, string.len(Start)) == Start
                end

                function run(ctx)
                    if ctx.x >= 0 then
                       return (null == nil and startsWith(tostring(null), "userdata:"));
                    else
                       return (null ~= nil and startsWith(tostring(null), "userdata:"));
                    end
                end
                ''')
        rows = self.query('''select lua_null_equals_lua_nil_scalar_returns_scalar_returns(1) from dual''')
        self.assertRowEqual((False,), rows[0])
        rows = self.query('''select lua_null_equals_lua_nil_scalar_returns_scalar_returns(-1) from dual''')
        self.assertRowEqual((True,), rows[0])




        #####################################################
        #####################################################
        #####################################################

    def test_lua_nils_and_nulls_are_sql_nulls(self):  
        self.query('''
                CREATE or REPLACE lua SCALAR SCRIPT 
                    lua_nils_and_nulls_are_sql_nulls(x double) returns varchar(100) AS
                    function run(ctx)
                        if ctx.x < 0 then return nil end
                        if ctx.x == 0 then return null end
                        return "+++" 
                    end
                    ''')
        rows = self.query('''select x from (values (1,1), (2,-3), (3,0), (4,12)) as t(x,y) where lua_nils_and_nulls_are_sql_nulls(y) is null''')
        self.assertRowsEqual([(2,), (3,)], sorted(rows))

        #####################################################
        #####################################################
        #####################################################



if __name__ == '__main__':
    udf.main()

# vim: ts=4:sts=4:sw=4:et:fdm=indent

