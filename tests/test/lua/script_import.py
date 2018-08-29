#!/usr/bin/env python2.7

import os
import string
import sys

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf
from udf import useData, expectedFailure, skip

class ScriptImport(udf.TestCase):

    def setUp(self):
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('DROP SCHEMA FN3 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')
        self.query('CREATE SCHEMA FN3')
        self.query('OPEN SCHEMA FN2')
    
        self.query(udf.fixindent('''
            CREATE lua SCALAR SCRIPT
            bottom()
            RETURNS DOUBLE AS

            function f()
                return 42
            end
            /
            '''))

    def test_import_works(self):
        self.query(udf.fixindent('''
            CREATE lua SCALAR SCRIPT
            foo()
            RETURNS DOUBLE AS

            exa.import('bottom', 'b')

            function run(ctx)
                return b.f()
            end
            /
            '''))
        rows = self.query('SELECT foo() FROM DUAL')
        self.assertRowsEqual([(42,)], rows)

    def test_import_works_with_default_name(self):
        self.query(udf.fixindent('''
            CREATE lua SCALAR SCRIPT
            foo()
            RETURNS DOUBLE AS

            exa.import('bottom')

            function run(ctx)
                return bottom.f()
            end
            /
            '''))
        rows = self.query('SELECT foo() FROM DUAL')
        self.assertRowsEqual([(42,)], rows)

    def test_import_in_function_works(self):
        self.query(udf.fixindent('''
            CREATE lua SCALAR SCRIPT
            foo()
            RETURNS DOUBLE AS

            function run(ctx)
                exa.import('bottom', 'b')
                return b.f()
            end
            /
            '''))
        rows = self.query('SELECT foo() FROM DUAL')
        self.assertRowsEqual([(42,)], rows)

    def test_import_is_semi_case_sensitive(self):
        def check(name, n):
            self.query(udf.fixindent('''
                CREATE OR REPLACE lua SCALAR SCRIPT
                foo()
                RETURNS DOUBLE AS

                function run(ctx)
                    exa.import(%s, 'm')
                    return m.f()
                end
                /''' % name))
            self.assertRowsEqual([(n,)],
                self.query('SELECT foo() FROM DUAL'))

        for name in 'bar', 'Bar', 'BAR':
            self.query(udf.fixindent('''
                CREATE lua SCALAR SCRIPT
                "%s"()
                RETURNS DOUBLE AS

                function f()
                    return %d
                end
                /''' % (name, sum(x in string.uppercase for x in name))
                ))

        check("'bar'", 3)
        check("'Bar'", 3)
        check("'Bar'", 3)
        check("'\"Bar\"'", 1)
        check("'\"bar\"'", 0)

    def test_import_error_is_catchable(self):
        self.query(udf.fixindent('''
            CREATE lua SCALAR SCRIPT
            foo()
            RETURNS DOUBLE AS

            function run(ctx)
                if not pcall(function()
                        exa.import('unknown_module')
                    end)
                then
                    return 4711
                end
            end
            /
            '''))
        rows = self.query('SELECT foo() FROM DUAL')
        self.assertRowsEqual([(4711,)], rows)

    def test_import_fails_for_python_script(self):
        self.query(udf.fixindent('''
            CREATE lua SCALAR SCRIPT
            foo()
            RETURNS DOUBLE AS

            exa.import('bar')

            function run(ctx)
                return bar.f()
            end
            /
            '''))
        self.query(udf.fixindent('''
            CREATE python SCALAR SCRIPT
            bar()
            RETURNS DOUBLE AS

            def f(): 
                return 32
            /
            '''))
        with self.assertRaisesRegexp(Exception, 'script .* wrong language PYTHON'):
            self.query('SELECT foo() FROM DUAL')
        
    def test_import_fails_for_r_script(self):
        self.query(udf.fixindent('''
            CREATE lua SCALAR SCRIPT
            foo()
            RETURNS DOUBLE AS

            exa.import('bar')

            function run(ctx)
                return bar.f()
            end
            /
            '''))
        self.query(udf.fixindent('''
            CREATE r SCALAR SCRIPT
            bar()
            RETURNS DOUBLE AS

            f <- function() { 
                32
            }
            /
            '''))
        with self.assertRaisesRegexp(Exception, 'script .* wrong language R'):
            self.query('SELECT foo() FROM DUAL')

    # ------------------------------------------------------------------------------------

    def test_access_parameter_array_of_old_scripting_in_new_lua_udf(self):
        self.query(udf.fixindent('''
            CREATE or replace script scrscr(p1,p2) returns table AS
               
                function fff()
                    return 37
                end

                local lp1 = fff(p1)
                local lp2 = p2

            /
            '''))
        self.query(udf.fixindent('''
            CREATE or replace lua scalar script 
            foo()
            returns varchar(200) as
            exa.import('scrscr', 's')
            function run(ctx)
                if (s.fff() == 37) then 
                    if (s.lp1 == nil) then s.lp1 = 7 else return("fail") end
                    s.p2 = 3
                    if (s.p2 == 3) then s.lp2 = s.p2 else return("fail") end
                    return(tostring(s.lp1) .. tostring(s.lp2) .. tostring(s.fff(s.p1)) .. tostring(s.p1))
                else
                    return("fail")
                end
            end
            /
            '''))
        self.assertEqual('7337nil', self.query('''select foo() from dual''')[0][0])

    def test_importing_parameter_array_of_old_scripting_fails_in_new_lua_udf_with_old_initialization(self):
        self.query(udf.fixindent('''
            CREATE or replace script scrscr2(p1,p2) returns table AS
               
                function fff()
                    return 37
                end

                local lp1 = fff(p1)
                local lp2 = p2

            /
            '''))
        self.query(udf.fixindent('''
            CREATE or replace lua scalar script 
            foo2()
            returns varchar(200) as
            exa.import('scrscr2', 's', {1,2})
            function run(ctx)
                    return("fail")
            end
            /
            '''))
        with self.assertRaisesRegexp(Exception, 'import function, too much arguments'):
            self.query('SELECT foo2() FROM DUAL')

    # ------------------------------------------------------------------------------------

    @skip('expected result unclear')
    def test_imported_scripts_are_cached(self):
        self.query(udf.fixindent('''
            CREATE lua SCALAR SCRIPT
            foo()
            RETURNS BOOLEAN AS

            function run(ctx):
                exa.import('bottom', 'a')
                exa.import('bottom', 'b')

                return a === b
            /
            '''))
        rows = self.query('SELECT foo() FROM DUAL')
        self.assertRowsEqual([(True,)], rows)
   
    @useData([
        ('fn2', 'bottom'),
        ('fn2', 'fn2.bottom'),
        ('fn2', 'exa_db.fn2.bottom'),
        ('fn3', 'fn2.bottom'),
        ('fn3', 'exa_db.fn2.bottom')
        ])
    def test_import_works_with_qualified_names(self, schema, name): 
        self.query('OPEN SCHEMA %s' % schema)
        self.query(udf.fixindent('''
            CREATE lua SCALAR SCRIPT
            foo()
            RETURNS DOUBLE AS

            exa.import('%s', 'b')

            function run(ctx)
                return b.f()
            end
            /
            ''' % name))
        rows = self.query('SELECT foo() FROM DUAL')
        self.assertRowsEqual([(42,)], rows)
    
    def test_chained_import_works_via_function_call(self):
        self.query(udf.fixindent('''
            CREATE lua SCALAR SCRIPT
            foo()
            RETURNS DOUBLE AS

            exa.import('bar')

            function run(ctx)
                return bar.b()
            end
            /
            '''))
        self.query(udf.fixindent('''
            CREATE lua SCALAR SCRIPT
            bar()
            RETURNS DOUBLE AS

            exa.import('bottom')

            function b()
                return bottom.f()
            end
            /
            '''))
        rows = self.query('SELECT foo() FROM DUAL')
        self.assertRowsEqual([(42,)], rows)

    def test_chained_import_works_via_chained_call(self):
        self.query(udf.fixindent('''
            CREATE lua SCALAR SCRIPT
            foo()
            RETURNS DOUBLE AS

            exa.import('bar')

            function run(ctx)
                return bar.bottom.f()
            end
            /
            '''))
        self.query(udf.fixindent('''
            CREATE lua SCALAR SCRIPT
            bar()
            RETURNS DOUBLE AS

            exa.import('bottom')
            /
            '''))
        rows = self.query('SELECT foo() FROM DUAL')
        self.assertRowsEqual([(42,)], rows)

    def test_chained_import_respects_namespace(self):
        self.query(udf.fixindent('''
            CREATE lua SCALAR SCRIPT
            foo()
            RETURNS DOUBLE AS

            exa.import('bar')

            function run(ctx)
                return bottom.f()
            end
            /
            '''))
        self.query(udf.fixindent('''
            CREATE lua SCALAR SCRIPT
            bar()
            RETURNS DOUBLE AS

            exa.import('bottom')
            /
            '''))
        with self.assertRaisesRegexp(Exception, 'nil value'):
            self.query('SELECT foo() FROM DUAL')

    def test_chained_import_respects_namespace_when_names_are_the_same(self):
        self.query(udf.fixindent('''
            CREATE lua SCALAR SCRIPT
            foo()
            RETURNS DOUBLE AS

            exa.import('bar', 'f')

            function run(ctx)
                return f.f.f()
            end
            /
            '''))
        self.query(udf.fixindent('''
            CREATE lua SCALAR SCRIPT
            bar()
            RETURNS DOUBLE AS

            exa.import('bottom', 'f')
            /
            '''))
        rows = self.query('SELECT foo() FROM DUAL')
        self.assertRowsEqual([(42,)], rows)

    @expectedFailure
    def test_mutual_import_works(self):
        '''DWA-13847'''
        self.query(udf.fixindent('''
            CREATE lua SCALAR SCRIPT
            ping()
            RETURNS DOUBLE AS

            exa.import('pong')

            function ping(n)
                if n > 0 then
                    return pong.pong(n-1) + 1
                else
                    return 0
                end
            end

            function run(ctx)
                return ping(42)
            end
            /
            '''))
        self.query(udf.fixindent('''
            CREATE lua SCALAR SCRIPT
            pong()
            RETURNS DOUBLE AS

            exa.import('ping')

            function pong(n)
                if n > 0 then
                    return ping.ping(n-1) + 1
                else
                    return 0
                end
            end
            /
            '''))
        rows = self.query('SELECT ping() FROM DUAL')
        self.assertRowsEqual([(42,)], rows)
    
    def testImportWithView(self):
        self.createUser("foo", "foo")
        self.commit()
        self.query(udf.fixindent('''
            CREATE lua SCALAR SCRIPT
            spot42542script()
            RETURNS DOUBLE AS

            exa.import('bottom', 'b')

            function run(ctx)
                return b.f()
            end
            /
            '''))
        self.query("create or replace view FN2.spot42542_view as SELECT spot42542script() as col FROM DUAL")
        self.query("grant select on FN2.spot42542_view to foo")
        self.commit()
        rows = self.query('''select * from FN2.spot42542_view''')
        self.assertRowsEqual([(42,)], rows)
        foo_conn = self.getConnection('foo', 'foo')
        rows = foo_conn.query('''select * from FN2.spot42542_view''')
        self.assertRowsEqual([(42,)], rows)

if __name__ == '__main__':
    udf.main()

# vim: ts=4:sts=4:sw=4:et:fdm=indent

