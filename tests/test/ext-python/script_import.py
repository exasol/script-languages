#!/usr/opt/bs-python-2.7/bin/python

import os
import string
import sys

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf
from udf import useData

class ScriptImport(udf.TestCase):

    def setUp(self):
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('DROP SCHEMA FN3 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')
        self.query('CREATE SCHEMA FN3')
        self.query('OPEN SCHEMA FN2')
    
        self.query(udf.fixindent('''
            CREATE EXTERNAL SCALAR SCRIPT
            bottom()
            RETURNS INT AS
            # redirector @@redirector_url@@

            def f():
                return 42
            /
            '''))

    def test_import_works(self):
        self.query(udf.fixindent('''
            CREATE EXTERNAL SCALAR SCRIPT
            foo()
            RETURNS INT AS
            # redirector @@redirector_url@@

            b = exa.import_script('bottom')

            def run(ctx):
                return b.f()
            /
            '''))
        rows = self.query('SELECT foo() FROM DUAL')
        self.assertRowsEqual([(42,)], rows)

    def test_import_is_semi_case_sensitive(self):
        def check(name, n):
            self.query(udf.fixindent('''
                CREATE OR REPLACE EXTERNAL SCALAR SCRIPT
                foo()
                RETURNS INT AS
                # redirector @@redirector_url@@

                def run(ctx):
                    m = exa.import_script(%s)
                    return m.f()
                /''' % name))
            self.assertRowsEqual([(n,)],
                self.query('SELECT foo() FROM DUAL'))

        for name in 'bar', 'Bar', 'BAR':
            self.query(udf.fixindent('''
                CREATE EXTERNAL SCALAR SCRIPT
                "%s"()
                RETURNS INT AS
                # redirector @@redirector_url@@

                def f():
                    return %d
                /''' % (name, sum(x in string.uppercase for x in name))
                ))

        check("'bar'", 3)
        check("'Bar'", 3)
        check("'Bar'", 3)
        check("'\"Bar\"'", 1)
        check("'\"bar\"'", 0)

    def test_import_error_is_catchable(self):
        self.query(udf.fixindent('''
            CREATE EXTERNAL SCALAR SCRIPT
            foo()
            RETURNS INT AS
            # redirector @@redirector_url@@

            def run(ctx):
                try:
                    exa.import_script('unknown_module')
                except ImportError:
                    return 4711
            /
            '''))
        rows = self.query('SELECT foo() FROM DUAL')
        self.assertRowsEqual([(4711,)], rows)

    def test_import_fails_for_lua_script(self):
        self.query(udf.fixindent('''
            CREATE EXTERNAL SCALAR SCRIPT
            foo()
            RETURNS INT AS
            # redirector @@redirector_url@@

            bar = exa.import_script('bar')

            def run(ctx):
                return bar.f()
            /
            '''))
        self.query(udf.fixindent('''
            CREATE lua SCALAR SCRIPT
            bar()
            RETURNS DOUBLE AS

            function f() 
                return 32
            end
            /
            '''))
        with self.assertRaisesRegexp(Exception, 'ImportError:.* wrong language LUA'):
            self.query('SELECT foo() FROM DUAL')
        
    def test_import_fails_for_r_script(self):
        self.query(udf.fixindent('''
            CREATE EXTERNAL SCALAR SCRIPT
            foo()
            RETURNS INT AS
            # redirector @@redirector_url@@

            bar = exa.import_script('bar')

            def run(ctx):
                return bar.f()
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
        with self.assertRaisesRegexp(Exception, 'ImportError:.* wrong language R'):
            self.query('SELECT foo() FROM DUAL')
        

    def test_imported_scripts_are_cached(self):
        self.query(udf.fixindent('''
            CREATE EXTERNAL SCALAR SCRIPT
            foo()
            RETURNS BOOLEAN AS
            # redirector @@redirector_url@@

            def run(ctx):
                a = exa.import_script('bottom')
                b = exa.import_script('bottom')

                return a is b
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
            CREATE EXTERNAL SCALAR SCRIPT
            foo()
            RETURNS INT AS
            # redirector @@redirector_url@@

            b = exa.import_script('%s')

            def run(ctx):
                return b.f()
            /
            ''' % name))
        rows = self.query('SELECT foo() FROM DUAL')
        self.assertRowsEqual([(42,)], rows)
    
    def test_chained_import_works_via_function_call(self):
        self.query(udf.fixindent('''
            CREATE EXTERNAL SCALAR SCRIPT
            foo()
            RETURNS INT AS
            # redirector @@redirector_url@@

            bar = exa.import_script('bar')

            def run(ctx):
                return bar.b()
            /
            '''))
        self.query(udf.fixindent('''
            CREATE EXTERNAL SCALAR SCRIPT
            bar()
            RETURNS INT AS
            # redirector @@redirector_url@@

            bottom = exa.import_script('bottom')

            def b():
                return bottom.f()
            /
            '''))
        rows = self.query('SELECT foo() FROM DUAL')
        self.assertRowsEqual([(42,)], rows)

    def test_chained_import_works_via_chained_call(self):
        self.query(udf.fixindent('''
            CREATE EXTERNAL SCALAR SCRIPT
            foo()
            RETURNS INT AS
            # redirector @@redirector_url@@

            bar = exa.import_script('bar')

            def run(ctx):
                return bar.bottom.f()
            /
            '''))
        self.query(udf.fixindent('''
            CREATE EXTERNAL SCALAR SCRIPT
            bar()
            RETURNS INT AS
            # redirector @@redirector_url@@

            bottom = exa.import_script('bottom')
            /
            '''))
        rows = self.query('SELECT foo() FROM DUAL')
        self.assertRowsEqual([(42,)], rows)

    def test_mutual_import_works(self):
        self.query(udf.fixindent('''
            CREATE EXTERNAL SCALAR SCRIPT
            ping()
            RETURNS INT AS
            # redirector @@redirector_url@@

            pong = exa.import_script('pong')

            def ping(n):
                if n > 0:
                    return pong.pong(n-1) + 1
                else:
                    return 0

            def run(ctx):
                return ping(42)
            /
            '''))
        self.query(udf.fixindent('''
            CREATE EXTERNAL SCALAR SCRIPT
            pong()
            RETURNS INT AS
            # redirector @@redirector_url@@

            ping = exa.import_script('ping')

            def pong(n):
                if n > 0:
                    return ping.ping(n-1) + 1
                else:
                    return 0
            /
            '''))
        rows = self.query('SELECT ping() FROM DUAL')
        self.assertRowsEqual([(42,)], rows)
    

if __name__ == '__main__':
    udf.main()

# vim: ts=4:sts=4:sw=4:et:fdm=indent

