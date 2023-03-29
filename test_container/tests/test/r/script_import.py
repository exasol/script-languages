#!/usr/bin/env python3

from exasol_python_test_framework import udf
from exasol_python_test_framework.udf import useData


class ScriptImport(udf.TestCase):

    def setUp(self):
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('DROP SCHEMA FN3 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')
        self.query('CREATE SCHEMA FN3')
        self.query('OPEN SCHEMA FN2')

        self.query(udf.fixindent('''
            CREATE r SCALAR SCRIPT
            bottom()
            RETURNS INT AS

            f <- function() {
                42
            }
            /
            '''))

    def test_import_works(self):
        self.query(udf.fixindent('''
            CREATE r SCALAR SCRIPT
            foo()
            RETURNS INT AS

            b <- exa$import_script('bottom')

            run <- function(ctx) {
                b$f()
            }
            /
            '''))
        rows = self.query('SELECT foo() FROM DUAL')
        self.assertRowsEqual([(42,)], rows)

    def test_import_is_semi_case_sensitive(self):
        def check(script_name, n):
            self.query(udf.fixindent('''
                CREATE OR REPLACE r SCALAR SCRIPT
                foo()
                RETURNS INT AS

                run <- function(ctx) {
                    m <- exa$import_script(%s)
                    m$f()
                }
                /''' % script_name))
            self.assertRowsEqual([(n,)],
                                 self.query('SELECT foo() FROM DUAL'))

        for name in 'bar', 'Bar', 'BAR':
            self.query(udf.fixindent('''
                CREATE r SCALAR SCRIPT
                "%s"()
                RETURNS INT AS

                f <- function() {
                    %d
                }
                /''' % (name, sum(x.isupper() for x in name))
                                     ))

        check("'bar'", 3)
        check("'Bar'", 3)
        check("'Bar'", 3)
        check("'\"Bar\"'", 1)
        check("'\"bar\"'", 0)

    def test_import_error_is_catchable(self):
        self.query(udf.fixindent('''
            CREATE r SCALAR SCRIPT
            foo()
            EMITS (d VARCHAR(1000), val INT) AS

            safe_import <- function(module) {
                tryCatch(exa$import_script(module), error=function(e) NULL)
            }
                

            run <- function(ctx) {
                a <- safe_import('unknown_module')
                ctx$emit('unknown_module', as.integer(is.null(a)))
                b <- safe_import('bottom')
                ctx$emit('bottom', as.integer(is.null(b)))
                ctx$emit('bottom_value', b$f())
            }
            /
            '''))
        rows = self.query('SELECT foo() FROM DUAL')
        self.assertRowsEqual([
            ('bottom', 0),
            ('bottom_value', 42),
            ('unknown_module', 1)],
            sorted(rows))

    def test_import_fails_for_lua_script(self):
        self.query(udf.fixindent('''
            CREATE r SCALAR SCRIPT
            foo()
            RETURNS INT AS

            bar <- exa$import_script('bar')

            run <- function(ctx) {
                bar$f()
            }
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
        with self.assertRaisesRegex(Exception, 'Error .* wrong language LUA'):
            self.query('SELECT foo() FROM DUAL')

    def test_import_fails_for_python_script(self):
        self.query(udf.fixindent('''
            CREATE r SCALAR SCRIPT
            foo()
            RETURNS INT AS

            bar <- exa$import_script('bar')

            run <- function(ctx) {
                bar$f()
            }
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
        with self.assertRaisesRegex(Exception, 'Error .* wrong language PYTHON'):
            self.query('SELECT foo() FROM DUAL')

    def test_imported_scripts_are_cached(self):
        self.query(udf.fixindent('''
            CREATE r SCALAR SCRIPT
            foo()
            RETURNS BOOLEAN AS

            run <- function(ctx) {
                a <- exa$import_script('bottom')
                b <- exa$import_script('bottom')
                identical(a, b)
            }
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
            CREATE r SCALAR SCRIPT
            foo()
            RETURNS INT AS

            b <- exa$import_script('%s')

            run <- function(ctx) {
                b$f()
            }
            /
            ''' % name))
        rows = self.query('SELECT foo() FROM DUAL')
        self.assertRowsEqual([(42,)], rows)

    def test_chained_import_works_via_function_call(self):
        self.query(udf.fixindent('''
            CREATE r SCALAR SCRIPT
            foo()
            RETURNS INT AS

            bar <- exa$import_script('bar')

            run <- function(ctx) {
                bar$b()
            }
            /
            '''))
        self.query(udf.fixindent('''
            CREATE r SCALAR SCRIPT
            bar()
            RETURNS INT AS

            bottom <- exa$import_script('bottom')

            b <- function() {
                bottom$f()
            }
            /
            '''))
        rows = self.query('SELECT foo() FROM DUAL')
        self.assertRowsEqual([(42,)], rows)

    def test_chained_import_works_via_chained_call(self):
        self.query(udf.fixindent('''
            CREATE r SCALAR SCRIPT
            foo()
            RETURNS INT AS

            bar <- exa$import_script('bar')

            run <- function(ctx) {
                bar$bottom$f()
            }
            /
            '''))
        self.query(udf.fixindent('''
            CREATE r SCALAR SCRIPT
            bar()
            RETURNS INT AS

            bottom <- exa$import_script('bottom')
            /
            '''))
        rows = self.query('SELECT foo() FROM DUAL')
        self.assertRowsEqual([(42,)], rows)

    def test_mutual_import_works(self):
        self.query(udf.fixindent('''
            CREATE r SCALAR SCRIPT
            ping()
            RETURNS INT AS

            pong <- exa$import_script('pong')

            ping <- function(n) {
                if (n > 0) {
                    pong$pong(n-1) + 1
                } else {
                    0
                }
            }

            run <- function(ctx) {
                ping(42)
            }
            /
            '''))
        self.query(udf.fixindent('''
            CREATE r SCALAR SCRIPT
            pong()
            RETURNS INT AS

            ping <- exa$import_script('ping')

            pong <- function(n) {
                if (n > 0) {
                    ping$ping(n-1) + 1
                } else {
                    0
                }
            }

            /
            '''))
        rows = self.query('SELECT ping() FROM DUAL')
        self.assertRowsEqual([(42,)], rows)

    def testImportWithView(self):
        self.createUser("foo", "foo")
        self.commit()
        self.query(udf.fixindent('''
            CREATE r SCALAR SCRIPT
            spot42542script()
            RETURNS INT AS

            b <- exa$import_script('bottom')

            run <- function(ctx) {
                b$f()
            }
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
