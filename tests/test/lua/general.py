#!/usr/opt/bs-python-2.7/bin/python

import os
import sys

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf
from udf import useData, expectedFailure

class LuaInterpreter(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')

    def test_body_is_not_executed_at_creation_time(self):
        self.query(udf.fixindent('''
                CREATE lua SCALAR SCRIPT
                body_error()
                RETURNS double AS

                if 1 == 1 then
                    error("foo bar")
                end

                function run(context)
                    return 42
                end
                /
                '''))

    def test_syntax_error_is_ignored_at_creation_time(self):
        sql = udf.fixindent('''
                CREATE lua SCALAR SCRIPT
                syntax_error()
                RETURNS double AS

                function run(context)
                    return 42
                /
                ''')

        self.query(sql)

    def test_exception_in_run_is_propagated(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE lua SCALAR SCRIPT
            foo()
            RETURNS DOUBLE AS

            function run(ctx)
                error('4711')
            end
            '''))
        with self.assertRaisesRegexp(Exception, '4711'):
            self.query('SELECT foo() FROM dual')

    def test_assertion_in_run_is_propagated(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE lua SCALAR SCRIPT
            foo()
            RETURNS DOUBLE AS

            function run(ctx)
                assert(1 == 4711)
            end
            '''))
        with self.assertRaisesRegexp(Exception, 'assertion failed'):
            self.query('SELECT foo() FROM dual')

    def test_exception_in_cleanup_is_propagated(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE lua SCALAR SCRIPT
            foo()
            RETURNS DOUBLE AS

            function run(ctx)
                return 42
            end

            function cleanup()
                error('4711')
            end
            '''))
        with self.assertRaisesRegexp(Exception, '4711'):
            self.query('SELECT foo() FROM dual')

    def test_assertion_in_cleanup_is_propagated(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE lua SCALAR SCRIPT
            foo()
            RETURNS DOUBLE AS

            function run(ctx)
                return 42
            end

            function cleanup()
                assert(1 == 4711)
            end
            '''))
        with self.assertRaisesRegexp(Exception, 'assertion failed'):
            self.query('SELECT foo() FROM dual')

    def test_cleanup_has_global_context(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE lua SCALAR SCRIPT
            foo()
            RETURNS DOUBLE AS

            flag = 21

            function run(ctx)
                flag = 4711
                return 42
            end

            function cleanup()
                if flag == 4711 then
                    error(flag)
                end
            end
            '''))
        with self.assertRaisesRegexp(Exception, '4711'):
            rows = self.query('SELECT foo() FROM dual')
            self.assertRowsEqual([(42,)], rows)

class LuaImport(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')

    modules = ['math', 'table', 'string', 'unicode', 'lxp']
    @useData((x,) for x in modules)
    def test_essential_modules_are_preloaded(self, module):
        self.query(udf.fixindent('''
            CREATE lua SCALAR SCRIPT
            batteries_included()
            RETURNS BOOLEAN AS

            function run(ctx)
                return %s ~= nil
            end
            ''' % module))
        rows = self.query('SELECT batteries_included() FROM dual')
        self.assertEqual(True, rows[0][0])
            
    modules = ['socket']
    @useData((x,) for x in modules)
    def test_extra_modules_are_loadable(self, module):
        self.query(udf.fixindent('''
            CREATE lua SCALAR SCRIPT
            batteries_included()
            RETURNS BOOLEAN AS

            a = require("%s")

            function run(ctx)
                assert(a ~= nil)
                return %s ~= nil
            end
            ''' % (module, module)))
        rows = self.query('SELECT batteries_included() FROM dual')
        self.assertEqual(True, rows[0][0])

    @expectedFailure
    def test_require_is_disabled(self):
        '''DWA-13814'''
        self.query(udf.fixindent('''
            CREATE lua SCALAR SCRIPT
            require_disabled()
            RETURNS BOOLEAN AS

            require('unicode')

            function run(ctx)
                return True
            end
            '''))
        with self.assertRaises(Exception):
            self.query('SELECT require_disabled() FROM dual')

    

class LuaSyntax(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')

    def test_sql_comments_are_ignored(self):
        self.query(udf.fixindent('''
            CREATE lua SCALAR SCRIPT
            sql_comments()
            RETURNS DOUBLE AS

            function run(ctx)
                a = (2
                    --[[
                        +4
                    --]]
                    --8
                    )
                return a
            end
            '''))
        rows = self.query('SELECT sql_comments() FROM dual')
        self.assertEqual(2, rows[0][0])



if __name__ == '__main__':
    udf.main()

# vim: ts=4:sts=4:sw=4:et:fdm=indent

