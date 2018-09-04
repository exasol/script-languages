#!/usr/bin/env python2.7

import os
import sys

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf
from udf import useData, expectedFailure

class PythonInterpreter(udf.TestCase):
    def setUp(self):
        self.query('CREATE SCHEMA FN2', ignore_errors=True)
        self.query('OPEN SCHEMA FN2', ignore_errors=True)

    def test_body_is_not_executed_at_creation_time(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE python SCALAR SCRIPT
                sleep()
                RETURNS int AS

                raise ValueError()

                def run(ctx):
                    ctx.emit(42)
                /
                '''))

    def test_syntax_errors_not_caught_at_creation_time(self):
        sql = udf.fixindent('''
                CREATE OR REPLACE python SCALAR SCRIPT
                syntax_error()
                RETURNS int AS

                def run(ctx)
                    ctx.emit(42)
                /
                ''')
        self.query(sql)

    def test_methods_have_access_to_global_context(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE python SCALAR SCRIPT
                global_fn()
                RETURNS int AS

                foo = 2

                def bar():
                    global foo
                    foo = 5

                def run(ctx):
                    bar()
                    return foo
                '''))
        row = self.query('SELECT global_fn() FROM DUAL')[0]
        self.assertEqual(5, row[0])

    def test_exception_in_cleanup_is_propagated(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE python SCALAR SCRIPT
            foo()
            RETURNS INT AS

            def run(ctx):
                return 42

            def cleanup():
                raise ValueError('4711')
            '''))
        with self.assertRaisesRegexp(Exception, '4711'):
            self.query('SELECT foo() FROM dual')

    def test_cleanup_has_global_context(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE python SCALAR SCRIPT
            foo()
            RETURNS INT AS

            flag = 21

            def run(ctx):
                global flag
                flag = 4711
                return 42

            def cleanup():
                raise ValueError(flag)
            '''))
        with self.assertRaisesRegexp(Exception, '4711'):
            rows = self.query('SELECT foo() FROM dual')
            self.assertRowsEqual([(42,)], rows)

class PythonImport(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')

    modules = '''
        calendar
        cmath
        collections
        contextlib
        copy
        csv
        datetime
        decimal
        functools
        hashlib
        hmac
        itertools
        locale
        math
        os
        random
        re
        shlex
        string
        sys
        textwrap
        time
        unicodedata
        uuid
        weakref
        '''.split()

    @useData((x,) for x in modules)
    def test_stdlib_modules_are_importable(self, module):
        sql = udf.fixindent('''
            CREATE OR REPLACE python SCALAR SCRIPT
            i()
            RETURNS VARCHAR(1024) AS

            import %(module)s

            def run(ctx):
                try:
                    return %(module)s.__file__
                except AttributeError:
                    return "(built-in)"
            ''')
        self.query(sql % {'module': module})
        self.query('SELECT i() FROM DUAL')

    modules = '''
            ujson
            lxml
            numpy
            pytz
            scipy
            '''.split()
    @useData((x,) for x in modules)
    def test_3rdparty_modules_are_importable(self, module):
        sql = udf.fixindent('''
            CREATE OR REPLACE python SCALAR SCRIPT
            i()
            RETURNS VARCHAR(1024) AS
            import %(module)s
            def run(ctx):
                try:
                    return %(module)s.__file__
                except AttributeError:
                    return "(built-in)"
            ''')
        self.query(sql % {'module': module})
        self.query('SELECT i() FROM DUAL')

    # modules = '''
    #         '''.split()
    # @useData((x,) for x in modules)
    # @expectedFailure
    # def test_3rdparty_modules_are_importable_negative_list(self, module):
    #     '''DWA-13850'''
    #     sql = udf.fixindent('''
    #         CREATE OR REPLACE python SCALAR SCRIPT
    #         i()
    #         RETURNS VARCHAR(1024) AS
    #         import %(module)s
    #         def run(ctx):
    #             try:
    #                 return %(module)s.__file__
    #             except AttributeError:
    #                 return "(built-in)"
    #         ''')
    #     self.query(sql % {'module': module})
    #     self.query('SELECT i() FROM DUAL')

    def test_ImportError_is_catchable(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE python SCALAR SCRIPT
                catch_import_error()
                RETURNS int AS

                try:
                    import unknown_module
                except ImportError:
                    pass

                def run(ctx):
                    return 42
                '''))
        rows = self.query('SELECT catch_import_error() FROM dual')
        self.assertRowsEqual([(42,)], rows)

    def test_import_error_message_is_case_sensitive(self):
        sql = udf.fixindent('''
                CREATE OR REPLACE python SCALAR SCRIPT
                import_error()
                RETURNS int AS

                import Unknown_Module
                def run(ctx): pass
                ''')
        with self.assertRaisesRegexp(Exception, 'Unknown_Module'):
            self.query(sql)
            self.query('SELECT import_error() FROM dual')

    def test_import_is_case_sensitive(self):
        scripts = [
                ('my_module', 4711),
                ('My_Module', 42),
                ('MY_MODULE', 1234),
                ]
        sql = udf.fixindent('''
                CREATE OR REPLACE python SCALAR SCRIPT
                "%s"()
                RETURNS int AS

                ID = %d

                def run(ctx): pass
                #raise RuntimeError(" *** ".join(globals()))        
                ''')
        for pair in scripts:
            self.query(sql % pair)

        self.query(udf.fixindent('''
                CREATE OR REPLACE python SCALAR SCRIPT
                import_case_sensitive()
                RETURNS int AS

                My_Module = exa.import_script('"My_Module"')

                def run(ctx):
                    try:
                        return My_Module.ID
                    except Exception as err:
                        raise RuntimeError("+".join(My_Module.__dict__))
                '''))
        rows = self.query('SELECT import_case_sensitive() FROM DUAL')
        self.assertRowsEqual([(42,)], rows)


class PythonSyntax(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')

    def test_sql_comments_are_ignored(self):
        self.query(udf.fixindent('''
            CREATE python SCALAR SCRIPT
            sql_comments()
            RETURNS INT AS

            def run(ctx):
                return (
                    --3
                    +4)
            '''))
        rows = self.query('SELECT sql_comments() FROM dual')
        self.assertEqual(7, rows[0][0])

    def test_python_comments_are_ignored_in_functions(self):
        self.query(udf.fixindent('''
            CREATE python SCALAR SCRIPT
            python_comments()
            RETURNS INT AS

            def run(ctx):
                return ( 19
                    #--3
                    #+4
                    )
            '''))
        rows = self.query('SELECT python_comments() FROM dual')
        self.assertEqual(19, rows[0][0])

    def test_python_comments_are_ignored_in_body(self):
        self.query(udf.fixindent('''
            CREATE python SCALAR SCRIPT
            python_comments()
            RETURNS INT AS

            #import foo

            def run(ctx):
                return 43
            '''))
        rows = self.query('SELECT python_comments() FROM dual')
        self.assertEqual(43, rows[0][0])


class PythonErrors(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')

    def create_and_execute(self, body):
        self.query(udf.fixindent('''
            CREATE OR REPLACE python SCALAR SCRIPT
            foo()
            RETURNS INT AS

            def run(ctx):
                %s
                return 42
            ''' % body))
        self.query('SELECT foo() FROM dual')

    def test_ZeroDivisionError(self):
        with self.assertRaisesRegexp(Exception, 'ZeroDivisionError'):
            self.create_and_execute('1/0')

    def test_AttributeError(self):
        with self.assertRaisesRegexp(Exception, 'AttributeError'):
            self.create_and_execute('int(4).append(3)')

    def test_TypeError(self):
        with self.assertRaisesRegexp(Exception, 'TypeError'):
            self.create_and_execute('sorted(5)')

    def test_ImportError(self):
        with self.assertRaisesRegexp(Exception, 'ModuleNotFoundError'):
            self.create_and_execute('import unknown_module')

    def test_NameError(self):
        with self.assertRaisesRegexp(Exception, 'NameError'):
            self.create_and_execute('unknown_thing')

    def test_IndexError(self):
        with self.assertRaisesRegexp(Exception, 'IndexError'):
            self.create_and_execute('range(10)[14]')

    def test_KeyError(self):
        with self.assertRaisesRegexp(Exception, 'KeyError'):
            self.create_and_execute('{}[5]')

    def test_SyntaxError(self):
        with self.assertRaisesRegexp(Exception, 'SyntaxError'):
            self.create_and_execute('45 (;')



class Robustness(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')

    def test_repeated_failed_import(self):
        '''
            internal server error after failed imports (DWA-13377)
            UDF: Python: no exception when importing unknown module + wrong result (DWA-13666)
        '''
        sql = udf.fixindent('''
            CREATE OR REPLACE python SET SCRIPT
            foo(x double)
            RETURNS int AS

            import completely_unknown_module

            def run(ctx):
                return 42
            ''')
        for _ in range(100):
            with self.assertRaisesRegexp(Exception, 'ModuleNotFoundError:'):
                self.query(sql)
                self.query('SELECT foo(NULL) from DUAL')

if __name__ == '__main__':
    udf.main()

# vim: ts=4:sts=4:sw=4:et:fdm=indent

