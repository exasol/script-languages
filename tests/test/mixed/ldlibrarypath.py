#!/usr/opt/bs-python-2.7/bin/python

import os
import sys
import decimal

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf

class LDLibraryPathPython(udf.TestCase):
    def checkType(self, query, type):
        self.query('''create or replace table tmp as  ''' + query)
        rows = self.query('describe tmp')
        self.assertEqual(rows[0][1], type)


    def setUp(self):
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SCALAR SCRIPT python_simple() returns varchar(1000) AS
            %env LD_LIBRARY_PATH=/buckets/to/heaven;
            import os
            def run(ctx):
                return os.environ["LD_LIBRARY_PATH"]
            /
            '''))

        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SCALAR SCRIPT imported() returns varchar(1000) AS
            %env SOME_VAR_IMPORTED=some_value_imported;
            imported_val="some_imported_value"
           '''))


        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SCALAR SCRIPT import_direct() returns varchar(1000) AS
            %env SOME_VAR=some_value;
            %import "FN2.IMPORTED";
            import os
            def run(ctx):
                return "import_direct:"+imported_val
            /
            '''))


        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SCALAR SCRIPT import_mod() returns varchar(1000) AS
            %env SOME_VAR=some_value;
            im = exa.import_script("FN2.IMPORTED");
            import os
            def run(ctx):
                return "import_direct:"+im.imported_val
            /
            '''))


        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SCALAR SCRIPT python_simple2() returns varchar(1000) AS
            %env LD_LIBRARY_PATH1=/buckets/to/heaven;
            import os
            def run(ctx):
                return os.environ["LD_LIBRARY_PATH1"]
            /
            '''))


    # Please note that we cannot test LD_LIBRARY_PATH here as
    # in our test system, nsexec_chroot is installed in NFS which
    # does not not support file capabilities like "chroot" and therefore
    # is installed with SUID bit which in turn causes Linux to prevent it
    # from setting LD_LIBRARY_PATH.
    # So this test should raise an exception (while it is fine in production) systems.

    def test_pythonSimpleLdlibraryPath(self):
        if os.path.isfile("/usr/opt/environ/bin/nsexec_chroot"):
            with self.assertRaisesRegexp(Exception, r"KeyError"):
                rows = self.query('''select fn2.python_simple()''')
        else:
            rows = self.query('''select fn2.python_simple2()''')
            self.assertRowsEqual([("/buckets/to/heaven",)],rows)

    # Variables other than LD_LIBRARY_PATH should work:
    def test_pythonSimpleLdlibraryPath2(self):
        rows = self.query('''select fn2.python_simple2()''')
        self.assertRowsEqual([("/buckets/to/heaven",)],rows)

    def test_exceptionWhenNoEqualSign(self):
        with self.assertRaisesRegexp(Exception, r"Script option %env does not have the form VAR=VALUE;"):
            self.query(udf.fixindent('''
                CREATE OR REPLACE PYTHON SCALAR SCRIPT python_simple_error() returns varchar(1000) AS
                %env LD_LIBRARY_PATH=/buckets/to/heaven;
                %env BIG_GREAT_WORLD;
                import os
                def run(ctx):
                    return os.environ["LD_LIBRARY_PATH"]
                /
                '''))

    def test_importScript_dynamic(self):
        rows = self.query('''select import_mod()''')
        self.assertRowsEqual([("import_direct:some_imported_value",)],rows)

    def test_importScript_static(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE java SCALAR SCRIPT
            IMPORTED_X()
            RETURNS int AS
            class IMPORTED_X {
                static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return 1;
                }
            }
            '''))

        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                IMPORTING_X()
                RETURNS int AS
                %import	IMPORTED_X;
                class IMPORTING_X {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return 1+IMPORTED_X.run(exa, ctx);
                    }
                }
                '''))
        rows = self.query('SELECT importing_x()')
        self.assertRowsEqual([(2,)], rows)


if __name__ == '__main__':
    udf.main()
