#!/usr/bin/env python2.7
# encoding: utf8

import os
import sys
import datetime

sys.path.append(os.path.realpath(__file__ + '/../../../../lib'))

import udf

class EnvDirective(udf.TestCase):
    ### Basic Functionality Test of %env directive in Python3 UDF Scripts

    def setUp(self):
        self.query('DROP SCHEMA EnvDirectiveSchema CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA EnvDirectiveSchema')

    # Set and Retrieve ENVIRONMENT_VARIABLE_VALUE.
    def test_env_set_and_use_env_var(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SCALAR SCRIPT env_var_test() returns varchar(1000) AS
            %env ENV_VAR=ENVIRONMENT_VARIABLE_VALUE;
            import os
            def run(ctx):
                return os.environ["ENV_VAR"]
            /
            '''))
        rows = self.query('''select EnvDirectiveSchema.env_var_test()''')
        self.assertRowsEqual([('ENVIRONMENT_VARIABLE_VALUE',)],rows)

    # %env directive ends without semicolon. Parsing Error should occur.
    def test_env_set_env_var_without_semicolon(self):
        with self.assertRaisesRegexp(Exception, r'Error while parsing %env option line'):
            self.query(udf.fixindent('''
                CREATE OR REPLACE PYTHON3 SCALAR SCRIPT env_var_test() returns varchar(1000) AS
                %env ENV_VAR="ENVIRONMENT_VARIABLE_VALUE"
                import os
                def run(ctx):
                    return os.environ["ENV_VAR"]
                /
                '''))

    # %env directive is commented out in Python3 style. KeyError expected.
    def test_env_set_env_var_python3_comment(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SCALAR SCRIPT env_var_test() returns varchar(1000) AS
            # %env ENV_VAR=ENVIRONMENT_VARIABLE_VALUE;
            import os
            def run(ctx):
                return os.environ["ENV_VAR"]
            /
            '''))
        with self.assertRaisesRegexp(Exception, 'KeyError'):
            rows = self.query('''select EnvDirectiveSchema.env_var_test()''')

    # %env directive is commented out NOT in Python3 style. Python3 SyntaxError expected.
    def test_env_set_env_var_not_python_comment(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SCALAR SCRIPT env_var_test() returns varchar(1000) AS
            // %env ENV_VAR=ENVIRONMENT_VARIABLE_VALUE;
            import os
            def run(ctx):
                return os.environ["ENV_VAR"]
            /
            '''))
        with self.assertRaisesRegexp(Exception, 'invalid syntax'):
            rows = self.query('''select EnvDirectiveSchema.env_var_test()''')

    # %env directive is appended after a source code line. Python3 SyntaxError expected.
    def test_env_set_env_var_after_code(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SCALAR SCRIPT env_var_test() returns varchar(1000) AS
            import os     %env ENV_VAR=ENVIRONMENT_VARIABLE_VALUE;
            def run(ctx):
                return os.environ["ENV_VAR"]
            /
            '''))
        with self.assertRaisesRegexp(Exception, 'invalid syntax'):
            rows = self.query('''select EnvDirectiveSchema.env_var_test()''')

    # %env directive is Python comment and appended after a source code line. KeyError expected.
    def test_env_set_env_var_python_comment_after_code(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SCALAR SCRIPT env_var_test() returns varchar(1000) AS
            import os    # %env ENV_VAR=ENVIRONMENT_VARIABLE_VALUE;
            def run(ctx):
                return os.environ["ENV_VAR"]
            /
            '''))
        with self.assertRaisesRegexp(Exception, 'KeyError'):
            rows = self.query('''select EnvDirectiveSchema.env_var_test()''')

if __name__ == '__main__':
    udf.main()

# vim: ts=4:sts=4:sw=4:et:fdm=indent
