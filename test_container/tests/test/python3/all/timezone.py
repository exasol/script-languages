#!/usr/bin/env python3

from exasol_python_test_framework import udf
from exasol_python_test_framework.udf import requires


class TimeZoneTest(udf.TestCase):

    def setUp(self):
        self.query('DROP SCHEMA tz_python CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA tz_python')

    def test_default(self):
        rows = self.query('''
            SELECT DBTIMEZONE 
            FROM DUAL
            ''')
        self.assertRowsEqual([("EUROPE/BERLIN",)], rows)
        self.query(udf.fixindent('''
        create python3 SCALAR SCRIPT
        default_tz()
        RETURNS VARCHAR(100) AS
        import time
        def run(ctx):
            return time.tzname[0]
        /

        '''))
        rows = self.query('''
            SELECT tz_python.default_tz() 
            FROM DUAL
            ''')
        self.assertRowsEqual([("CET",)], rows)


    def test_set_tz(self):
        self.query(udf.fixindent('''
        create python3 SCALAR SCRIPT
        modify_tz(tzname VARCHAR(100))
        RETURNS VARCHAR(100) AS
        import time
        import os
        def run(ctx):
            os.environ["TZ"] = ctx.tzname
            time.tzset()
            return time.tzname[0]
        /
        '''))
        rows = self.query('''
            SELECT tz_python.modify_tz('America/New_York')
            FROM DUAL
            ''')
        self.assertRowsEqual([("EST",)], rows)

    def test_set_tz_via_script_option(self):
        self.query(udf.fixindent('''
        create python3 SCALAR SCRIPT
        modify_tz_via_script_option()
        RETURNS VARCHAR(100) AS
        %env TZ=America/New_York;
        import time
        import os
        def run(ctx):
            return time.tzname[0]
        /
        '''))
        rows = self.query('''
            SELECT tz_python.modify_tz_via_script_option()
            FROM DUAL
            ''')
        self.assertRowsEqual([("EST",)], rows)

if __name__ == '__main__':
    udf.main()
