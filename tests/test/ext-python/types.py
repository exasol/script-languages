#!/usr/bin/env python2.7
# encoding: utf8

import os
import sys
import datetime

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf

class PythonTypes(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')

    def test_not_convert_string_to_double(self):
        self.query(udf.fixindent('''
                CREATE EXTERNAL SCALAR SCRIPT
                wrong_type() RETURNS DOUBLE AS
                # redirector @@redirector_url@@
                def run(ctx):
                    return "one point five"
                '''))
        with self.assertRaisesRegexp(Exception, r'type float but data given have type'):
            self.query('''SELECT wrong_type() FROM DUAL''')

    def test_raises_with_incompatible_type(self):
        self.query(udf.fixindent('''
                CREATE EXTERNAL SCALAR SCRIPT
                wrong_type() RETURNS DOUBLE AS
                # redirector @@redirector_url@@
                def run(ctx):
                    return "one point five"
                '''))
        with self.assertRaisesRegexp(Exception, r'type float but data given have type'):
            self.query('''SELECT wrong_type() FROM DUAL''')



    def test_timestamps_work_with_timestamp_with_local_time_zone_type(self):
        self.query(udf.fixindent('''
                CREATE EXTERNAL SCALAR SCRIPT PY_ECHO(ts TIMESTAMP) RETURNS TIMESTAMP AS
                # redirector @@redirector_url@@
                def run(ctx):
                    return ctx.ts
                '''))
        self.query(udf.fixindent('''
                CREATE TABLE TTT(ts timestamp with local time zone);
                '''))
        self.query(udf.fixindent('''
                alter session set time_zone = '0';
                '''))
        self.query(udf.fixindent('''
                insert into TTT values timestamp '2013-04-22 13:55:50';
                '''))
        rows = self.query(udf.fixindent('''
                select PY_ECHO(ts) from TTT
                '''))
        expected = [(datetime.datetime(2013, 4, 22, 13, 55, 50),)]
        self.assertRowsEqual(expected, rows)
        self.query(udf.fixindent('''
                alter session set time_zone = '-2';
                '''))
        rows = self.query(udf.fixindent('''
                select PY_ECHO(ts) from TTT
                '''))
        expected = [(datetime.datetime(2013, 4, 22, 11, 55, 50),)]
        self.assertRowsEqual(expected, rows)        
        self.query(udf.fixindent('''
                create or replace script reset_time_zone() as
                     local success,dbtz = pquery([[select dbtimezone]]);
                     query([[ALTER SESSION SET TIME_ZONE = :tz]], {tz=dbtz[1][1]})
                '''))
        self.query(udf.fixindent('''
                   execute script reset_time_zone();
                '''))
        

if __name__ == '__main__':
    udf.main()
                
# vim: ts=4:sts=4:sw=4:et:fdm=indent
