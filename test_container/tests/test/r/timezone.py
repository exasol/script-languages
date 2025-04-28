#!/usr/bin/env python3

from exasol_python_test_framework import udf
from exasol_python_test_framework.udf import requires


class TimeZoneTest(udf.TestCase):

    def setUp(self):
        self.query('DROP SCHEMA tz_r CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA tz_r')

    def test_default(self):
        rows = self.query('''
            SELECT DBTIMEZONE 
            FROM DUAL
            ''')
        self.assertRowsEqual([("EUROPE/BERLIN",)], rows)
        self.query(udf.fixindent('''
        create r SCALAR SCRIPT
        default_tz()
        RETURNS VARCHAR(100) AS
        run <- function(ctx) {
            non_dst_date <- as.POSIXct("2023-01-01 12:00:00")
            timezone_short_name <- format(non_dst_date, "%Z")
            timezone_short_name
        }
        /
        '''))
        rows = self.query('''
            SELECT tz_r.default_tz() 
            FROM DUAL
            ''')
        self.assertRowsEqual([("CET",)], rows)


    def test_set_tz(self):
        self.query(udf.fixindent('''
        create r SCALAR SCRIPT
        modify_tz(tzname VARCHAR(100))
        RETURNS VARCHAR(100) AS
        run <- function(ctx) {
            Sys.setenv("TZ" = ctx$tzname)
            non_dst_date <- as.POSIXct("2023-01-01 12:00:00")
            timezone_short_name <- format(non_dst_date, "%Z")
            timezone_short_name
        }
        /
        '''))
        rows = self.query('''
            SELECT tz_r.modify_tz('America/New_York')
            FROM DUAL
            ''')
        self.assertRowsEqual([("EST",)], rows)

    def test_set_tz_via_script_options(self):
        self.query(udf.fixindent('''
        create r SCALAR SCRIPT
        modify_tz_via_script_option()
        RETURNS VARCHAR(100) AS
        %env TZ=America/New_York;
        run <- function(ctx) {
            non_dst_date <- as.POSIXct("2023-01-01 12:00:00")
            timezone_short_name <- format(non_dst_date, "%Z")
            timezone_short_name
        }
        /
        '''))
        rows = self.query('''
            SELECT tz_r.modify_tz_via_script_option()
            FROM DUAL
            ''')
        self.assertRowsEqual([("EST",)], rows)


if __name__ == '__main__':
    udf.main()
