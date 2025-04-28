# !/usr/bin/env python3

from exasol_python_test_framework import udf
from exasol_python_test_framework.udf import requires


class TimeZoneTest(udf.TestCase):

    def setUp(self):
        self.query('DROP SCHEMA tz_java CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA tz_java')

    def test_default(self):
        rows = self.query('''
                          SELECT DBTIMEZONE
                          FROM DUAL
                          ''')
        self.assertRowsEqual([("EUROPE/BERLIN",)], rows)
        self.query(udf.fixindent('''
        CREATE OR REPLACE java SCALAR SCRIPT
        default_tz()
        RETURNS VARCHAR(100) AS
        import java.time.ZoneId;
        import java.util.TimeZone;

        class DEFAULT_TZ{
            static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                TimeZone timeZone = TimeZone.getDefault();
                return timeZone.getDisplayName(false, TimeZone.SHORT);

            }
        }
        /
        '''))
        rows = self.query('''
                          SELECT tz_java.default_tz()
                          FROM DUAL
                          ''')
        self.assertRowsEqual([("CET",)], rows)

    def test_set_tz(self):
        self.query(udf.fixindent('''

        CREATE OR REPLACE java SCALAR SCRIPT
        modify_tz_to_est()
        RETURNS VARCHAR(100) AS
        %env TZ=America/New_York;
        import java.time.ZoneId;
        import java.util.TimeZone;
        class MODIFY_TZ_TO_EST{
            static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                TimeZone timeZone = TimeZone.getDefault();
                return timeZone.getDisplayName(false, TimeZone.SHORT);
            }
        }
        /
        '''))
        rows = self.query('''
                          SELECT tz_java.modify_tz_to_est()
                          FROM DUAL
                          ''')
        self.assertRowsEqual([("EST",)], rows)


if __name__ == '__main__':
    udf.main()
