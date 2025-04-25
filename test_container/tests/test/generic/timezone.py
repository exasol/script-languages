
#!/usr/bin/env python3

from exasol_python_test_framework import udf
from exasol_python_test_framework.udf import requires


class TimeZoneTest(udf.TestCase):

    @requires('DEFAULT_TZ')
    def test_default(self):

        rows = self.query('''
            SELECT DBTIMEZONE 
            FROM DUAL
            ''')
        self.assertRowsEqual([("EUROPE/BERLIN",)], rows)
        rows = self.query('''
            SELECT fn1.default_tz() 
            FROM DUAL
            ''')
        self.assertRowsEqual([("CET",)], rows)


    @requires('MODIFY_TZ_TO_NEW_YORK')
    def test_set_tz(self):
        rows = self.query('''
            SELECT fn1.modify_tz_to_new_york()
            FROM DUAL
            ''')
        self.assertRowsEqual([("EST",)], rows)


if __name__ == '__main__':
    udf.main()
