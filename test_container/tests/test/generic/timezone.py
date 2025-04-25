
#!/usr/bin/env python3

from exasol_python_test_framework import udf
from exasol_python_test_framework.udf import requires


class BasicTest(udf.TestCase):

    @requires('DEFAULT_TZ')
    def test_default(self):

        rows = self.query('''
            SELECT DBTIMEZONE 
            FROM DUAL
            ''')
        self.assertRowsEqual([("EUROPE/BERLIN",)], rows)
        rows = self.query('''
            SELECT DEFAULT_TZ() 
            FROM DUAL
            ''')
        self.assertRowsEqual([("CET",)], rows)


    @requires('MODIFY_TZ_TO_NEW_YORK')
    def test_set_tz(self):
        rows = self.query('''
            SELECT MODIFY_TZ()
            FROM DUAL
            ''')
        self.assertRowsEqual([("EST",)], rows)


if __name__ == '__main__':
    udf.main()
