#!/usr/opt/bs-python-2.7/bin/python

import os
import sys
import datetime

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf

class NaNAndNOTNULL(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')
        self.query('''create view
	v as
select
	7 a
from
	dual''')

    def test_insert_nan_in_notnull_column(self):
        rows = self.query('''select replace(view_text, char(9), 'X') from exa_all_views
                                    where view_name='V' ''')
        self.assertEqual('create view\nXv as\nselect\nX7 a\nfrom\nXdual', rows[0][0])

if __name__ == '__main__':
    udf.main()
