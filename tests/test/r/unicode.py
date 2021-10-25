#!/usr/bin/env python3

import locale

from exasol_python_test_framework import udf

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


class RUnicode(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')


if __name__ == '__main__':
    udf.main()
