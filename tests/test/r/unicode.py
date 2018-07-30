#!/usr/bin/env python2.7

import locale
import os
import sys

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

class RUnicode(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')

if __name__ == '__main__':
    udf.main()

# vim: ts=4:sts=4:sw=4:et:fdm=indent

