#!/usr/bin/env python3

import locale
import os
import subprocess

from exasol_python_test_framework import udf


class _JavaUdfSetup(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN1')
        self.query('OPEN SCHEMA FN1')
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


class WordCount(_JavaUdfSetup):

    def setUp(self):
        super().setUp()
        self.query('OPEN SCHEMA FN1')

