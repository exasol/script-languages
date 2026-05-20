#!/usr/bin/env python3

import locale
import os
import subprocess

from exasol_python_test_framework import udf

class _Python3UdfSetup(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN1')
        self.query('OPEN SCHEMA FN1')

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


class WordCount(_Python3UdfSetup):

    def setUp(self):
        super().setUp()
        # FN1 is already opened by parent, but keeping this for explicitness
        self.query('OPEN SCHEMA FN1')
