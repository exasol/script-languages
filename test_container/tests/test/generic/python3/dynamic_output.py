#!/usr/bin/env python3

from exasol_python_test_framework import udf
from exasol_python_test_framework import exatest


def setUpModule():
    """Set default language for manual testing without --lang parameter."""
    if udf.opts and udf.opts.lang is None:
        udf.opts.lang = 'python3'


class _Python3UdfSetup(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN1')
        self.query('OPEN SCHEMA FN1')
        
        self.query('DROP SCHEMA dynamic_output CASCADE', ignore_errors=True)
        self.query('''drop connection SPOT4245''', ignore_errors=True)
        self.query('CREATE SCHEMA dynamic_output')
        self.query('CREATE TABLE dynamic_output.small(x VARCHAR(2000), y DOUBLE)')
        self.query('''INSERT INTO dynamic_output.small VALUES ('Some string ... and some more', 2.2)''')
        self.query('create table dynamic_output.groupt(id int, n double, v varchar(999))')
        self.query('''insert into dynamic_output.groupt values (1,1,'aa'),
                                                (1,2,'ab'),
                                                (2,2,'ba')
                                                ''')
        self.query('create table dynamic_output.target (a int, b double, c varchar(100));')
        self.query('''create connection SPOT4245 to 'a' user 'b' identified by 'c' ''')

class Test(_Python3UdfSetup):
    pass



class DynamicOutputCreateScript(Test):
    pass
