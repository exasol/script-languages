#!/usr/bin/env python3

from exasol_python_test_framework import udf
from exasol_python_test_framework import exatest


class _Python3UdfSetup(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN1')
        self.query('OPEN SCHEMA FN1')
        
        self.query(
            '''CREATE OR REPLACE CONNECTION test_get_connection_bug_connection TO '' USER 'ialjksdhfalskdjhflaskdjfhalskdjhflaksjdhflaksdjfhalksjdfhlaksjdhflaksjdhfalskjdfhalskdjhflaksjdhflaksjdfhlaksjsadajksdhfaksjdfhalksdjfhalksdjfhalksjdfhqwiueryqw;er;lkjqwe;rdhflaksjdfhlaksdjfhaabcdefghijklmnopqrstuvwxyz' IDENTIFIED BY 'abcdeoqsdfgsdjfglksjdfhglskjdfhglskdjfglskjdfghuietyewlrkjthertrewerlkjhqwelrkjhqwerlkjnwqerlkjhqwerkjlhqwerlkjhqwerlkhqwerkljhqwerlkjhqwerfghijklmnopqrstuvwxyz';''')

class GetConnectionMemoryBug(_Python3UdfSetup):
    pass

