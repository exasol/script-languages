#!/usr/opt/bs-python-2.7/bin/python

import os
import sys
import datetime

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf

class MergeForward(udf.TestCase):
    def merge_test(self, merge_stmt, rows_affected, target_after):
        self.query(merge_stmt)
        self.assertEqual(rows_affected, self.rowcount())
        rows = self.query('SELECT id, a FROM target_t')
        self.assertRowsEqual(sorted(target_after), sorted(rows))
        self.query('rollback')
    
    def setUp(self):
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')
        self.query('create or replace table source_t(id int, x int)')
        self.query('create or replace table target_t(id int, a int)')
        self.query('insert into source_t values (1,100),(2,200),(3,300)')
        self.query('insert into target_t values (1,1),(2,2),(10,10)')
        self.commit()

    def test_delete(self):
        self.merge_test(
            '''merge into target_t t using source_t s on (t.id=s.id)
                     when matched then delete;''',
            2,
            [(10,10)] );
        
    def test_delete_filter_target(self):
        self.merge_test(
            '''merge into target_t t using source_t s on (t.id=s.id)
                     when matched then delete where a >= 2;''',
            1,
            [(1,1),(10,10)] );

    def test_delete_insert(self):
        self.merge_test(
            '''merge into target_t t using source_t s on (t.id=s.id)
                     when matched then delete
                     when not matched then insert values (x,x);''',
            3,
            [(300,300),(10,10)] );

    def test_update(self):
        self.merge_test(
            '''merge into target_t t using source_t s on (t.id=s.id)
                     when matched then update set a=x;''',
            2,
            [(1,100),(2,200),(10,10)] );

    def test_update_delete(self):
        self.merge_test(
            '''merge into target_t t using source_t s on (t.id=s.id)
                     when matched then update set a=x delete;''',
            2,
            [(10,10)] );

    def test_update_insert(self):
        self.merge_test(
            '''merge into target_t t using source_t s on (t.id=s.id)
                     when matched then update set a=x
                     when not matched then insert values (x,x);''',
            3,
            [(2,200),(300,300),(1,100),(10,10)] );

    def test_update_delete_insert(self):
        self.merge_test(
            '''merge into target_t t using source_t s on (t.id=s.id)
                     when matched then update set a=x delete where a > 10
                     when not matched then insert values (x,x);''',
            3,
            [(300,300),(10,10)] );

    def test_insert(self):
        self.merge_test(
            '''merge into target_t t using source_t s on (t.id=s.id)
                     when not matched then insert values (x,x);''',
            1,
            [(2,2),(1,1),(10,10),(300,300)] );

if __name__ == '__main__':
    udf.main()
