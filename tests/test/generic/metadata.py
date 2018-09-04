#!/usr/bin/env python2.7

import os
import sys

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf
from udf import requires

class MetaDataTest(udf.TestCase):

   
    @requires('GET_DATABASE_NAME')
    def test_database_name(self):
        rows = self.query('''SELECT fn1.get_database_name() FROM DUAL''')
        self.assertTrue(len(rows[0][0]) > 0)

    @requires('GET_DATABASE_VERSION')
    def test_database_version(self):
        rows = self.query('''select fn1.get_database_version() from dual''')
        self.assertTrue(len(rows[0][0]) > 0)

    @requires('GET_SCRIPT_LANGUAGE')
    def test_script_language(self):
        rows = self.query('''select fn1.get_script_language() from dual''')
        self.assertTrue((rows[0][0]).upper().startswith((rows[0][1]).upper()))

    @requires('GET_SCRIPT_NAME')
    def test_script_name(self):
        rows = self.query('''select fn1.get_script_name() from dual''')
        self.assertRowEqual(('GET_SCRIPT_NAME',), rows[0])

    @requires('GET_SCRIPT_SCHEMA')
    def test_script_schema(self):
        if (udf.opts.is_compat_mode != "true"):
            rows = self.query('''select fn1.get_script_schema() from dual''')
            self.assertRowEqual(('FN1',), rows[0])

    @requires('GET_CURRENT_USER')
    def test_script_user(self):
        if (udf.opts.is_compat_mode != "true"):
            rows = self.query('''select fn1.get_current_user() from dual''')
            self.assertRowEqual(('SYS',), rows[0])

    @requires('GET_SCOPE_USER')
    def test_scope_user(self):
        if (udf.opts.is_compat_mode != "true"):
            rows = self.query('''select fn1.get_scope_user() from dual''')
            self.assertRowEqual(('SYS',), rows[0])

    @requires('GET_CURRENT_SCHEMA')
    def test_current_schema_null(self):
        if (udf.opts.is_compat_mode != "true"):
            rows = self.query('''select fn1.get_current_schema() from dual''')
            self.assertRowEqual(('NULL',), rows[0])

    @requires('GET_CURRENT_SCHEMA')
    def test_current_schema(self):
        if (udf.opts.is_compat_mode != "true"):
            self.query('''create schema test_schema''')
            rows = self.query('''select fn1.get_current_schema() from dual''')
            self.assertRowEqual(('TEST_SCHEMA',), rows[0])
            self.query('''drop schema test_schema cascade''')

    @requires('GET_SCRIPT_CODE')
    def test_script_code(self):
        rows = self.query('''select fn1.get_script_code() from dual''')
        self.assertTrue((rows[0][0]).upper().find('CTX') >= 0)

    @requires('GET_SESSION_ID')
    def test_session_id(self):
        rows = self.query('''select fn1.get_session_id() from dual''')
        self.assertTrue(len(rows[0][0]) > 0)

    @requires('GET_STATEMENT_ID')
    def test_statement_id(self):
        rows = self.query('''select fn1.get_statement_id() from dual''')
        self.assertTrue(rows[0][0] >= 0)

    @requires('GET_NODE_ID')
    def test_node_id(self):
        rows = self.query('''select fn1.get_node_id() from dual''')
        self.assertTrue(rows[0][0] >= 0)

    @requires('GET_VM_ID')
    def test_vm_id(self):
        rows = self.query('''select fn1.get_vm_id() from dual''')
        self.assertTrue(len(rows[0][0]) > 0)

    @requires('GET_INPUT_TYPE_SCALAR')
    def test_input_type_scalar(self):
        rows = self.query('''select fn1.get_input_type_scalar() from dual''')
        self.assertRowEqual(('SCALAR',), rows[0])

    @requires('GET_INPUT_TYPE_SET')
    def test_input_type_set(self):
        rows = self.query('''select fn1.get_input_type_set(x) from (values 1,2,3) as t(x)''')
        self.assertRowEqual(('SET',), rows[0])

    @requires('GET_INPUT_COLUMN_COUNT_SCALAR')
    def test_input_column_count_scalar(self):
        rows = self.query('''select fn1.get_input_column_count_scalar(12.3, 'hihihi') from dual''')
        self.assertRowEqual((2,), rows[0])

    
    @requires('GET_INPUT_COLUMN_COUNT_SET')
    def test_input_column_count_set(self):
        rows = self.query('''select fn1.get_input_column_count_set(x, y) from (values (12.3, 'hihihi')) as t(x,y)''')
        self.assertRowEqual((2,), rows[0])

    @requires('GET_INPUT_COLUMNS')
    def test_input_columns(self):
        rows = self.query('''select fn1.get_input_columns(1.2, '123') from dual order by column_id''')
        r0 = rows[0]
        r1 = rows[1]
        self.assertTrue(r0[0] == 1)
        self.assertTrue(r0[1].upper() == 'C1')
        self.assertTrue(r0[2].upper() == 'NUMBER' or r0[2].upper() == "<TYPE 'FLOAT'>" or r0[2].upper() == "DOUBLE" or r0[2].upper() == "JAVA.LANG.DOUBLE" or r0[2].upper() == "<CLASS 'FLOAT'>")
        self.assertTrue(r0[3].upper() == 'DOUBLE')
        self.assertTrue(r1[0] == 2)
        self.assertTrue(r1[1].upper() == 'C2')
        self.assertTrue(r1[2].upper() == 'STRING' or r1[2].upper() == "<TYPE 'UNICODE'>" or r1[2].upper() == "CHARACTER" or r1[2].upper() == "JAVA.LANG.STRING" or r1[2].upper() == "<CLASS 'STR'>")
        self.assertTrue(r1[3].upper().startswith('VARCHAR(200)'))
        self.assertTrue(r0[6] == 0)
        self.assertTrue(r1[6] == 200) 
   
    @requires('GET_OUTPUT_TYPE_RETURN')
    def test_output_type_return(self):
        rows = self.query('''select fn1.get_output_type_return() from dual''')
        self.assertTrue(rows[0][0] == 'RETURN')

     
    @requires('GET_OUTPUT_TYPE_EMIT')
    def test_output_type_emit(self):
        rows = self.query('''select fn1.get_output_type_emit() from dual''')
        self.assertTrue(rows[0][0] == 'EMIT')


    @requires('GET_OUTPUT_COLUMN_COUNT_RETURN')
    def test_output_column_count_return(self):
        rows = self.query('''select fn1.get_output_column_count_return() from dual''')
        self.assertRowEqual((1,),rows[0])


    @requires('GET_OUTPUT_COLUMN_COUNT_EMIT')
    def test_output_column_count_emit(self):
        rows = self.query('''select fn1.get_output_column_count_emit() from dual''')
        self.assertRowEqual((3,3,3),rows[0])

    @requires('GET_OUTPUT_COLUMNS')
    def test_output_columns(self):
        rows = self.query('''select fn1.get_output_columns() from dual order by column_id''')
        r0 = rows[0]
        r1 = rows[1]
        r2 = rows[2]
        r3 = rows[3]
        r4 = rows[4]
        r5 = rows[5]
        r6 = rows[6]
        self.assertTrue(r0[0] == 1)
        self.assertTrue(r1[0] == 2)
        self.assertTrue(r2[0] == 3)
        self.assertTrue(r3[0] == 4)
        self.assertTrue(r4[0] == 5)
        self.assertTrue(r5[0] == 6)
        self.assertTrue(r6[0] == 7)
        self.assertTrue(r0[1].upper() == 'COLUMN_ID')
        self.assertTrue(r1[1].upper() == 'COLUMN_NAME')
        self.assertTrue(r2[1].upper() == 'COLUMN_TYPE')
        self.assertTrue(r3[1].upper() == 'COLUMN_SQL_TYPE')
        self.assertTrue(r4[1].upper() == 'COLUMN_PRECISION')
        self.assertTrue(r5[1].upper() == 'COLUMN_SCALE')
        self.assertTrue(r6[1].upper() == 'COLUMN_LENGTH')
        self.assertTrue(r0[2].upper() == 'NUMBER' or r0[2].upper() == "<TYPE 'FLOAT'>" or r0[2].upper() == "DOUBLE" or r0[2].upper() == "JAVA.LANG.DOUBLE" or r0[2].upper() == "<CLASS 'FLOAT'>")
        self.assertTrue(r1[2].upper() == 'STRING' or r1[2].upper() == "<TYPE 'UNICODE'>" or r1[2].upper() == "CHARACTER" or r1[2].upper() == "JAVA.LANG.STRING" or r1[2].upper() == "<CLASS 'STR'>")
        self.assertTrue(r2[2].upper() == 'STRING' or r2[2].upper() == "<TYPE 'UNICODE'>" or r2[2].upper() == "CHARACTER" or r2[2].upper() == "JAVA.LANG.STRING" or r2[2].upper() == "<CLASS 'STR'>")
        self.assertTrue(r3[2].upper() == 'STRING' or r3[2].upper() == "<TYPE 'UNICODE'>" or r3[2].upper() == "CHARACTER" or r3[2].upper() == "JAVA.LANG.STRING" or r3[2].upper() == "<CLASS 'STR'>")
        self.assertTrue(r4[2].upper() == 'NUMBER' or r4[2].upper() == "<TYPE 'FLOAT'>" or r4[2].upper() == "DOUBLE" or r4[2].upper() == "JAVA.LANG.DOUBLE" or r4[2].upper() == "<CLASS 'FLOAT'>")
        self.assertTrue(r5[2].upper() == 'NUMBER' or r5[2].upper() == "<TYPE 'FLOAT'>" or r5[2].upper() == "DOUBLE" or r5[2].upper() == "JAVA.LANG.DOUBLE" or r5[2].upper() == "<CLASS 'FLOAT'>")
        self.assertTrue(r6[2].upper() == 'NUMBER' or r6[2].upper() == "<TYPE 'FLOAT'>" or r6[2].upper() == "DOUBLE" or r6[2].upper() == "JAVA.LANG.DOUBLE" or r6[2].upper() == "<CLASS 'FLOAT'>")
        self.assertTrue(r0[3].upper() == 'DOUBLE')
        self.assertTrue(r1[3].upper().startswith('VARCHAR(200)'))
        self.assertTrue(r2[3].upper().startswith('VARCHAR(20)'))
        self.assertTrue(r3[3].upper().startswith('VARCHAR(20)'))
        self.assertTrue(r4[3].upper() == 'DOUBLE')
        self.assertTrue(r5[3].upper() == 'DOUBLE')
        self.assertTrue(r6[3].upper() == 'DOUBLE')
        self.assertTrue(r1[6] == 200) 
        self.assertTrue(r2[6] == 20) 
        self.assertTrue(r3[6] == 20) 



    @requires('GET_PRECISION_SCALE_LENGTH')
    def test_precision_scale_length(self):
        rows = self.query('''select fn1.get_precision_scale_length(2.5, '0123456789') from dual''')
        self.assertRowEqual((6,3,0,0,0,10), rows[0])


    @requires('GET_CHAR_LENGTH')
    def test_char_length(self):
        rows = self.query('''select fn1.get_char_length('0123456789') from dual''')
        self.assertRowEqual((10,20,'9876543210          '), (int(rows[0][0]), int(rows[0][1]), rows[0][2]))

if __name__ == '__main__':
    udf.main()

# vim: ts=4:sts=4:sw=4:et:fdm=indent
