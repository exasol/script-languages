#!/usr/bin/env python3

from exasol_python_test_framework import udf


class MetaDataTest(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN1')
        self.query('OPEN SCHEMA FN1')
        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT get_char_length(text char(10))
            EMITS (len1 number, len2 number, dummy char(20)) AS
            def run(ctx):
            	v = exa.meta.input_columns[0]
            	w = exa.meta.output_columns[2]
            	ctx.emit(v.length,w.length,'9876543210')
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT get_current_schema()
            RETURNS varchar(200) AS
            def run(ctx):
                    return exa.meta.current_schema
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT get_current_user()
            RETURNS varchar(200) AS
            def run(ctx):
                    return exa.meta.current_user
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT get_database_name()
            RETURNS varchar(300) AS
            def run(ctx):
            	return exa.meta.database_name
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT get_database_version()
            RETURNS varchar(20) AS
            def run(ctx):
            	return exa.meta.database_version
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT get_input_column_count_scalar(c1 double, c2 varchar(100))
            RETURNS number AS
            def run(ctx):
             	return exa.meta.input_column_count
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SET SCRIPT get_input_column_count_set(c1 double, c2 varchar(100))
            RETURNS number AS
            def  run(ctx):
                 return exa.meta.input_column_count
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT get_input_columns(c1 double, c2 varchar(200))
            EMITS (column_id number, column_name varchar(200), column_type varchar(20),
             	   column_sql_type varchar(20), column_precision number, column_scale number,
              	   column_length number) AS
            def run(ctx):
                    cols = exa.meta.input_columns
                    for i in range(0, len(cols)):
                            name  = cols[i].name
                            precision = cols[i].precision
                            thetype = repr(cols[i].type)
                            sql_type = cols[i].sql_type
                            scale = cols[i].scale
                            length = cols[i].length
                            if name == None: name = 'no-name'
                            if thetype == None: thetype = 'no-type'
                            if sql_type == None: sql_type = 'no-sql-type'
                            if precision == None: precision = 0
                            if scale == None: scale = 0
                            if length == None: length = 0
                            ctx.emit(i+1, name, thetype, sql_type, precision, scale, length)
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT get_input_type_scalar()
            RETURNS varchar(200) AS
            def run(ctx):
             	return exa.meta.input_type
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SET SCRIPT get_input_type_set(a double)
            RETURNS varchar(200) AS
            def run(ctx):
             	return exa.meta.input_type
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT get_node_id()
            RETURNS number AS
            def run(ctx):
            	return exa.meta.node_id
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT get_output_column_count_emit()
            EMITS (x number, y number, z number) AS
            def run(ctx):
            	ctx.emit(exa.meta.output_column_count,exa.meta.output_column_count,exa.meta.output_column_count)
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT get_output_column_count_return()
            RETURNS number AS
            def run(ctx):
             	return exa.meta.output_column_count
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT get_output_columns()
            EMITS (column_id number, column_name varchar(200), column_type varchar(20),
             	   column_sql_type varchar(20), column_precision number, column_scale number,
              	   column_length number) AS
            def run(ctx):
            	cols = exa.meta.output_columns
            	for i in range(0, len(cols)):
            		name = cols[i].name
            		thetype = repr(cols[i].type)
            		sql_type = cols[i].sql_type
            		precision = cols[i].precision
            		scale = cols[i].scale
            		length = cols[i].length
            		if name == None: name = 'no-name'
            		if thetype == None: thetype = 'no-type'
            		if sql_type == None: sql_type = 'no-sql-type'
            		if precision == None: precision = 0
            		if scale == None: scale = 0
            		if length == None: length = 0
            		ctx.emit(i+1, name, thetype, sql_type, precision, scale, length)
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT get_output_type_emit()
            EMITS (t varchar(200)) AS
            def run(ctx):
               ctx.emit(exa.meta.output_type)
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT get_output_type_return()
            RETURNS varchar(200) AS
            def run(ctx):
             	return exa.meta.output_type
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT get_precision_scale_length(n decimal(6,3), v varchar(10))
            EMITS (precision1 number, scale1 number, length1 number, precision2 number, scale2 number, length2 number) AS
            def run(ctx):
                    v = exa.meta.input_columns[0]
                    precision1 = v.precision
                    scale1 = v.scale
                    length1 = v.length
                    w = exa.meta.input_columns[1]
                    precision2 = w.precision
                    scale2 = w.scale
                    length2 = w.length
                    if precision1== None: precision1= 0 
                    if scale1== None: scale1= 0
                    if length1== None: length1= 0
                    if precision2== None: precision2= 0 
                    if scale2== None: scale2= 0
                    if length2== None: length2= 0
                    ctx.emit(precision1, scale1, length1, precision2, scale2, length2)
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT get_scope_user()
            RETURNS varchar(200) AS
            def run(ctx):
                    return exa.meta.scope_user
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT get_script_code()
            RETURNS varchar(2000) AS
            def run(ctx):
            	return exa.meta.script_code
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT get_script_language()
            EMITS (s1 varchar(300), s2 varchar(300)) AS
            def run(ctx):
            	ctx.emit(exa.meta.script_language, "Python")
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT get_script_name()
            RETURNS varchar(200) AS
            def run(ctx):
            	return exa.meta.script_name
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT get_script_schema()
            RETURNS varchar(200) AS
            def run(ctx):
                    return exa.meta.script_schema
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT get_session_id()
            RETURNS varchar(200) AS
            def run(ctx):
            	return exa.meta.session_id
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT get_statement_id()
            RETURNS number AS
            def run(ctx):
            	return exa.meta.statement_id
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT get_vm_id()
            RETURNS varchar(200) AS
            def run(ctx):
             	return exa.meta.vm_id
            /
        '''))
        
        # Close schema so test_current_schema_null passes (expects NULL)
        self.query('CLOSE SCHEMA')

    def test_database_name(self):
        rows = self.query('''SELECT fn1.get_database_name() FROM DUAL''')
        self.assertTrue(len(rows[0][0]) > 0)

    def test_database_version(self):
        rows = self.query('''select fn1.get_database_version() from dual''')
        self.assertTrue(len(rows[0][0]) > 0)

    def test_script_language(self):
        rows = self.query('''select fn1.get_script_language() from dual''')
        self.assertTrue((rows[0][0]).upper().startswith((rows[0][1]).upper()))

    def test_script_name(self):
        rows = self.query('''select fn1.get_script_name() from dual''')
        self.assertRowEqual(('GET_SCRIPT_NAME',), rows[0])

    def test_script_schema(self):
        rows = self.query('''select fn1.get_script_schema() from dual''')
        self.assertRowEqual(('FN1',), rows[0])

    def test_script_user(self):
        rows = self.query('''select fn1.get_current_user() from dual''')
        self.assertRowEqual(('SYS',), rows[0])

    def test_scope_user(self):
        rows = self.query('''select fn1.get_scope_user() from dual''')
        self.assertRowEqual(('SYS',), rows[0])

    def test_current_schema_null(self):
        rows = self.query('''select fn1.get_current_schema() from dual''')
        self.assertRowEqual(('NULL',), rows[0])

    def test_current_schema(self):
        self.query('''create schema test_schema''')
        rows = self.query('''select fn1.get_current_schema() from dual''')
        self.assertRowEqual(('TEST_SCHEMA',), rows[0])
        self.query('''drop schema test_schema cascade''')

    def test_script_code(self):
        rows = self.query('''select fn1.get_script_code() from dual''')
        self.assertTrue((rows[0][0]).upper().find('CTX') >= 0)

    def test_session_id(self):
        rows = self.query('''select fn1.get_session_id() from dual''')
        self.assertTrue(len(rows[0][0]) > 0)

    def test_statement_id(self):
        rows = self.query('''select fn1.get_statement_id() from dual''')
        self.assertTrue(rows[0][0] >= 0)

    def test_node_id(self):
        rows = self.query('''select fn1.get_node_id() from dual''')
        self.assertTrue(rows[0][0] >= 0)

    def test_vm_id(self):
        rows = self.query('''select fn1.get_vm_id() from dual''')
        self.assertTrue(len(rows[0][0]) > 0)

    def test_input_type_scalar(self):
        rows = self.query('''select fn1.get_input_type_scalar() from dual''')
        self.assertRowEqual(('SCALAR',), rows[0])

    def test_input_type_set(self):
        rows = self.query('''select fn1.get_input_type_set(x) from (values 1,2,3) as t(x)''')
        self.assertRowEqual(('SET',), rows[0])

    def test_input_column_count_scalar(self):
        rows = self.query('''select fn1.get_input_column_count_scalar(12.3, 'hihihi') from dual''')
        self.assertRowEqual((2,), rows[0])

    
    def test_input_column_count_set(self):
        rows = self.query('''select fn1.get_input_column_count_set(x, y) from (values (12.3, 'hihihi')) as t(x,y)''')
        self.assertRowEqual((2,), rows[0])

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
   
    def test_output_type_return(self):
        rows = self.query('''select fn1.get_output_type_return() from dual''')
        self.assertTrue(rows[0][0] == 'RETURN')

     
    def test_output_type_emit(self):
        rows = self.query('''select fn1.get_output_type_emit() from dual''')
        self.assertTrue(rows[0][0] == 'EMIT')


    def test_output_column_count_return(self):
        rows = self.query('''select fn1.get_output_column_count_return() from dual''')
        self.assertRowEqual((1,),rows[0])


    def test_output_column_count_emit(self):
        rows = self.query('''select fn1.get_output_column_count_emit() from dual''')
        self.assertRowEqual((3,3,3),rows[0])

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



    def test_precision_scale_length(self):
        rows = self.query('''select fn1.get_precision_scale_length(2.5, '0123456789') from dual''')
        self.assertRowEqual((6,3,0,0,0,10), rows[0])


    def test_char_length(self):
        rows = self.query('''select fn1.get_char_length('0123456789') from dual''')
        self.assertRowEqual((10,20,'9876543210          '), (int(rows[0][0]), int(rows[0][1]), rows[0][2]))


if __name__ == '__main__':
    udf.main()
