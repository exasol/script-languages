#!/usr/bin/env python3

from exasol_python_test_framework import udf


class JavaInterpreter(udf.TestCase):
    def setUp(self):
        self.query('CREATE SCHEMA FN2', ignore_errors=True)
        self.query('OPEN SCHEMA FN2')

    def test_main_is_not_executed_at_creation_time(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                main_error()
                RETURNS int AS
                class MAIN_ERROR {
                    static void main(String[] args) {
                        throw new RuntimeException("Main Executed");
                    }

                    static int run(ExaMetadata exa, ExaIterator ctx) {
                        ctx.emit(42);
                    }
                }
                /
                '''))

    def test_syntax_errors_not_caught_at_creation_time(self):
        sql = udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                syntax_error()
                RETURNS int AS
                class SYNTAX_ERROR {
                    static int run(ExaMetadata exa, ExaIterator ctx) {
                        return 42
                    }
                }
                /
                ''')
        self.query(sql)

    def test_methods_have_access_to_class_members(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                other_mem()
                RETURNS int AS
                class OTHER_MEM {
                    static int foo = 2;
                    static int other() {
                        foo = 5;
                        return foo;
                    }
                    static int run(ExaMetadata exa, ExaIterator ctx) {
                        return other();
                    }
                }
                '''))
        row = self.query('SELECT other_mem() FROM DUAL')[0]
        self.assertEqual(5, row[0])

    def test_exception_in_init_is_propagated(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE java SCALAR SCRIPT
            foo()
            RETURNS INT AS
            class FOO {
                static int run(ExaMetadata exa, ExaIterator ctx) {
                    return 42;
                }
                static int init(ExaMetadata exa) throws Exception {
                    throw new Exception("4711");
                }
            }
            '''))
        with self.assertRaisesRegex(Exception, '4711'):
            self.query('SELECT foo() FROM dual')

    def test_init_has_global_context(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE java SCALAR SCRIPT
            foo()
            RETURNS INT AS
            class FOO {
                static int flag = 21;
                static int run(ExaMetadata exa, ExaIterator ctx) {
                    flag = 4711;
                    return 42;
                }
                static int init(ExaMetadata exa) throws Exception {
                    throw new Exception(Integer.toString(flag));
                }
            }
            '''))
        with self.assertRaisesRegex(Exception, '21'):
            self.query('SELECT foo() FROM dual')

    def test_exception_in_cleanup_is_propagated(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE java SCALAR SCRIPT
            foo()
            RETURNS INT AS
            class FOO {
                static int run(ExaMetadata exa, ExaIterator ctx) {
                    return 42;
                }
                static int cleanup(ExaMetadata exa) throws Exception {
                    throw new Exception("4711");
                }
            }
            '''))
        with self.assertRaisesRegex(Exception, '4711'):
            self.query('SELECT foo() FROM dual')
            
    def test_exception_in_run_and_cleanup_is_propagated(self):
        out, _err = self.query_via_exaplus(udf.fixindent('''
            DROP SCHEMA test_exception_in_run_and_cleanup_is_propagated CASCADE;
            CREATE SCHEMA test_exception_in_run_and_cleanup_is_propagated;
            OPEN SCHEMA test_exception_in_run_and_cleanup_is_propagated;
            CREATE OR REPLACE java SCALAR SCRIPT
            foo()
            RETURNS INT AS
            class FOO {
                static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    throw new Exception("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX");
                }
                static int cleanup(ExaMetadata exa) throws Exception {
                    throw new Exception("YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY");
                }
            }
            /
            SELECT foo() FROM dual;
            DROP SCHEMA test_exception_in_run_and_cleanup_is_propagated CASCADE;
            ROLLBACK;
            '''))
        print(out)
        print()
        print()
        print(_err)
        _err = _err.decode("utf-8")
        self.assertRegexpMatches(_err.replace("\n"," "), "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX.*YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY")

    def test_cleanup_has_global_context(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE java SCALAR SCRIPT
            foo()
            RETURNS INT AS
            class FOO {
                static int flag = 21;
                static int run(ExaMetadata exa, ExaIterator ctx) {
                    flag = 4711;
                    return 42;
                }
                static int cleanup(ExaMetadata exa) throws Exception {
                    throw new Exception(Integer.toString(flag));
                }
            }
            '''))
        with self.assertRaisesRegex(Exception, '4711'):
            rows = self.query('SELECT foo() FROM dual')
            self.assertRowsEqual([(42,)], rows)


class JavaJar(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')

    def test_jar_path(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                test_jar_path()
                RETURNS int AS
                %jar
                '''))
        with self.assertRaisesRegex(Exception, 'No values found for %jar statement'):
            rows = self.query('SELECT test_jar_path() FROM dual')

    def test_jar_path2(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                test_jar_path2()
                RETURNS int AS
                %jar
                class TEST_JAR_PATH2 {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        X.run(exa, ctx);
                        return 42;
                    }
                }
                '''))
        with self.assertRaisesRegex(Exception, 'End of %jar statement not found'):
            rows = self.query('SELECT test_jar_path2() FROM dual')

    def test_jar_path3(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                test_jar_path3()
                RETURNS int AS
                %jar ;
                class TEST_JAR_PATH3 {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        X.run(exa, ctx);
                        return 42;
                    }
                }
                '''))
        with self.assertRaisesRegex(Exception, 'No values found for %jar statement'):
            rows = self.query('SELECT test_jar_path3() FROM dual')

    def test_jar_path_end(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                test_jar_path_end()
                RETURNS int AS
                %jar /my/path/x.jar'''))
        with self.assertRaisesRegex(Exception, 'End of %jar statement not found'):
            rows = self.query('SELECT test_jar_path_end() FROM dual')

    def test_jar_tab(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                jar_case_tab()
                RETURNS int AS
                %jar		/my/path/x.jar;
                class JAR_CASE_TAB {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return 1;
                    }
                }
                '''))
        with self.assertRaisesRegex(Exception, 'No such file or directory'):
            rows = self.query('SELECT jar_case_tab() FROM DUAL')

    def test_jar_tab_end(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                jar_case_tab_end()
                RETURNS int AS
                %jar		/my/path/x.jar 		 ;
                class JAR_CASE_TAB_END {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return 1;
                    }
                }
                '''))
        with self.assertRaisesRegex(Exception, 'No such file or directory'):
            rows = self.query('SELECT jar_case_tab_end() FROM DUAL')

    def test_jar_multiple_statements(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                jar_case_multi_statements()
                RETURNS int AS
                %jar /my/path/x.jar;
                %jar /my/path/x2.jar;
                class JAR_CASE_MULTI_STATEMENTS {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return 1;
                    }
                }
                '''))
        with self.assertRaisesRegex(Exception, 'No such file or directory'):
            rows = self.query('SELECT jar_case_multi_statements() FROM DUAL')

    def test_jar_commented(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                jar_case_commented()
                RETURNS int AS
                // %jar /my/path/x.jar;
                class JAR_CASE_COMMENTED {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return 1;
                    }
                }
                '''))
        rows = self.query('SELECT jar_case_commented() FROM DUAL')
        self.assertEqual(1, rows[0][0])

    def test_jar_commented_after_code(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                jar_case_commented_after_code()
                RETURNS int AS
                class JAR_CASE_COMMENTED_AFTER_CODE {  // %jar /my/path/x.jar;
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return 1;
                    }
                }
                '''))
        rows = self.query('SELECT jar_case_commented_after_code() FROM DUAL')
        self.assertEqual(1, rows[0][0])

    def test_jar_after_code(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                jar_case_after_code()
                RETURNS int AS
                class JAR_CASE_AFTER_CODE { %jar /my/path/x.jar;
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return 1;
                    }
                }
                '''))
        with self.assertRaisesRegex(Exception, 'VM error:'):
            rows = self.query('SELECT jar_case_after_code() FROM DUAL')


    def test_jar_multiple_jars(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                jar_case_multi_jars()
                RETURNS int AS
                %jar /my/path/x.jar:/my/path/x2.jar;
                class JAR_CASE_MULTI_JARS {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return 1;
                    }
                }
                '''))
        with self.assertRaisesRegex(Exception, 'No such file or directory'):
            rows = self.query('SELECT jar_case_multi_jars() FROM DUAL')


class JavaSyntax(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')

    def test_sql_comments_are_not_ignored(self):
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT
            sql_comments()
            RETURNS INT AS
            class SQL_COMMENTS {
                static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return (
                        --3
                        +4);
                }
            }
            '''))
        with self.assertRaisesRegex(Exception, 'ExaCompilationException'):
            rows = self.query('SELECT sql_comments() FROM dual')

    def test_java_comments_are_ignored_in_functions(self):
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT
            java_comments()
            RETURNS INT AS
            class JAVA_COMMENTS {
                static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return (
                        //3
                        /*
                        +6
                        */
                        +4);
                }
            }
            '''))
        rows = self.query('SELECT java_comments() FROM dual')
        self.assertEqual(4, rows[0][0])

    def test_java_comments_are_ignored_in_body(self):
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT
            java_comments()
            RETURNS INT AS
            class JAVA_COMMENTS {
                // InT x <- 21;
                /* InT x <- 21; */
                static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return 43;
                }
            }
            '''))
        rows = self.query('SELECT java_comments() FROM dual')
        self.assertEqual(43, rows[0][0])


class JavaJvmOption(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')

    def test_jvm_opt(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                test_jvm_opt()
                RETURNS int AS
                %jvmoption
                '''))
        with self.assertRaisesRegex(Exception, 'No values found for %jvmoption statement'):
            rows = self.query('SELECT test_jvm_opt() FROM dual')

    def test_jvm_opt2(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                test_jvm_opt2()
                RETURNS int AS
                %jvmoption
                class TEST_JVM_OPT2 {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        X.run(exa, ctx);
                        return 42;
                    }
                }
                '''))
        with self.assertRaisesRegex(Exception, 'End of %jvmoption statement not found'):
            rows = self.query('SELECT test_jvm_opt2() FROM dual')

    def test_jvm_opt3(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                test_jvm_opt3()
                RETURNS int AS
                %jvmoption ;
                class TEST_JVM_OPT3 {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        X.run(exa, ctx);
                        return 42;
                    }
                }
                '''))
        with self.assertRaisesRegex(Exception, 'No values found for %jvmoption statement'):
            rows = self.query('SELECT test_jvm_opt3() FROM dual')

    def test_jvm_opt4(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                test_jvm_opt4()
                RETURNS int AS
                %jvmoption -Xmx512m'''))
        with self.assertRaisesRegex(Exception, 'End of %jvmoption statement not found'):
            rows = self.query('SELECT test_jvm_opt4() FROM dual')

    def test_jvm_opt_tab(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                test_jvm_opt_tab()
                RETURNS int AS
                %jvmoption		-Xmx512m;
                class TEST_JVM_OPT_TAB {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return 1;
                    }
                }
                '''))
        rows = self.query('SELECT test_jvm_opt_tab() FROM DUAL')
        self.assertEqual(1, rows[0][0])

    def test_jvm_opt_tab_end(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                test_jvm_opt_tab_end()
                RETURNS int AS
                %jvmoption		-Xmx512m 		 ;
                class TEST_JVM_OPT_TAB_END {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return 1;
                    }
                }
                '''))
        rows = self.query('SELECT test_jvm_opt_tab_end() FROM DUAL')
        self.assertEqual(1, rows[0][0])

    def test_jvm_opt_multiple_opts(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                test_jvm_opt_multiple_opts()
                RETURNS int AS
                %jvmoption -Xms56m -Xmx128m -Xss512k;
                class TEST_JVM_OPT_MULTIPLE_OPTS {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return 1;
                    }
                }
                '''))
        rows = self.query('SELECT test_jvm_opt_multiple_opts() FROM DUAL')
        self.assertEqual(1, rows[0][0])

    def test_jvm_opt_multiple_opts2(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                test_jvm_opt_multiple_opts2()
                RETURNS int AS
                %jvmoption -Xmx5000m;
                %jvmoption -Xms56m -Xmx128m -Xss1k;
                %jvmoption -Xss512k -Xms128m;
                class TEST_JVM_OPT_MULTIPLE_OPTS2 {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return 1;
                    }
                }
                '''))
        rows = self.query('SELECT test_jvm_opt_multiple_opts2() FROM DUAL')
        self.assertEqual(1, rows[0][0])

    def test_jvm_opt_invalid_opt(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                test_jvm_opt_invalid_opt()
                RETURNS int AS
                %jvmoption -Xmj56m;
                class TEST_JVM_OPT_INVALID_OPT {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return 1;
                    }
                }
                '''))
        with self.assertRaisesRegex(Exception, '.*Cannot start the JVM: unknown error.*'):
            rows = self.query('SELECT test_jvm_opt_invalid_opt() FROM dual')

    def test_jvm_opt_invalid_opt2(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                test_jvm_opt_invalid_opt2()
                RETURNS int AS
                %jvmoption -Xmx56m junk;
                class TEST_JVM_OPT_INVALID_OPT2 {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return 1;
                    }
                }
                '''))
        with self.assertRaisesRegex(Exception, '.*Cannot start the JVM: unknown error.*'):
            rows = self.query('SELECT test_jvm_opt_invalid_opt2() FROM dual')

    def test_jvm_opt_invalid_opt3(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                test_jvm_opt_invalid_opt3()
                RETURNS int AS
                %jvmoption -Xjunk;
                %jvmoption -Xmx56m;
                class TEST_JVM_OPT_INVALID_OPT3 {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return 1;
                    }
                }
                '''))
        with self.assertRaisesRegex(Exception, '.*Cannot start the JVM: unknown error.*'):
            rows = self.query('SELECT test_jvm_opt_invalid_opt3() FROM dual')

    def test_jvm_opt_invalid_opt4(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                test_jvm_opt_invalid_opt4()
                RETURNS int AS
                %jvmoption -Xms56q;
                class TEST_JVM_OPT_INVALID_OPT4 {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return 1;
                    }
                }
                '''))
        with self.assertRaisesRegex(Exception, 'invalid arguments'):
            rows = self.query('SELECT test_jvm_opt_invalid_opt4() FROM dual')

    def test_jvm_opt_invalid_mem(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                test_jvm_opt_invalid_mem()
                RETURNS int AS
                %jvmoption -Xmx900000000m;
                class TEST_JVM_OPT_INVALID_MEM {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return 1;
                    }
                }
                '''))
        with self.assertRaisesRegex(Exception, 'VM crashed'):
            rows = self.query('SELECT test_jvm_opt_invalid_mem() FROM dual')

    def test_jvm_opt_invalid_mem2(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                test_jvm_opt_invalid_mem2()
                RETURNS int AS
                %jvmoption -Xms1m -Xmx1m;
                class TEST_JVM_OPT_INVALID_MEM2 {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return 1;
                    }
                }
                '''))
        with self.assertRaisesRegex(Exception, 'VM error:'):
            rows = self.query('SELECT test_jvm_opt_invalid_mem2() FROM dual')

    def test_jvm_opt_invalid_stack_size(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                test_jvm_opt_invalid_stack_size()
                RETURNS int AS
                %jvmoption -Xss1k;
                class TEST_JVM_OPT_INVALID_STACK_SIZE {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return 1;
                    }
                }
                '''))
        with self.assertRaisesRegex(Exception, 'unknown error'):
            rows = self.query('SELECT test_jvm_opt_invalid_stack_size() FROM dual')


class JavaScriptClass(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')

    def test_set_script_class(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT A()
                RETURNS int AS
                %scriptclass com.exasol.B;
                class B {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return 1;
                    }
                }
                '''))
        self.assertRowsEqual([(1,)],
            self.query('''SELECT a()'''))

    def test_set_script_class_2(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT A()
                RETURNS int AS
                %scriptclass   com.exasol.B   ;
                class B {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return 1;
                    }
                }
                '''))
        self.assertRowsEqual([(1,)],
            self.query('''SELECT a()'''))

    def test_set_invalid_script_class(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT A()
                RETURNS int AS
                %scriptclass com.exasol.C;
                class B {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return 1;
                    }
                }
                '''))
        with self.assertRaisesRegex(Exception, 'The main script class .* cannot be found:'):
            self.query('''SELECT a()''')

    def test_set_invalid_script_class_2(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT A()
                RETURNS int AS
                // Looks correct, however the script B is in the com.exasol package implicitly.
                %scriptclass B;
                class B {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return 1;
                    }
                }
                '''))
        with self.assertRaisesRegex(Exception, 'The main script class .* cannot be found:'):
            self.query('''SELECT a()''')

    def test_invalid_script_class(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT A()
                RETURNS int AS
                class B {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return 1;
                    }
                }
                '''))
        with self.assertRaisesRegex(Exception, 'The main script class .* cannot be found:'):
            self.query('''SELECT a()''')


class JavaGenericEmit(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')
        
    def test_emit_object_array_multi_arg(self):
        self.query(udf.fixindent('''
                CREATE JAVA SET SCRIPT EMIT_OBJECT (a int) EMITS (a varchar(100), b int) AS
                class EMIT_OBJECT {
                  static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    Object[] ret = new Object[2];
                    ret[0] = "object-array-emit";
                    ret[1] = 12345;
                    ctx.emit(ret);
                  }
                }
                '''))
        rows = self.query('''
            select a || 'X' || B from (
            select EMIT_OBJECT(1));
            ''')
        self.assertRowEqual(('object-array-emitX12345',), rows[0])

    def test_emit_object_array_single_arg(self):
        self.query(udf.fixindent('''
                CREATE JAVA SET SCRIPT EMIT_OBJECT_AND_REGULAR (a int) EMITS (a varchar(100)) AS
                class EMIT_OBJECT_AND_REGULAR {
                  static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    Object[] ret = new Object[1];
                    ret[0] = "object-array-emit";
                    ctx.emit(ret);
                    ctx.emit("regular-emit");
                  }
                }
                '''))
        rows = self.query('''
            SELECT EMIT_OBJECT_AND_REGULAR(1);
            ''')
        self.assertRowEqual(('object-array-emit',), rows[0])
        self.assertRowEqual(('regular-emit',), rows[1])

    def test_emit_object_array_varemit(self):
        self.query(udf.fixindent('''
                CREATE JAVA SET SCRIPT VAREMIT_GENERIC_EMIT (a varchar(100)) EMITS (...) AS
                class VAREMIT_GENERIC_EMIT {
                  static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    int cols = (int)exa.getOutputColumnCount();
                    Object[] ret = new Object[cols];
                    for (int i=0; i<cols; i++) {
                      ret[i] = ctx.getString(0);
                    }
                    ctx.emit(ret);
                  }
                }
                '''))
        rows = self.query('''
            SELECT "A" || 'x' || "B" FROM (
            SELECT VAREMIT_GENERIC_EMIT('SUPERDYNAMIC') EMITS (a varchar(100), b varchar(100)));
            ''')
        self.assertRowEqual(('SUPERDYNAMICxSUPERDYNAMIC',), rows[0])

if __name__ == '__main__':
    udf.main()

