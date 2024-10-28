#!/usr/bin/env python3

from exasol_python_test_framework import udf
from exasol_python_test_framework.exatest import useData


class ScriptImport(udf.TestCase):

    legacy_env_declaration = ""
    ctpg_parser_env_declaration = "%env SCRIPT_OPTIONS_PARSER_VERSION=2;"
    additional_env_declarations = [(legacy_env_declaration,), (ctpg_parser_env_declaration,)]

    def setUp(self):
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('DROP SCHEMA FN3 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')
        self.query('CREATE SCHEMA FN3')
        self.query('OPEN SCHEMA FN2')

        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT
            bottom()
            RETURNS INT AS
            class BOTTOM {
                static int f() {
                        return 42;
                    }
                }
            /
            '''))

    def test_ImportError_is_catchable(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                catch_import_exception()
                RETURNS int AS
                class CATCH_IMPORT_EXCEPTION {
                   static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        try {
                            exa.importScript("x");
                        }
                        catch (ExaCompilationException ex) { }
                        return 42;
                    }
                }
                '''))
        rows = self.query('SELECT catch_import_exception() FROM dual')
        self.assertRowsEqual([(42,)], rows)

    @useData(additional_env_declarations)
    def test_preprocessed_ImportError_is_not_catchable(self, additional_env_declaration):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                cannot_catch_import_exception()
                RETURNS int AS
                ''' + additional_env_declaration + '''
                %import X;
                class CANNOT_CATCH_IMPORT_EXCEPTION {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        X.run(exa, ctx);
                        return 42;
                    }
                }
                '''))
        with self.assertRaisesRegex(Exception, 'script X not found'):
            rows = self.query('SELECT cannot_catch_import_exception() FROM dual')

    @useData(((legacy_env_declaration, 'No values found for %import statement'),
              (ctpg_parser_env_declaration,
               "Error parsing script options at line 1: \[1:8\] PARSE: Syntax error: Unexpected \'<eof>\'")))
    def test_preprocessed_Import_missing_script_name(self, additional_env_declaration, expected_error):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                missing_import_script_exception()
                RETURNS int AS
                ''' + additional_env_declaration + '''
                %import
                '''))
        with self.assertRaisesRegex(Exception, expected_error):
            rows = self.query('SELECT missing_import_script_exception() FROM dual')

    @useData(((legacy_env_declaration, 'End of %import statement not found'),
              (ctpg_parser_env_declaration,
               "Error parsing script options at line 1: \[1:8\] PARSE: Syntax error: Unexpected \'<eof>\'")))
    def test_preprocessed_Import_missing_script_name2(self, additional_env_declaration, expected_error):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                missing_import_script_exception2()
                RETURNS int AS
                ''' + additional_env_declaration + '''
                %import
                class CANNOT_CATCH_IMPORT_EXCEPTION {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        X.run(exa, ctx);
                        return 42;
                    }
                }
                '''))
        with self.assertRaisesRegex(Exception, expected_error):
            rows = self.query('SELECT missing_import_script_exception2() FROM dual')

    @useData(((legacy_env_declaration, 'No values found for %import statement'),
              (ctpg_parser_env_declaration,
               "Error parsing script options at line 1: \[1:9\] PARSE: Syntax error: Unexpected \';\'")))
    def test_preprocessed_Import_missing_script_name3(self, additional_env_declaration, expected_error):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                missing_import_script_exception3()
                RETURNS int AS
                ''' + additional_env_declaration + '''
                %import ;
                class CANNOT_CATCH_IMPORT_EXCEPTION {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        X.run(exa, ctx);
                        return 42;
                    }
                }
                '''))
        with self.assertRaisesRegex(Exception, expected_error):
            rows = self.query('SELECT missing_import_script_exception3() FROM dual')

    @useData(((legacy_env_declaration, 'End of %import statement not found'),
              (ctpg_parser_env_declaration,
               "Error parsing script options at line 1: \[1:10\] PARSE: Syntax error: Unexpected \'<eof>\'")))
    def test_preprocessed_Import_missing_import_end(self, additional_env_declaration, expected_error):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                missing_import_end_exception()
                RETURNS int AS
                ''' + additional_env_declaration + '''
                %import X'''))
        with self.assertRaisesRegex(Exception, expected_error):
            rows = self.query('SELECT missing_import_end_exception() FROM dual')

    @useData(additional_env_declarations)
    def test_import_is_case_sensitive(self, additional_env_declaration):
        scripts = [
            ('my_module', additional_env_declaration, 'my_module', 4711),
            ('My_Module', additional_env_declaration, 'My_Module', 42),
            ('MY_MODULE', additional_env_declaration, 'MY_MODULE', 1234),
        ]
        sql = '''
                CREATE OR REPLACE java SCALAR SCRIPT
                "%s"()
                RETURNS int AS
                %s
                class %s {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return %d;
                    }
                }
                '''
        for triple in scripts:
            self.query(udf.fixindent(sql % triple))

        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                import_case_sensitive()
                RETURNS int AS
                import java.lang.reflect.Method;
                class IMPORT_CASE_SENSITIVE {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        Class<?> scriptClass = exa.importScript("\\"My_Module\\"");
                        Class[] runParams = {exa.getClass().getInterfaces()[0], ctx.getClass().getInterfaces()[0]};
                        Method runMethod = scriptClass.getDeclaredMethod("run", runParams);
                        Object returnValue = runMethod.invoke(null, exa, ctx);
                        return (returnValue instanceof Integer) ? (int) returnValue : 0;
                    }
                }
                '''))
        rows = self.query('SELECT import_case_sensitive() FROM DUAL')
        self.assertRowsEqual([(42,)], rows)

    @useData(additional_env_declarations)
    def test_preprocessed_import_is_case_sensitive(self, additional_env_declaration):
        scripts = [
            ('my_module', additional_env_declaration, 'my_module', 4711),
            ('My_Module', additional_env_declaration, 'My_Module', 42),
            ('MY_MODULE', additional_env_declaration, 'MY_MODULE', 1234),
        ]
        sql = '''
                CREATE OR REPLACE java SCALAR SCRIPT
                "%s"()
                RETURNS int AS
                %s
                class %s {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return %d;
                    }
                }
                '''
        for triple in scripts:
            self.query(udf.fixindent(sql % triple))

        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                preprocessed_import_case_sensitive()
                RETURNS int AS
                ''' + additional_env_declaration + '''
                %import "My_Module";
                class PREPROCESSED_IMPORT_CASE_SENSITIVE {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return My_Module.run(exa, ctx);
                    }
                }
                '''))
        rows = self.query('SELECT preprocessed_import_case_sensitive() FROM DUAL')
        self.assertRowsEqual([(42,)], rows)

    @useData(additional_env_declarations)
    def test_preprocessed_import_tab(self, additional_env_declaration):
        self.query(udf.fixindent('''
            CREATE OR REPLACE java SCALAR SCRIPT
            X()
            RETURNS int AS
            ''' + additional_env_declaration + '''
            class X {
                static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return 1;
                }
            }
            '''))

        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                preprocessed_import_case_tab()
                RETURNS int AS
                ''' + additional_env_declaration + '''
                %import		X;
                class PREPROCESSED_IMPORT_CASE_TAB {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return X.run(exa, ctx);
                    }
                }
                '''))
        rows = self.query('SELECT preprocessed_import_case_tab() FROM DUAL')
        self.assertRowsEqual([(1,)], rows)

    @useData(additional_env_declarations)
    def test_preprocessed_import_quoted_whitespaces(self, additional_env_declaration):
        self.query(udf.fixindent('''
            CREATE OR REPLACE java SCALAR SCRIPT
            "X 	 "()
            RETURNS int AS
            ''' + additional_env_declaration + '''
            class X {
                static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return 1;
                }
            }
            '''))

        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                preprocessed_import_case_tab_end()
                RETURNS int AS
                ''' + additional_env_declaration + '''
                %import		"X 	 ";
                class PREPROCESSED_IMPORT_CASE_TAB_END {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return X.run(exa, ctx);
                    }
                }
                '''))
        rows = self.query('SELECT preprocessed_import_case_tab_end() FROM DUAL')
        self.assertRowsEqual([(1,)], rows)

    @useData(additional_env_declarations)
    def test_preprocessed_import_unquoted_whitespaces(self, additional_env_declaration):
        self.query(udf.fixindent('''
            CREATE OR REPLACE java SCALAR SCRIPT
            X()
            RETURNS int AS
            ''' + additional_env_declaration + '''
            class X {
                static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return 1;
                }
            }
            '''))

        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                preprocessed_import_case_tab_end()
                RETURNS int AS
                ''' + additional_env_declaration + '''
                %import		X 	 ;
                class PREPROCESSED_IMPORT_CASE_TAB_END {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return X.run(exa, ctx);
                    }
                }
                '''))
        rows = self.query('SELECT preprocessed_import_case_tab_end() FROM DUAL')
        self.assertRowsEqual([(1,)], rows)

    @useData(additional_env_declarations)
    def test_preprocessed_import_multiple(self, additional_env_declaration):
        scripts = [
            ('A', additional_env_declaration, 'A', 1),
            ('B', additional_env_declaration, 'B', 2),
            ('C', additional_env_declaration, 'C', 3),
        ]
        sql = '''
                CREATE OR REPLACE java SCALAR SCRIPT
                "%s"()
                RETURNS int AS
                %s
                %%import A;
                %%import B;
                %%import C;
                class %s {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return %d;
                    }
                }
                '''
        for triple in scripts:
            self.query(udf.fixindent(sql % triple))

        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                X()
                RETURNS int AS
                ''' + additional_env_declaration + '''
                %import A;
                %import B;
                %import C;
                class X {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return A.run(exa, ctx) + B.run(exa, ctx) + C.run(exa, ctx);
                    }
                }
                '''))
        rows = self.query('SELECT X() FROM DUAL')
        self.assertRowsEqual([(6,)], rows)

    @useData(additional_env_declarations)
    def test_preprocessed_import_recursive(self, additional_env_declaration):
        sql = '''
                CREATE OR REPLACE java SCALAR SCRIPT
                A()
                RETURNS int AS
                ''' + additional_env_declaration + '''
                %import B;
                class A {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return 1 + B.run(exa, ctx);
                    }
                }
                '''
        self.query(udf.fixindent(sql))
        sql = '''
                CREATE OR REPLACE java SCALAR SCRIPT
                B()
                RETURNS int AS
                ''' + additional_env_declaration + '''
                %import C;
                class B {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return 2 + C.run(exa, ctx);
                    }
                }
                '''
        self.query(udf.fixindent(sql))
        sql = '''
                CREATE OR REPLACE java SCALAR SCRIPT
                C()
                RETURNS int AS
                ''' + additional_env_declaration + '''
                class C {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return 3;
                    }
                }
                '''
        self.query(udf.fixindent(sql))

        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                X()
                RETURNS int AS
                ''' + additional_env_declaration + '''
                %import A;
                class X {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return A.run(exa, ctx);
                    }
                }
                '''))
        rows = self.query('SELECT X() FROM DUAL')
        self.assertRowsEqual([(6,)], rows)

    @useData(additional_env_declarations)
    def test_import_works(self, additional_env_declaration):
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT
            foo()
            RETURNS INT AS
            ''' + additional_env_declaration + '''
            %import bottom;
            class FOO {
                static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return BOTTOM.f();
                }
            }
            /
            '''))
        rows = self.query('SELECT foo() FROM DUAL')
        self.assertRowsEqual([(42,)], rows)

    @useData(additional_env_declarations)
    def test_import_is_semi_case_sensitive(self, additional_env_declaration):
        def check(import_name, classname, n):
            self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                foo()
                RETURNS INT AS
                %s
                %%import %s;
                class FOO {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return %s.f();
                    }
                }
                /''' % (additional_env_declaration, import_name, classname)))
            self.assertRowsEqual([(n,)],
                                 self.query('SELECT foo() FROM DUAL'))

        for name in 'bar', 'Bar', 'BAR':
            self.query(udf.fixindent('''
                CREATE java SCALAR SCRIPT
                "%s"()
                RETURNS INT AS
                %s
                class %s {
                    static int f() {
                            return %d;
                    }
                }
                /''' % (name, additional_env_declaration, name, sum(x.isupper() for x in name))
                                     ))

        check("bar", "BAR", 3)
        check("Bar", "BAR", 3)
        check("\"Bar\"", "Bar", 1)
        check("\"bar\"", "bar", 0)

    @useData(additional_env_declarations)
    def test_import_fails_for_lua_script(self, additional_env_declaration):
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT
            foo()
            RETURNS INT AS
            ''' + additional_env_declaration + '''
            %import bar;
            class FOO {
                static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return BAR.f();
                }
            }
            /
            '''))
        self.query(udf.fixindent('''
            CREATE lua SCALAR SCRIPT
            bar()
            RETURNS DOUBLE AS
            ''' + additional_env_declaration + '''
            function f()
                return 32
            end
            /
            '''))
        with self.assertRaisesRegex(Exception, 'VM error:.* wrong language LUA'):
            self.query('SELECT foo() FROM DUAL')

    @useData(additional_env_declarations)
    def test_import_fails_for_r_script(self, additional_env_declaration):
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT
            foo()
            RETURNS INT AS
            ''' + additional_env_declaration + '''
            %import bar;
            class FOO {
                static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return BAR.f();
                }
            }
            /
            '''))
        self.query(udf.fixindent('''
            CREATE r SCALAR SCRIPT
            bar()
            RETURNS DOUBLE AS
            ''' + additional_env_declaration + '''

            f <- function() {
                32
            }
            /
            '''))
        with self.assertRaisesRegex(Exception, 'VM error:.* wrong language R'):
            self.query('SELECT foo() FROM DUAL')

    @useData(additional_env_declarations)
    def test_import_fails_for_python_script(self, additional_env_declaration):
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT
            foo()
            RETURNS INT AS
            ''' + additional_env_declaration + '''
            %import bar;
            class FOO {
                static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return BAR.f();
                }
            }
            /
            '''))
        self.query(udf.fixindent('''
            CREATE python3 SCALAR SCRIPT
            bar()
            RETURNS DOUBLE AS
            ''' + additional_env_declaration + '''
            def f():
                return 32
            /
            '''))
        with self.assertRaisesRegex(Exception, 'VM error:.* wrong language PYTHON3'):
            self.query('SELECT foo() FROM DUAL')

    @useData([
        ('fn2', 'bottom', 'BOTTOM', ''),
        ('fn2', 'fn2.bottom', 'BOTTOM', ''),
        ('fn2', 'exa_db.fn2.bottom', 'BOTTOM', ''),
        ('fn3', 'fn2.bottom', 'BOTTOM', ''),
        ('fn3', 'exa_db.fn2.bottom', 'BOTTOM', ''),
        ('fn2', 'bottom', 'BOTTOM', "%env SCRIPT_OPTIONS_PARSER_VERSION=2;"),
        ('fn2', 'fn2.bottom', 'BOTTOM', "%env SCRIPT_OPTIONS_PARSER_VERSION=2;"),
        ('fn2', 'exa_db.fn2.bottom', 'BOTTOM', "%env SCRIPT_OPTIONS_PARSER_VERSION=2;"),
        ('fn3', 'fn2.bottom', 'BOTTOM', "%env SCRIPT_OPTIONS_PARSER_VERSION=2;"),
        ('fn3', 'exa_db.fn2.bottom', 'BOTTOM', "%env SCRIPT_OPTIONS_PARSER_VERSION=2;"),
    ])
    def test_import_works_with_qualified_names(self, schema, name, classname, additional_env_declaration):
        self.query('OPEN SCHEMA %s' % schema)
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT
            foo()
            RETURNS INT AS
            %s
            %%import %s;
            class FOO {
                static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return %s.f();
                }
            }
            /
            ''' % (additional_env_declaration, name, classname)))
        rows = self.query('SELECT foo() FROM DUAL')
        self.assertRowsEqual([(42,)], rows)

    @useData(additional_env_declarations)
    def test_chained_import_works_via_function_call(self, additional_env_declaration):
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT
            foo()
            RETURNS INT AS
            ''' + additional_env_declaration + '''
            %import bar;
            class FOO {
                static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return BAR.b();
                }
            }
            /
            '''))
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT
            bar()
            RETURNS INT AS
            ''' + additional_env_declaration + '''
            %import bottom;
            class BAR {
                static int b() {
                    return BOTTOM.f();
                }
            }
            /
            '''))
        rows = self.query('SELECT foo() FROM DUAL')
        self.assertRowsEqual([(42,)], rows)

    @useData(additional_env_declarations)
    def test_mutual_import_works(self, additional_env_declaration):
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT
            ping()
            RETURNS INT AS
            ''' + additional_env_declaration + '''
            %import pong;
            class PING {
                static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return ping(42);
                }
                static int ping(int n) {
                    if (n > 0)
                        return PONG.pong(n - 1) + 1;
                    else
                        return 0;
                }
            }
            /
            '''))
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT
            pong()
            RETURNS INT AS
            ''' + additional_env_declaration + '''
            %import ping;
            class PONG {
                static int pong(int n) {
                    if (n > 0)
                        return PING.ping(n - 1) + 1;
                    else
                        return 0;
                }
            }
            /
            '''))
        rows = self.query('SELECT ping() FROM DUAL')
        self.assertRowsEqual([(42,)], rows)

    @useData(additional_env_declarations)
    def testImportWithView(self, additional_env_declaration):
        self.createUser("foo", "foo")
        self.commit()
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT
            spot42542script()
            RETURNS INT AS
            ''' + additional_env_declaration + '''
            %import bottom;
            class SPOT42542SCRIPT {
                static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return BOTTOM.f();
                }
            }
            /
            '''))
        self.query("create or replace view FN2.spot42542_view as SELECT spot42542script() as col FROM DUAL")
        self.query("grant select on FN2.spot42542_view to foo")
        self.commit()
        rows = self.query('''select * from FN2.spot42542_view''')
        self.assertRowsEqual([(42,)], rows)
        foo_conn = self.getConnection('foo', 'foo')
        rows = foo_conn.query('''select * from FN2.spot42542_view''')
        self.assertRowsEqual([(42,)], rows)


if __name__ == '__main__':
    udf.main()
