#!/usr/bin/env python3

from exasol_python_test_framework import udf


class SysExecutableTest(udf.TestCase):
    def setUp(self):
        self.schema = "SYS_EXECUTABLE_TEST"
        self.query(f'CREATE SCHEMA {self.schema}', ignore_errors=True)
        self.query(f'OPEN SCHEMA {self.schema}', ignore_errors=True)

    def tearDown(self):
        self.query(f'DROP SCHEMA {self.schema} CASCADE', ignore_errors=True)

    def test_environment_of_udf_and_sys_executabe(self):
        """
        Older version of pyexasol used subprocess(sys.executable, ...) when using httptransport.

	This failed in SLC version 6.0.0, because the subprocess didn't find the package pyexasol.  This was
        caused by two Python versions being installed in SLC version 6.0.0 and the UDF's sys.executable
        returned an interpreter different from the one running the UDF.

	The underlying issue will be fixed with https://github.com/exasol/script-languages-release/issues/872.

	This test only checks that the environment in the python interpreter of the UDF is close to the
        environment in the python interpreter returned by sys.executable.  For that, the test compares the
        interpreter's version and implementation (interpreter-specific value) and the sys.path (search path
        for packages).  This should make sure, that everything that works in the python interpreter UDF works
        also in the python interpreter returned by the sys.executable.
        """
        self.query(udf.fixindent('''
                CREATE OR REPLACE python3 SCALAR SCRIPT
                check_sys_executable()
                EMITS (UDF_RESULT VARCHAR(10000), SYS_EXECUTABLE_RESULT VARCHAR(10000)) AS
                 
                import sys
                import subprocess

                def get_property_from_sys_executable(command):
                    return subprocess.check_output(
                            [sys.executable, "-c", command], 
                            encoding="UTF-8").strip()

                def run(ctx):
                    udf_python_version = str(sys.version_info)
                    sys_executable_version = get_property_from_sys_executable("import sys; print(sys.version_info)")
                    ctx.emit(udf_python_version, sys_executable_version)

                    udf_python_implementation = str(sys.implementation)
                    sys_executable_implementation = get_property_from_sys_executable("import sys; print(sys.implementation)")
                    ctx.emit(udf_python_implementation, sys_executable_implementation)
                    
                    udf_python_path = str(sys.path)
                    sys_executable_path = get_property_from_sys_executable("import sys; print(sys.path)")
                    ctx.emit(udf_python_path, sys_executable_path)
                /
                '''))
        rows = self.query("select check_sys_executable()")
        self.assertEqual(rows[0][0], rows[0][1])


if __name__ == '__main__':
    udf.main()
