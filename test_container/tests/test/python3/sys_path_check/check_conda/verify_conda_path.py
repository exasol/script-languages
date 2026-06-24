#!/usr/bin/env python3

from exasol_python_test_framework import udf


class CondaSysPathCheck(udf.TestCase):
    def setUp(self):
        self.schema = "SYS_PATH_CHECK_CONDA"
        self.query(f'CREATE SCHEMA {self.schema}', ignore_errors=True)
        self.query(f'OPEN SCHEMA {self.schema}', ignore_errors=True)

    def tearDown(self):
        self.query(f'DROP SCHEMA {self.schema} CASCADE', ignore_errors=True)

    def test_conda_interpreter_and_sys_path(self):
        """
        For conda flavor, Python runtimes shall be:
        1) sys.executable resolves to /opt/conda
        2) first non-empty sys.path entry resolves to /opt/conda
        """
        self.query(udf.fixindent('''
                CREATE OR REPLACE python3 SCALAR SCRIPT
                check_conda_runtime()
                RETURNS VARCHAR(10000) AS

                import sys
                from pathlib import Path

                def normalize(path_value):
                    try:
                        return str(Path(path_value).resolve())
                    except Exception:
                        return str(path_value)

                def run(ctx):
                    failures = []

                    executable = normalize(sys.executable)
                    if not executable.startswith("/opt/conda"):
                        failures.append(f"sys.executable outside /opt/conda: {executable}")

                    path_entries = [entry for entry in sys.path if entry]
                    if not path_entries:
                        failures.append("sys.path has no non-empty entries")
                    else:
                        first_path = normalize(path_entries[0])
                        if not first_path.startswith("/opt/conda"):
                            failures.append(
                                f"sys.path first entry outside /opt/conda: {first_path}"
                            )

                    return "; ".join(failures) if failures else "OK"
                /
                '''))

        rows = self.query("SELECT check_conda_runtime()")
        result = rows[0][0]
        self.assertEqual(result, "OK", f"Conda runtime check failed. Got: {result}")


if __name__ == '__main__':
    udf.main()
