#!/usr/bin/env python3

from exasol_python_test_framework import udf


class AptSysPathCheck(udf.TestCase):
    def setUp(self):
        self.schema = "SYS_PATH_CHECK_APT"
        self.query(f'CREATE SCHEMA {self.schema}', ignore_errors=True)
        self.query(f'OPEN SCHEMA {self.schema}', ignore_errors=True)

    def tearDown(self):
        self.query(f'DROP SCHEMA {self.schema} CASCADE', ignore_errors=True)

    def test_apt_runtime_does_not_use_conda_paths(self):
        """
        For apt-based Python runtimes, sys.executable and sys.path entries
        must not point to /opt/conda.
        """
        self.query(udf.fixindent('''
                CREATE OR REPLACE python3 SCALAR SCRIPT
                check_apt_runtime_no_conda_paths()
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
                    if "/opt/conda" in executable:
                        failures.append(f"sys.executable contains /opt/conda: {executable}")

                    for path_entry in [entry for entry in sys.path if entry]:
                        normalized_path = normalize(path_entry)
                        if "/opt/conda" in normalized_path:
                            failures.append(
                                f"sys.path contains /opt/conda entry: {normalized_path}"
                            )

                    return "; ".join(failures) if failures else "OK"
                /
                '''))

        rows = self.query("SELECT check_apt_runtime_no_conda_paths()")
        result = rows[0][0]
        self.assertEqual(result, "OK", f"Apt runtime conda-path check failed. Got: {result}")


if __name__ == '__main__':
    udf.main()
