#!/usr/bin/env python3

import json

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

                import json
                import sys
                from pathlib import Path

                def normalize(path_value):
                    try:
                        return str(Path(path_value).resolve())
                    except Exception:
                        return str(path_value)

                def run(ctx):
                    executable = normalize(sys.executable)
                    path_entries = [entry for entry in sys.path if entry]
                    first_path = normalize(path_entries[0]) if path_entries else ""

                    return json.dumps(
                        {
                            "sys_executable": executable,
                            "first_sys_path_entry": first_path,
                        }
                    )
                /
                '''))

        rows = self.query("SELECT check_conda_runtime()")
        result = json.loads(rows[0][0])

        executable = result["sys_executable"]
        first_path = result["first_sys_path_entry"]

        self.assertTrue(
            executable.startswith("/opt/conda"),
            f"sys.executable shall start with /opt/conda; But the value is {executable}",
        )
        self.assertTrue(
            first_path.startswith("/opt/conda"),
            f"sys.path first entry shall start with /opt/conda; But the value is {first_path}",
        )


if __name__ == '__main__':
    udf.main()
