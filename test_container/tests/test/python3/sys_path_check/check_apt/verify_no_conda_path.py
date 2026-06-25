#!/usr/bin/env python3

import json

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
                    normalized_sys_path_entries = [
                        normalize(path_entry) for path_entry in sys.path if path_entry
                    ]

                    return json.dumps(
                        {
                            "sys_executable": executable,
                            "sys_path_entries": normalized_sys_path_entries,
                        }
                    )
                /
                '''))

        rows = self.query("SELECT check_apt_runtime_no_conda_paths()")
        result = json.loads(rows[0][0])

        executable = result["sys_executable"]
        sys_path_entries = result["sys_path_entries"]

        self.assertNotIn(
            "/opt/conda",
            executable,
            f"sys.executable shall not contain /opt/conda; But the value is {executable}",
        )
        for path_entry in sys_path_entries:
            self.assertNotIn(
                "/opt/conda",
                path_entry,
                f"sys.path entry shall not contain /opt/conda; But the value is {path_entry}",
            )


if __name__ == '__main__':
    udf.main()
