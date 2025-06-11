#!/usr/bin/env python3

from exasol_python_test_framework import udf
import math


class NvidiaSMITest(udf.TestCase):
    def setUp(self):
        self.query('create schema nvidiasmi', ignore_errors=True)

    def test_nvidia_smi_available(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE PYTHON3 SCALAR SCRIPT test_nvidia_smi_available()
                RETURNS VARCHAR(10000) AS
                 %perInstanceRequiredAcceleratorDevices GpuNvidia;
        
                import subprocess
        
                def run(ctx):
                    cmd = ["nvidia-smi", "-L"] #List GPU's
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    exit_code = process.wait()
                    
                    if exit_code != 0:
                        raise Exception(f"nvidia-smi returned non-zero exit code: '{process.stderr}'")
                    return process.stdout.read()
                /
                '''))

        row = self.query("SELECT nvidiasmi.test_nvidia_smi_available();")[0]
        self.assertIn("GPU 0", row[0])


if __name__ == '__main__':
    udf.main()
