#!/usr/bin/env python3

from exasol_python_test_framework import udf


class TestGPUEnabled(udf.TestCase):

    def test_gpu_enabled(self):
        select_sql = """
                     SELECT PARAM_VALUE, PARAM_NAME
                     FROM EXA_METADATA
                     WHERE PARAM_NAME LIKE '%accelerator%'
                     ORDER BY PARAM_NAME; \
                     """
        rows = self.query(select_sql)
        self.assertRowsEqual(
            [
                ("1", "acceleratorDeviceDetected"),
                ("1", "acceleratorDeviceGpuNvidiaDetected"),
            ],
            rows,
        )


if __name__ == "__main__":
    udf.main()
