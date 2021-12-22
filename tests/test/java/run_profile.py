#!/usr/bin/env python3
import datetime

from exasol_python_test_framework import udf
from exasol_python_test_framework.udf.udf_debug import UdfDebugger


class SimpleJavaTest(udf.TestCase):
    def setUp(self):
        self.query('CREATE SCHEMA FN2', ignore_errors=True)
        self.query('OPEN SCHEMA FN2')

    def test_simple(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                simple()
                RETURNS int AS
                import java.time.LocalDateTime;
                import java.time.format.DateTimeFormatter;
                class SIMPLE {
                    static void main(String[] args) {
                        int i = 0;
                    }

                    static int run(ExaMetadata exa, ExaIterator ctx) {
                        LocalDateTime localDateTime = LocalDateTime.now();
                        DateTimeFormatter dateTimeFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSS");
                        String forDate = localDateTime.format(dateTimeFormatter);
                        System.out.println("PROFILE[UDF] " + forDate);
                        return 0;
                    }
                }
                /
                '''))
        with UdfDebugger(test_case=self):
            ct = datetime.datetime.now()
            print(f"PROFILING[START SQL QUERY] {ct.strftime('%H:%M:%S.%f')}")
            row = self.query('SELECT simple() FROM DUAL')[0]


if __name__ == '__main__':
    udf.main()

