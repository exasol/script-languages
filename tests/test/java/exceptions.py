#!/usr/bin/env python3
import os
import subprocess

from exasol_python_test_framework import udf


class ExceptionTest(udf.TestCase):

    def test_xml_processing(self):
        q = '''DROP SCHEMA IF EXISTS t1 CASCADE;
               CREATE SCHEMA t1;
               create or replace java scalar script
               throw_exception()
               EMITS (firstname VARCHAR(100), lastname VARCHAR(100)) AS
               class THROW_EXCEPTION {
                   public static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                       try {
                           throw new RuntimeException("Your Message");
                       } catch (final Exception ex) {
                           throw new RuntimeException("Got exception", ex);
                       }
                   }
               }; /
               SELECT throw_exception() FROM DUAL;
            '''

        cmd = [os.environ.get('EXAPLUS'), "-c", udf.opts.server, "-u", "sys", "-P", "exasol",
               "-sql", q]

        exaplus = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
        exaplus.wait()
        raise RuntimeError(f"exaplus.stdout:{exaplus.stdout.read()}")


if __name__ == '__main__':
    udf.main()
