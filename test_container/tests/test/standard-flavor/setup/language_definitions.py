#!/usr/bin/env python3

from exasol_python_test_framework import udf


class LanguageDefinitions(udf.TestCase):
    def setUp(self):
        self.query('create schema language_definitions', ignore_errors=True)

    def test_no_python2_bin(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SCALAR SCRIPT language_definitions.check_language_definitions() returns int AS
            from pathlib import Path
            def run(ctx): 
                p = Path("/buckets/bfsdefault/myudfs")
                lang_def_json_list = list(p.glob("**/language_definitions.json"))
                return len(lang_def_json_list)
            /
            '''))
        try:
            rows = self.query('''SELECT language_definitions.check_language_definitions()
                                 FROM dual''')
            self.assertRowsEqual([(1,)], rows)
        except:
            print("Error checking 'language_definitions.json' files.")
            raise


if __name__ == '__main__':
    udf.main()
