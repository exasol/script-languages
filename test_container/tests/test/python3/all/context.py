#!/usr/bin/env python3

from exasol_python_test_framework import udf


NUM_ROW_INSERTS = 15
PAYLOAD = '0123456789ABCDEF0123456789ABCDEF'


class CtxSizeAPITest(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA context_api CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA context_api')

        self.query('''
            CREATE OR REPLACE TABLE ctx_size_input(
                a VARCHAR(32),
                i DECIMAL(9,0)
            )
        ''')
        self.query(f'''
            INSERT INTO ctx_size_input VALUES ('{PAYLOAD}', 0)
        ''')
        self.query(f'''
            INSERT INTO ctx_size_input VALUES ('{PAYLOAD}', 1)
        ''')
        self._duplicate_rows('ctx_size_input', 'a, i')
        self.commit()

    def _duplicate_rows(self, table_name, column_names):
        for i in range(NUM_ROW_INSERTS):
            self.query(
                f'INSERT INTO {table_name} ({column_names}) '
                f'SELECT {column_names} FROM {table_name}'
            )

    def test_ctx_size_scalar_udf(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SCALAR SCRIPT
            scalar_ctx_size(a VARCHAR(32))
            RETURNS INTEGER AS

            def run(ctx):
                return ctx.size()
            /
        '''))

        rows = self.query('''
            SELECT MAX(scalar_ctx_size(a))
            FROM ctx_size_input
        ''')
        self.assertRowsEqual([(2048,)], rows)

    def test_ctx_size_set_udf(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SET SCRIPT
            set_ctx_size(a VARCHAR(32), i INTEGER)
            RETURNS INTEGER AS

            def run(ctx):
                return ctx.size()
            /
        '''))

        rows = self.query('''
            SELECT set_ctx_size(a, i)
            FROM ctx_size_input
            GROUP BY i
        ''')
        self.assertRowsEqual([(32768,), (32768,)], rows)


class CtxGetitemAPITest(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA context_api CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA context_api')

    def test_ctx_getitem_supports_non_identifier_input_names(self):
        self.query('''
            CREATE OR REPLACE TABLE t(
                "valid_name" INTEGER,
                "123col" INTEGER,
                "has-hyphen" INTEGER
            )
        ''')
        self.query('''
            INSERT INTO t VALUES (1, 2, 3)
        ''')

        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SCALAR SCRIPT
            scalar_ctx_getitem("valid_name" INTEGER, "123col" INTEGER, "has-hyphen" INTEGER)
            RETURNS INTEGER AS

            def run(ctx):
                return ctx.valid_name + ctx["123col"] + ctx["has-hyphen"]
            /
        '''))

        rows = self.query('SELECT scalar_ctx_getitem("valid_name", "123col", "has-hyphen") FROM t')
        self.assertRowsEqual([(6,)], rows)


if __name__ == '__main__':
    udf.main()