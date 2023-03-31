#!/usr/bin/env python3

from exasol_python_test_framework import udf


class REmptyUDF(udf.TestCase):
    def setUp(self):
        self.schema = "REmptyUDF"
        self.query(f'DROP SCHEMA {self.schema} CASCADE', ignore_errors=True)
        self.query(f'CREATE SCHEMA {self.schema}')

    def test_return_int_scalar_returns(self):
        self.query(udf.fixindent('''
                CREATE r SCALAR SCRIPT return_int_scalar_returns(i int)
                RETURNS int AS

                run <- function(ctx) {
                    return(10)
                }
                /
                '''))
        rows = self.query('select return_int_scalar_returns(i) from VALUES BETWEEN 1 AND 9 AS t(i)')
        self.assertRowsEqual([(10,) for i in range(1,10)], rows)

    def test_return_vector_int_scalar_returns(self):
        self.query(udf.fixindent('''
                CREATE r SCALAR SCRIPT return_vector_int_scalar_returns(i int)
                RETURNS int AS

                run <- function(ctx) {
                    return(c(10))
                }
                /
                '''))
        rows = self.query('select return_vector_int_scalar_returns(i) from VALUES BETWEEN 1 AND 9 AS t(i)')
        self.assertRowsEqual([(10,) for i in range(1,10)], rows)

    def test_return_int_scalar_returns_batch(self):
        self.query(udf.fixindent('''
                CREATE r SCALAR SCRIPT return_int_scalar_returns_batch(i int)
                RETURNS int AS

                run <- function(ctx) {
                    ctx$next_row(NA)
                    return(10)
                }
                /
                '''))
        rows = self.query('select return_int_scalar_returns_batch(i) from VALUES BETWEEN 1 AND 9 AS t(i)')
        self.assertRowsEqual([(10,) for i in range(1,10)], rows)

    def test_return_vector_int_scalar_returns_batch(self):
        self.query(udf.fixindent('''
                CREATE r SCALAR SCRIPT return_vector_int_scalar_returns_batch(i int)
                RETURNS int AS

                run <- function(ctx) {
                    ctx$next_row(NA)
                    return(c(10))
                }
                /
                '''))
        rows = self.query('select return_vector_int_scalar_returns_batch(i) from VALUES BETWEEN 1 AND 9 AS t(i)')
        self.assertRowsEqual([(10,) for i in range(1,10)], rows)

    def test_return_incomplete_vector_int_scalar_returns_batch(self):
        self.query(udf.fixindent('''
                CREATE r SCALAR SCRIPT return_incomplete_vector_int_scalar_returns_batch(i int)
                RETURNS int AS

                run <- function(ctx) {
                    ctx$next_row(NA)
                    return(c(9,10))
                }
                /
                '''))
        rows = self.query('select return_incomplete_vector_int_scalar_returns_batch(i) from VALUES BETWEEN 1 AND 9 AS t(i)')
        self.assertRowsEqual([(9,),(10,),(9,),(10,),(9,),(10,),(9,),(10,),(9,)], rows)

    def test_return_null_scalar_returns(self):
        self.query(udf.fixindent('''
                CREATE r SCALAR SCRIPT return_null_scalar_returns(i int)
                RETURNS int AS

                run <- function(ctx) {
                    return(NULL)
                }
                /
                '''))
        rows = self.query('select return_null_scalar_returns(i) from VALUES BETWEEN 1 AND 9 AS t(i)')
        self.assertRowsEqual([(None,) for i in range(1,10)], rows)

    def test_return_vector_null_scalar_returns(self):
        self.query(udf.fixindent('''
                CREATE r SCALAR SCRIPT return_vector_null_scalar_returns(i int)
                RETURNS int AS

                run <- function(ctx) {
                    return(c(NULL))
                }
                /
                '''))
        rows = self.query('select return_vector_null_scalar_returns(i) from VALUES BETWEEN 1 AND 9 AS t(i)')
        self.assertRowsEqual([(None,) for i in range(1,10)], rows)
    
    def test_return_null_scalar_returns_batch(self):
        self.query(udf.fixindent('''
                CREATE r SCALAR SCRIPT return_null_scalar_returns_batch(i int)
                RETURNS int AS

                run <- function(ctx) {
                    ctx$next_row(NA)
                    return(NULL)
                }
                /
                '''))
        rows = self.query('select return_null_scalar_returns_batch(i) from VALUES BETWEEN 1 AND 9 AS t(i)')
        self.assertRowsEqual([(None,) for i in range(1,10)], rows)

    def test_return_vector_null_scalar_returns_batch(self):
        self.query(udf.fixindent('''
                CREATE r SCALAR SCRIPT return_vector_null_scalar_returns_batch(i int)
                RETURNS int AS

                run <- function(ctx) {
                    ctx$next_row(NA)
                    return(c(NULL))
                }
                /
                '''))
        rows = self.query('select return_vector_null_scalar_returns_batch(i) from VALUES BETWEEN 1 AND 9 AS t(i)')
        self.assertRowsEqual([(None,) for i in range(1,10)], rows)



    def test_empty_scalar_returns(self):
        self.query(udf.fixindent('''
                CREATE r SCALAR SCRIPT empty_scalar_returns(i int)
                RETURNS int AS

                run <- function(ctx) {
                }
                /
                '''))
        rows = self.query('select empty_scalar_returns(i) from VALUES BETWEEN 1 AND 9 AS t(i)')
        self.assertRowsEqual([(None,) for i in range(1,10)], rows)

    def test_return_na_scalar_returns(self):
        self.query(udf.fixindent('''
                CREATE r SCALAR SCRIPT return_na_scalar_returns(i int)
                RETURNS int AS

                run <- function(ctx) {
                    return(NA)
                }
                /
                '''))
        rows = self.query('select return_na_scalar_returns(i) from VALUES BETWEEN 1 AND 9 AS t(i)')
        self.assertRowsEqual([(None,) for i in range(1,10)], rows)

    def test_return_vector_na_scalar_returns(self):
        self.query(udf.fixindent('''
                CREATE r SCALAR SCRIPT return_vector_na_scalar_returns(i int)
                RETURNS int AS

                run <- function(ctx) {
                    return(c(NA))
                }
                /
                '''))
        rows = self.query('select return_vector_na_scalar_returns(i) from VALUES BETWEEN 1 AND 9 AS t(i)')
        self.assertRowsEqual([(None,) for i in range(1,10)], rows)

    def test_return_na_scalar_returns_batch(self):
        self.query(udf.fixindent('''
                CREATE r SCALAR SCRIPT return_na_scalar_returns_batch(i int)
                RETURNS int AS

                run <- function(ctx) {
                    ctx$next_row(NA)
                    return(NA)
                }
                /
                '''))
        rows = self.query('select return_na_scalar_returns_batch(i) from VALUES BETWEEN 1 AND 9 AS t(i)')
        self.assertRowsEqual([(None,) for i in range(1,10)], rows)

    def test_return_vector_na_scalar_returns_batch(self):
        self.query(udf.fixindent('''
                CREATE r SCALAR SCRIPT return_vector_na_scalar_returns_batch(i int)
                RETURNS int AS

                run <- function(ctx) {
                    ctx$next_row(NA)
                    return(c(NA))
                }
                /
                '''))
        rows = self.query('select return_vector_na_scalar_returns_batch(i) from VALUES BETWEEN 1 AND 9 AS t(i)')
        self.assertRowsEqual([(None,) for i in range(1,10)], rows)


if __name__ == '__main__':
    udf.main()
