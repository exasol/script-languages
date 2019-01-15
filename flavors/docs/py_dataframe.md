# Pandas DataFrame support
The `python3-ds-*` flavors now have direct DataFrame support for accessing and emitting data in Exasol.

## Accessing data
Instead of accessing each column of a row individually and calling `next()` for every row, the `get_dataframe(num_rows, start_col)` function can now be called which returns a block of data as a Pandas DataFrame.

The parameters of `get_dataframe` are the following.

| Parameter | Value | Description |
| ----- | ----- | ----- |
| num_rows | 'all' or a positive integer. Default 1. | The number of rows to be returned in the DataFrame.<br>Please keep memory usage in mind when setting this value. |
| start_col | A nonnegative integer. Default 0. | The UDF column (0-based) which specifies the start of the data to be included in the returned DataFrame. The data for `start_col` and all columns thereafter will be included in the DataFrame. |

`get_dataframe` will return a DataFrame containing `num_rows` rows or a lesser number if `num_rows` are not available. If there are zero rows available, `get_dataframe` will return `None`. The DataFrame column labels will be set to the corresponding UDF parameter names for the columns. The UDF data iterator will then point to the next row (i.e. following the last row in the DataFrame) just as with `next()`.

## Emitting data
An entire DataFrame can be emitted by passing it to `emit()` just as with single values. Each column of the DataFrame will be  automatically converted to a column in the result set.

## Example
```python
CREATE OR REPLACE TABLE DF_TEST_TABLE(C0 DOUBLE, C1 INT, C2 VARCHAR(50));
INSERT INTO DF_TEST_TABLE VALUES (0.1, 1, 'a'), (0.2, 2, 'b');

CREATE OR REPLACE PYTHON3 SET SCRIPT DF_TEST(C0 DOUBLE, C1 INT, C2 VARCHAR(50))
EMITS (C1 INT, C2 VARCHAR(50), C3 BOOL) AS
import pandas as pd
def run(ctx):
  df = ctx.get_dataframe(num_rows='all', start_col=1)
  df['BOOLS'] = [True, False]
  ctx.emit(df)
/

SELECT DF_TEST(C0, C1, C2) FROM DF_TEST_TABLE;
```
Output:

| C1<br>(Type: INT) | C2<br>(Type: VARCHAR(50)) | C3<br>(Type: BOOL) |
| --- | --- | --- |
| 1 | a | TRUE |
| 2 | b | FALSE |
