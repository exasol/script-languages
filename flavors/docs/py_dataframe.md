# Pandas DataFrame support
The `standard-EXASOL-6.1.0` flavor (per default avialable since in EXASOL 6.2.*) and the `python3.6-ds-*` flavors now have direct DataFrame support for accessing and emitting data in Exasol.

## Accessing data
Instead of accessing each column of a row individually and calling `next()` for every row, the `get_dataframe(num_rows, start_col)` function can now be called which returns a block of data as a Pandas DataFrame.

The parameters of `get_dataframe` are the following.

| Parameter | Value | Description |
| ----- | ----- | ----- |
| num_rows | 'all' or a positive integer. Default 1. | The number of rows to be returned in the DataFrame.<br>Please keep memory usage in mind when setting this value. |
| start_col | A nonnegative integer. Default 0. | The UDF column (0-based) which specifies the start of the data to be included in the returned DataFrame. The data for `start_col` and all columns thereafter will be included in the DataFrame. |

`get_dataframe` will return a DataFrame containing `num_rows` rows or a lesser number if `num_rows` are not available. If there are zero rows available, `get_dataframe` will return `None`. The DataFrame column labels will be set to the corresponding UDF parameter names for the columns. After calling `get_dataframe`, the UDF data iterator will point to the next row (i.e. following the last row in the DataFrame) just as with `next()`.

## Emitting data
An entire DataFrame can be emitted by passing it to `emit()` just as with single values. Each column of the DataFrame will be  automatically converted to a column in the result set.

## Example
```python
CREATE OR REPLACE TABLE DF_TEST_TABLE(C0 DOUBLE, C1 INT, C2 VARCHAR(50));
INSERT INTO DF_TEST_TABLE VALUES (0.1, 1, 'a'), (0.2, 2, 'b');

CREATE OR REPLACE PYTHON3 SET SCRIPT DF_TEST(C0 DOUBLE, C1 INT, C2 VARCHAR(50))
EMITS (C1 INT, C2 VARCHAR(50), C3 BOOL) AS
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


## Mixed usage of get_dataframe() and iterator

Some special attention needs to be paid to the case where you mix the usage of `get_dataframe` and the iterator functions. The defined behavior is the following: A `get_dataframe` call consumes as many rows as specified in `num_rows`, and after this the iterator points to next row after the consumed ones.

The following example iterates over a table with numbers from 0 to 3. In each iteration the `get_dataframe` call consumes exactly one row. It then emits this row and additionally the row to which the iterator points after calling `get_dataframe`.

```python
CREATE OR REPLACE TABLE TEST3(C0 INT IDENTITY, C1 INTEGER);
INSERT INTO TEST3 (C1) VALUES (0), (1), (2), (3);

CREATE OR REPLACE PYTHON3 SET SCRIPT foo(C1 INTEGER) EMITS(R VARCHAR(1000)) AS
def run(ctx):
    BATCH_ROWS = 1
    while True:
        df = ctx.get_dataframe(num_rows=BATCH_ROWS)
        if df is None:
            break
        ctx.emit(df.applymap(lambda x: "df_"+str(x)))
        try:
            ctx.emit("getattr_"+str(ctx.C1))
            ctx.emit("eob") # end of batch
        except:
            ctx.emit("eoi") # end of iteration
/
SELECT foo(C1) FROM TEST3;
```

Output:


| C1<br>(Type: INT) |
| --- |
| df_0 |
| getattr_1 |
| eob |
| df_1 |
| getattr_2 |
| eob |
| df_2 |
| getattr_3 |
| eob |
|  df_3 |
| eoi |
