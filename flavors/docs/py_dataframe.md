# Pandas DataFrame support
The `python3-ds-*` flavors now have direct DataFrame support for accessing and emitting data in Exasol.

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


### Mixed usage of get_dataframes and iterator

Some special attention needs the case were you mix the usage of get_dataframes and the iterator functions. The defined behavior is as following:

A get_dataframes call consumes as many rows as specified in num_rows and the after this the iterator points to next row after the consumed ones. The following example iterates over a Table with number from 0 to 9. In each iteration the get_dataframes call consumes exactly one row. It than emits this row and additionally emits the row at which the iterator points at after the call of get_dataframes. 

```python
import pyexasol
import textwrap

conn = pyexasol.connect(dsn=EXASOL_HOST, user=EXASOL_USER, password=EXASOL_PASSWORD, compression=True)
conn.execute(f"ALTER SESSION SET SCRIPT_LANGUAGES='{EXASOL_SCRIPT_LANGUAGES}'")

conn.execute('CREATE OR REPLACE TABLE TEST3(C0 INT IDENTITY, C1 INTEGER)')
for i in range(10):
    conn.execute('INSERT INTO TEST3 (C1) VALUES (%s)' % i)

conn.execute(textwrap.dedent('''
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
    '''))
rows = conn.execute('SELECT foo(C1) FROM TEST3').fetchall()
print(rows)
conn.close()

# Expected Output
# [('df_0',), ('getattr_1',), ('eob',), ('df_1',), ('getattr_2',), ('eob',), ('df_2',), ('getattr_3',), ('eob',), ('df_3',), ('getattr_4',), ('eob',), ('df_4',), ('getattr_5',), ('eob',), ('df_5',), ('getattr_6',), ('eob',), ('df_6',), ('getattr_7',), ('eob',), ('df_7',), ('getattr_8',), ('eob',), ('df_8',), ('getattr_9',), ('eob',), ('df_9',), ('eoi',)]
```


