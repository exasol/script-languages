# Exasol script language flavors overview
Script language flavors are configurations of languages and libraries to be available in script languages for the Exasol database.
Currently, we have
* the `standard` flavors: These flavors include three langauage implementations: Java, Python, and R and a collection of typical libraries
* the `python3-ds` flavors: These flavors include Python3 as language and a number of typical data science and machine learning libraries
* the `fancyr` flavors: These flavors include R as language and a large collection of popular R packages.

# Naming convention
Script language flavors for different purposes come in different versions that are reflected by their name.
The naming convention is like this:

`<flavor-name>-EXASOL-<minimum-Exasol-Version>`

This allows us to provide new versions of a flavor when new features become available in Exasol that are not supported in older flavors.
For instance, the flavors for Exasol 6.1.0 support newer Linux distributions as their basis and hence, overall contain newer versions of libraries and languages.

So, in order to find the correct version of a flavor for your version of Exasol, simply take the latest container with version less than or equal to your Exasol version.
So for Exasol 6.1.1, you would use the `*-EXASOL-6.1.0` flavors, while for Exasol 6.0.14 you would use the `*-EXASOL-6.0.0` flavors.

# Flavor-specific features
## python3-ds
### Pandas DataFrame support
The `python3-ds-*` flavors now have direct DataFrame support for accessing and emitting data in Exasol.

#### Accessing data
Instead of accessing each column of a row individually and calling `next()` for every row, the `get_dataframe(num_rows, start_col)` function can now be called which returns a block of data as a Pandas DataFrame. The parameters of `get_dataframe` are the following.

| Parameter | Description |
| ----- | ----- |
| num_rows | The number of rows to be returned in the Dataframe. Values: 'all' or a positive integer. Default: 1. |
| start_col | The UDF column (0-based) which specifies the start of the data to be included in the returned DataFrame. The data for `start_col` and all columns thereafter will be included in the DataFrame. Values: a positive integer. Default: 0. |

#### Emitting data
`emit()`
