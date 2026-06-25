# Generic R vs Python3 Test Comparison

## Scope

This document compares the generic test coverage under:

- `test_container/tests/test/generic/r`
- `test_container/tests/test/generic/python3`

The main question was whether all tests in the R file `import_alias.py` are also available in the Python 3 counterpart, and then whether the same pattern holds for the other files in the generic R suite.

## Executive Summary

All tests from `test_container/tests/test/generic/r/import_alias.py` are covered in the Python 3 counterpart `test_container/tests/test/generic/python3/import_alias.py`.

The mismatch goes in the other direction: the Python 3 file contains 6 additional tests that are not present in the generic R file.

More broadly, the generic R suite is not a one-to-one mirror of the Python 3 suite. The strongest repo-level evidence is provenance:

- The generic R files were introduced together in commit `23d185b` on 2026-06-24 with message `committing generic tests of R`.
- The Python 3 generic files come from an earlier refactor commit `4e5e66b` on 2026-06-17 with message `#640: Refactored generic tests python3 (#645)`.

That strongly suggests the generic R suite is a newer subset port rather than a parity-complete copy.

## Import Alias

### Result

All tests in `test_container/tests/test/generic/r/import_alias.py` are present in or covered by `test_container/tests/test/generic/python3/import_alias.py`.

Python 3 has these additional tests:

- `test_import_use_all_subselect`
- `test_import_use_connection_fooconn_fails_for_user_foo`
- `test_import_use_connection_fooconn_for_user_foo_and_view`
- `test_prepared_statement_params`
- `test_prepared_statement_conn`
- `test_import_in_lua_scripting`

### Reasoning for the missing R-side tests

#### `test_import_use_all_subselect`

This looks unported, not unsupported.

Evidence:

- The older R language-specific test already exercises `subselect_column_types` and `subselect_column_names` in `test_container/tests/lang/r/import_alias.sql`.
- The generic R file defines `impal_use_all(...)`, but the test only covers the non-subselect path.

Conclusion:

- The capability appears to exist for R.
- The generic R test simply does not include the equivalent subselect coverage.

#### `test_import_use_connection_fooconn_fails_for_user_foo`
#### `test_import_use_connection_fooconn_for_user_foo_and_view`

These are privilege / access-control regression tests rather than basic import-spec coverage.

Evidence:

- The Python 3 file brings in `exatest.ODBCClient` and helper methods for user creation and alternate-session execution.
- The generic R file stays focused on import-spec values and does not bring in that broader access-control harness.
- One of the Python 3 tests is explicitly skipped with reason: `IMPORT FROM SCRIPT cannot be used in view definitions`.

Conclusion:

- These tests were likely omitted from the generic R port because they belong to broader privilege/integration coverage rather than core R import alias behavior.
- This is not strong evidence that R cannot support them.

#### `test_prepared_statement_params`
#### `test_prepared_statement_conn`

These look like SQL parser / statement-handling regression tests, not R-language-specific behavior tests.

Conclusion:

- Their absence in generic R looks like a porting omission or scope trimming.
- There is no direct evidence in the repo that R specifically cannot support these checks.

#### `test_import_in_lua_scripting`

This is a cross-language integration scenario.

Conclusion:

- It is not specific to R behavior itself.
- Its absence in the generic R file looks like another case where the R generic suite remained narrower than the Python 3 suite.

## Cross-File Comparison

### Method-count summary

- `basic.py`: `r=12`, `python3=11`, `only_r=1`, `only_python3=0`
- `combinations.py`: `r=28`, `python3=32`, `only_r=3`, `only_python3=7`
- `dynamic_input.py`: `r=9`, `python3=19`, `only_r=1`, `only_python3=11`
- `dynamic_output.py`: `r=19`, `python3=27`, `only_r=2`, `only_python3=10`
- `emit.py`: `r=18`, `python3=36`, `only_r=0`, `only_python3=18`
- `export_alias.py`: `r=7`, `python3=8`, `only_r=1`, `only_python3=2`
- `generic_types.py`: `r=16`, `python3=17`, `only_r=2`, `only_python3=3`
- `get_connection.py`: `r=2`, `python3=3`, `only_r=0`, `only_python3=1`
- `import_alias.py`: `r=10`, `python3=16`, `only_r=0`, `only_python3=6`
- `metadata.py`: `r=27`, `python3=26`, `only_r=2`, `only_python3=1`
- `numeric_functions.py`: `r=8`, `python3=9`, `only_r=2`, `only_python3=3`
- `pathological_functions.py`: `r=1`, `python3=1`, `only_r=0`, `only_python3=0`
- `performance.py`: `r=3`, `python3=4`, `only_r=1`, `only_python3=2`
- `unicode.py`: `r=3`, `python3=5`, `only_r=1`, `only_python3=3`
- `vectorsize.py`: `r=7`, `python3=3`, `only_r=5`, `only_python3=1`

## File-by-File Findings

### `basic.py`

R has one extra method name:

- `test_set_with_empty_input`

This is effectively redundant with another R test that already covers the same behavior:

- `test_set_returns_has_empty_input_no_group_by`

Conclusion:

- This is not a meaningful Python 3 coverage hole.

### `combinations.py`

There are method-name differences in both directions.

However, several of these are naming or organization differences rather than true coverage gaps. In particular, the R 3-ary tests correspond to Python 3 3-ary coverage.

Python 3 additionally contains `n`-ary stress-style tests.

Conclusion:

- Mixed case: some renamed equivalents, plus extra Python 3 stress coverage.
- Not a strong sign of missing R capability.

### `dynamic_input.py`

Python 3 contains substantially more tests, including:

- exception-path checks
- type-specific behavior checks
- optimizer-related coverage

The R generic file mainly covers basic dynamic-input behavior and metadata access.

Conclusion:

- No explicit R limitation was found.
- This looks like a reduced-scope R port.

### `dynamic_output.py`

Python 3 contains broader coverage for:

- metadata correctness
- insert target column ordering
- syntax edge cases
- empty default output definitions

R already supports default output column handling and dynamic return behavior in the generic suite.

Conclusion:

- The missing R tests look unported rather than unsupported.

### `emit.py`

Python 3 has much broader data-type and null-handling coverage.

Conclusion:

- This is a coverage-depth difference.
- No explicit R blocker was found.

### `export_alias.py`

Python 3 adds tests for:

- lowercase column-name handling
- explicit column selection

R includes:

- `test_export_use_column_names`

Conclusion:

- The two files emphasize different export-spec cases.
- This looks like uneven porting, not a documented capability gap.

### `generic_types.py`

This file contains one of the clearest documented R-specific reasons for missing parity.

Python 3 includes limit tests such as:

- `test_echo_integer_limits`
- `test_echo_decimal_36_0_limits`
- `test_echo_decimal_36_36_limits`

These are annotated with:

- `@udf.TestCase.expectedFailureIfLang('r')`
- docstring `DWA-13784 (R)`

Conclusion:

- These missing R-side tests are supported by an explicit known R issue.

### `get_connection.py`

R covers the basic positive and negative path:

- existing connection
- missing connection

Python 3 adds broader regression and access-control coverage.

Conclusion:

- The generic R port kept only the core functionality tests.
- Missing Python 3 tests are broader regression coverage, not proof of an R feature gap.

### `metadata.py`

There is minor asymmetry in both directions.

R includes some extra count-oriented tests, while Python 3 includes an `output_columns`-style check.

Conclusion:

- This looks more like test-organization drift than a real functional mismatch.

### `numeric_functions.py`

R and Python 3 structure the tests differently.

R includes more direct arithmetic-oriented checks such as:

- `add_functions`
- `digit_split`

Python 3 includes wrapper-style query usage such as:

- `select`
- `select_into`
- `subselect`

Conclusion:

- Coverage shape differs more than capability.

### `performance.py`

R has simpler word-count style coverage.
Python 3 adds more scenarios.

Conclusion:

- Different depth of performance coverage, not a documented R limitation.

### `unicode.py`

Python 3 contains broader Unicode conformance checks.

One Python 3 test includes an explicit R-related note:

- `DWA-13782 (R)`

Conclusion:

- At least part of the missing R Unicode coverage is tied to a known R issue.
- The rest still looks like narrower R-suite coverage.

### `vectorsize.py`

This file is asymmetric in both directions.

Evidence from Python 3:

- It uses parameterized stress tests.
- It defines a language limit table with `r: 3000`.

Evidence from R:

- It includes explicit fixed-size cases such as 100, 1000, and 3000.

Conclusion:

- The absence of larger Python 3-style size tests on the R side is consistent with a lower R threshold.
- This appears to be a practical performance / scale limit rather than just accidental omission.

### `pathological_functions.py`

Parity is effectively fine here.

## Overall Conclusion

For `import_alias.py`, nothing from the generic R file is missing in Python 3. The Python 3 file simply contains additional tests.

Across the full generic suite, most mismatches are best explained by this pattern:

1. The generic R suite is a newer, narrower port.
2. Python 3 retains broader regression, integration, and stress coverage.
3. Only a subset of gaps have explicit documented R-specific reasons.

The clearest documented R-specific reasons found were:

- `generic_types.py`: `DWA-13784 (R)`
- `unicode.py`: `DWA-13782 (R)`
- `vectorsize.py`: practical limit evidence via the language-specific size threshold used by the Python 3 suite

Everything else is more consistent with incomplete parity than with proven R-language inability.

## Candidate Python3 Tests Likely Addable to R

Note: This list intentionally excludes tests with explicit R-related limitations or known non-applicability (for example: `generic_types` limit tests marked `DWA-13784 (R)`, `unicode` part3 with `DWA-13782 (R)`, `vectorsize_5000`, and the skipped view-definition case in `import_alias`).

- `combinations.py`: `test_n_scalar_emits`, `test_scalar_emits_scalar_emits`, `test_scalar_emits_set_returns_inline`, `test_set_returns_n_scalar_emits`, `test_set_returns_scalar_emits_scalar_emits`, `test_set_returns_scalar_returns_scalar_emits`, `test_set_returns_set_emits_scalar_emits`
- `dynamic_input.py`: `test_basic_scalar_emit`, `test_basic_scalar_return_constants`, `test_basic_set_emit_one_group`, `test_basic_set_return_one_group`, `test_exception_empty_set_emits`, `test_exception_empty_set_returns`, `test_exception_wrong_arg`, `test_exception_wrong_operation`, `test_mapreduce_optimization`, `test_type_specific_add_number`, `test_type_specific_add_string`
- `dynamic_output.py`: `test_copy_relation`, `test_create_script_syntax_var`, `test_empty_string_error`, `test_error_built_in_scalar_not_supported`, `test_error_built_in_set_not_supported`, `test_error_empty_emit_2`, `test_error_non_var_emit_2`, `test_insert_metadata_correctness`, `test_insert_target_columns_change_order`, `test_metadata_correctness`
- `emit.py`: `test_char_ascii`, `test_char_null`, `test_char_utf8`, `test_date_null`, `test_dec_128bit_with_scale`, `test_dec_32bit_with_scale`, `test_dec_64bit_with_scale`, `test_geo_null`, `test_geometry`, `test_int128_null`, `test_int32_null`, `test_int64_null`, `test_interval_ds`, `test_interval_ym`, `test_intervalds_null`, `test_intervalym_null`, `test_timestamp_null`, `test_timestamp_with_timezone`
- `export_alias.py`: `test_export_use_column_name_lower_case`, `test_export_use_column_selection`
- `get_connection.py`: `test_get_connection`
- `import_alias.py`: `test_import_in_lua_scripting`, `test_import_use_all_subselect`, `test_import_use_connection_fooconn_fails_for_user_foo`, `test_prepared_statement_conn`, `test_prepared_statement_params`
- `metadata.py`: `test_output_columns`
- `numeric_functions.py`: `test_select`, `test_select_into`, `test_subselect`
- `performance.py`: `test_frequency_analysis`, `test_word_count`
- `unicode.py`: `test_unicode`, `test_unicode_upper_is_subset_of_Unicode520_part2`

## Final Clarification

It is not fully accurate to say all Python3-only tests were simply missed in R.

The best-supported interpretation is:

1. Many tests were likely not ported yet to the generic R suite.
2. Some tests were intentionally or practically excluded due to known R-specific issues or limits.
3. Some are broader integration/regression scenarios and were probably outside the initial generic R port scope.

So the gap is mostly unported coverage, but not universally accidental omission.