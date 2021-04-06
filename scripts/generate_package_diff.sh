#!/bin/bash
tmpfile1=$(mktemp)
tmpfile2=$(mktemp)
perl ../ext/scripts/list_newest_versions/extract_columns_from_package_lisl.pl --file "$1" --columns 0,1 > "$tmpfile1"
perl ../ext/scripts/list_newest_versions/extract_columns_from_package_lisl.pl --file "$1" --columns 0,1 > "$tmpfile2"
python generate_package_diff.py "$tmpfile1" "$tmpfile2"
