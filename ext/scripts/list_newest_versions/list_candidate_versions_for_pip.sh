#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
package_list_file=$1
perl "$SCRIPT_DIR/extract_columns_from_package_lisl.pl" --file "$package_list_file" --columns 0,1 | xargs -n 1 -I{} python3 "$SCRIPT_DIR/get_versions.py" "{}"
