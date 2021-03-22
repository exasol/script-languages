#!/bin/bash
package_list_file=$1
perl extract_columns_from_package_lisl.pl --file "$package_list_file" --columns 0,1 | xargs -n 1 -I{} python get_versions.py "{}"
