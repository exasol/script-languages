#!/bin/bash
bash find_error_codes.sh | cut -f 2,3 -d "-" | awk '$2 > a[$1] { a[$1] = $2 } END {for( k in a) print k "," a[k]}' FS=-
