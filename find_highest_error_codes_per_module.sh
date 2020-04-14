#!/bin/bash
bash find_error_codes.sh \
  | cut -f 3,4 \
  | awk '$2 > a[$1] { a[$1] = $2 } END {for( k in a) print k "\t" a[k]}'
