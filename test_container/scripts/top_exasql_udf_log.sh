top -d 1 -c -b -w 10000 | grep -E "/usr/opt/.*/exasql |[^(mountjail)] exaudf/exaudfclient" | grep -E -v "grep" --
