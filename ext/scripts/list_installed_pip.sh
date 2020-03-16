$1 -m pip list --format columns | tail -n +3 | sed "s/  */|/" | sort -f -d
