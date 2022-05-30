micromamba list | tail +5 | sed -E "s/\s+/|/g" | cut -f2,3,5 -d "|" | sort -f -d

