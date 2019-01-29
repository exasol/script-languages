set -e
INPUT=$1
OUTPUT=$2
PYTHON_PREFIX=$3
PYTHON_SYSPATH=$4
echo "import sys, os" > extension
CURRENT_SYSPATH_=$($PYTHON_PREFIX/bin/python -c 'import sys; import site; print sys.path')
echo "sys.path.extend($CURRENT_SYSPATH)" >> extension
if [ ! "X$PYTHON_SYSPATH" = "X" ]; then
    echo "sys.path.extend($PYTHON_SYSPATH)" >> extension
fi
cat extension "$INPUT" >> "$OUTPUT"