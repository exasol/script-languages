set -e
INPUT=$1
OUTPUT=$2
PYTHON_PREFIX=$3
PYTHON_VERSION=$4
PYTHON_SYSPATH=$5
echo "import sys, os" > extension
if ["PYTHON_VERSION" == "2*"]
then
    CURRENT_SYSPATH=$($PYTHON_PREFIX/bin/$PYTHON_VERSION -c 'import sys; import site; print sys.path')
else
    CURRENT_SYSPATH=$($PYTHON_PREFIX/bin/$PYTHON_VERSION -c 'import sys; import site; print(sys.path)')
fi
echo "PYTHON_CURRENT_SYSPATH=$CURRENT_SYSPATH"

echo "sys.path.extend($CURRENT_SYSPATH)" >> extension
if [ ! "X$PYTHON_SYSPATH" = "X" ]; then
    echo "PYTHON_SYSPATH=$PYTHON_SYSPATH"
    echo "sys.path.extend($PYTHON_SYSPATH)" >> extension
fi
cat extension "$INPUT" >> "$OUTPUT"