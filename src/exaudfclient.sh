echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
echo "Changing to script directory  $SCRIPT_DIR"
cd $SCRIPT_DIR
export LIBEXAUDFLIB_PATH="$SCRIPT_DIR/libexaudflib_complete.so"
./exaudfclient_wrong_rpath $* 