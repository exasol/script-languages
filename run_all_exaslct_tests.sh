#!/bin/bash 
   
COMMAND_LINE_ARGS=$* 
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" 
 
source pipenv_utils.sh 
 
discover_pipenv 
init_pipenv "$PIPENV_BIN" 

PYTHONPATH=. $PIPENV run python3 -m unittest discover exaslct_src/test
