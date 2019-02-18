mkfifo /tmp/fff
INPUT=exasol_emulator/input.csv
SCRIPT=exasol_emulator/script.java
LANGUAGE=java
python exasol_emulator/exasolution.py unix:/tmp/fff $INPUT exasol_emulator/output.csv $SCRIPT &
$1/exaudfclient ipc:///tmp/fff lang=$LANGUAGE
