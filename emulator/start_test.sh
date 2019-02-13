mkfifo /tmp/fff
INPUT=exasol_emulator/input.csv
SCRIPT=exasol_emulator/script.java
LANGUAGE=java
LD_LIBRARY_PATH=/usr/lib/jvm/java-9-openjdk-amd64/lib:/usr/lib/jvm/java-9-openjdk-amd64/lib/amd64:/usr/lib/jvm/java-9-openjdk-amd64/lib/amd64/server:/usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/jli
python exasol_emulator/exasolution.py unix:/tmp/fff $INPUT exasol_emulator/output.csv $SCRIPT &
/exaudf/exaudfclient ipc:///tmp/fff lang=$LANGUAGE
