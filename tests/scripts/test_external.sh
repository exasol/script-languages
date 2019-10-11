IP="$1"
shift 1
PASSWORD="9UyIRRuYPYaxjeqb"
./exaslct run-db-test --external-exasol-db-host "$IP" --external-exasol-db-port 8563 --external-exasol-bucketfs-port 6583 --external-exasol-db-user sys --external-exasol-db-password "$PASSWORD" --external-exasol-bucketfs-write-password "$PASSWORD" --environment-type external_db --external-exasol-xmlrpc-host "$IP" --external-exasol-xmlrpc-password "$PASSWORD" $*
