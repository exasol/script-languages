SOURCE_HOST=$1
SOURCE_PORT=$2
DESTINATION_HOST=$3
DESTINATION_PORT=$4
trap "exit" INT
while true
do
  echo "Start Connect Connect Server"
  socat -d -d -d TCP4:$SOURCE_HOST:$SOURCE_PORT,reuseaddr,forever,fork TCP4:$DESTINATION_HOST:$DESTINATION_PORT,reuseaddr,forever
done
