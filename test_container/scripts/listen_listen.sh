INPUT_PORT=$1
OUTPUT_PORT=$2
trap "exit" INT
while true
do
  echo "Start Listen Listen Server"
  socat -d -d -d  TCP4-LISTEN:$INPUT_PORT,reuseaddr,forever,fork TCP4-LISTEN:$OUTPUT_PORT,reuseaddr,forever
done
