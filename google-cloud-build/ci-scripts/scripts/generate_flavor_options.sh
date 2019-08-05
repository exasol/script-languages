function generate_flavor_options(){
  FLAVORS="$1"
  echo "FLAVORS: $FLAVORS"
  FLAVOR_OPTIONS=""
  for FLAVOR in $FLAVORS
  do
    FLAVOR_OPTIONS="$FLAVOR_OPTIONS --flavor-path 'flavors/$FLAVOR'"
  done
  if [ -z "$FLAVOR_OPTIONS=" ]
  then
    echo "No flavors specified"
    exit 1
  fi
}
