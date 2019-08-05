#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

function generate_run_json(){
  cat "$env_file" $* > data.yaml
  echo "commitSha: $commitSha" >> data.yaml
  jinja2 $triggers/run.json data.yaml > run.json
  rm data.yaml
}

function run(){
  echo "running" $I
  TRIGGER_ID=$(cat $env_flavor_config_path | yq -r .trigger_id)
  generate_run_json
  cat run.json
  $setup_scripts/run_build_trigger.sh $TRIGGER_ID run.json
  rm run.json
}

function main(){
  commitSha=$2
  filter=$1
	setup_scripts=setup-scripts
	triggers=triggers
	env_file=".env/env.yaml"
	for I in $(find $triggers/flavor-config -name '*.yaml' | grep $filter)
	do
		env_flavor_config_path=".env/$I"
		if [ -f "$env_flavor_config_path" ]
		then
			run
		fi
	done
}

main $*
