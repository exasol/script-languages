#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

function update(){
	echo "deleting" $I
	trigger_id=$(cat "$env_flavor_config_path" | yq -r .trigger_id)
	$SCRIPT_DIR/delete_build_trigger.sh "$trigger_id"
}

function main(){
	triggers=triggers
	for I in $(find $triggers/flavor-config -name '*.yaml')
	do
		env_flavor_config_path=".env/$I"
		if [ -f "$env_flavor_config_path" ]
		then
			update
		fi
	done
}

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
main
