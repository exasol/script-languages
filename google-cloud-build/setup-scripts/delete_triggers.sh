#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

function update(){
	echo "deleting" "$I"
	trigger_id=$(yq -r .trigger_id < "$env_flavor_config_path")
	"$SCRIPT_DIR/delete_build_trigger.sh" "$trigger_id"
}

function main(){
	triggers=triggers

#Ignore shellcheck rule, alternatives recommended by shellcheck are worse
#shellcheck disable=SC2044
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
