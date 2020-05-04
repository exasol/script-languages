#!/bin/bash

set -e
set -u
set -o pipefail

apt list | grep installed | cut -f 1,2 -d " " | sed "s#/.* #|#g" | sort -f -d 
