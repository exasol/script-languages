#!/bin/bash

set -euo pipefail

pip_version=$1
curl -o get-pip.py https://bootstrap.pypa.io/get-pip.py
python3.12 get-pip.py "$pip_version" --break-system-packages
rm get-pip.py
rm -rf "$(python3.10 -m pip cache dir)"
