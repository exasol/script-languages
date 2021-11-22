#!/bin/bash

set -euo pipefail

curl -o get-pip.py https://bootstrap.pypa.io/get-pip.py
python3.7 get-pip.py "pip < 21"
rm get-pip.py
rm -rf $(python3.7 -m pip cache dir)
