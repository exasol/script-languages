#!/bin/bash

set -euo pipefail

curl -o get-pip.py https://bootstrap.pypa.io/pip/2.7/get-pip.py
python2.7 get-pip.py "pip < 21"
rm get-pip.py
rm -rf $(python2.7 -m pip cache dir)