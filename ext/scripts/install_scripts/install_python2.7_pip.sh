#!/bin/bash

set -euo pipefail

curl -o get-pip.py https://bootstrap.pypa.io/2.7/get-pip.py
python2.7 get-pip.py "pip < 21"
rm get-pip.py
