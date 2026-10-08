#!/usr/bin/env bash
set -eu
cd /workspace/CAE
export PIP_CACHE_DIR=/tmp/cae-pip-cache
export MPLCONFIGDIR=/tmp/cae-matplotlib
export XDG_CACHE_HOME=/tmp/cae-cache
if [ ! -x .venv/bin/python ]; then
    python -m venv --system-site-packages .venv
fi
.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q
