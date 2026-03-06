#!/bin/bash
set -e
set -u

PYTHON_VERSION=3.14

input=$1
input_abs=$(realpath "$input")
dir=$(dirname "$input_abs")
output=codegen/requirements.txt

uv python install $PYTHON_VERSION
pushd "$dir"
cd ..
uv pip compile \
  --python-version $PYTHON_VERSION \
  codegen/pyproject.toml \
  -o "$output"
popd
