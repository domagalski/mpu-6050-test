#!/bin/bash

set -o errexit
set -o pipefail

if [ "$1" == "4b" ]; then
  RUSTC_TARGET=armv7-unknown-linux-gnueabihf
elif [ "$1" == "zero" ]; then
  RUSTC_TARGET=arm-unknown-linux-gnueabi
else
  echo "unknown target: $1"
  exit 1
fi

cross build --release --target $RUSTC_TARGET
