#!/bin/sh
filename="$1"
if [ ! -f "$filename" ]; then
  touch $filename
fi
