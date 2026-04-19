#!/bin/bash
# Poor man's "watch"
# Depends on bash, date, stty & head.

if [ ${#@} -lt 2 ]; then
	echo "Usage: $(basename $0) <interval> <command>"
	exit 1
fi

t=$1
shift

line="Every $t: $*"
clear

while true; do
	# Calculate and draw top line with padding
	ts=$(date)
	dims=($(stty size))
	let padlen="${dims[1]} - (${#ts} + ${#line})"
	padding=$(printf "%${padlen}s")
	echo -e "\e[1m$line$padding$ts\e[0m\n"

	# Execute command, show top $height rows, then sleep
	let height=${dims[0]}-3
	$* | head -n $height
	sleep $t
	clear
done
