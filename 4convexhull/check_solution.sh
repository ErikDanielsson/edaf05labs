#!/bin/bash

for x in `ls data|sort -n`
do
	for y in data/$x/*.in
	do
		pre=${y%.in}

		out=$pre.out

		correct=$pre.correct

		$* < $y > $out

		if diff -w $out $correct
		then 
			echo passed $y
		else
			echo failed for $y
			exit 1
		fi
	done
done
