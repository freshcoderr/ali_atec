#!/bin/bash
#echo "id	s1	s2	is_duplicate" > tmp
#cat tmp $1 > input_file
python predict.py $1 $2
