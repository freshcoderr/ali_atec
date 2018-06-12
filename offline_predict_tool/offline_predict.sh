#!/bin/bash

# $1 input
# id s1 s2 label
input=$1

# $2 output
# id predict
output=$2

# predict_output
# id predict label
awk -F'\t' 'BEGIN{OFS="\t"}NR==FNR{label[$1]=$4;} NR!=FNR && ($1 in label){print $1,$2,label[$1]}' ${input} ${output} > predict_output

TP=`cat predict_output | awk -F'\t' '$2==1 && $3==$2' | wc -l`
FP=`cat predict_output | awk -F'\t' '$2==1 && $3!=$2' | wc -l`
TN=`cat predict_output | awk -F'\t' '$2==0 && $3==$2' | wc -l`
FN=`cat predict_output | awk -F'\t' '$2==0 && $3!=$2' | wc -l`

precision=`echo "scale=4;${TP} / ((${TP} + ${FP}))" | bc`
recall=`echo "scale=4;${TP} / ((${TP}+${FN}))" | bc`
Acc=`echo "scale=4;((${TP}+${TN})) / ((${TP}+${FP}+${TN}+${FN}))" | bc`
F1=`echo "scale=4;2 * ${precision} * ${recall} / ((${precision}+${recall}))" | bc`

echo "F1-score:  "${F1}
echo "     Acc:  "${Acc}
