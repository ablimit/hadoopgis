#! /bin/bash


for task in containment aggregation #join
do

    echo "Task: ${task} "
    cd ${task}
    sh runMapReduce.sh
    cd -
done

