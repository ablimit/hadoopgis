#! /bin/bash

#if [ ! $# == 1 ]; then
#    echo "Usage: $0 [replication factor]"
#    exit 0
#fi

localfileloc=/data2/ablimit/Data/spatialdata/pais/boundarytest
hdfsloc=/user/aaji/paisboundary
# factor=${1}
for factor in 3 4 5 6 7 8 9
do
    echo "uploading PAIS data with replication factor ${factor}.."

    sudo -u hdfs hdfs dfs -copyFromLocal ${localfileloc}/rep${factor} ${hdfsloc}/rep${factor}
done
echo "done."
