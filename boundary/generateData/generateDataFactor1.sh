#! /bin/bash

NEWPATH=/data2/hoang/mjoinc1

OLDPATH=/data2/ablimit/algo1
HDFSPATH=/user/hvo8/mjoin/algo1c1

mkdir $NEWPATH
chmod 777 $NEWPATH

for ((counter = 6; counter <= 6; counter++))
do
        sudo -u hdfs hdfs dfs -rm -r ${HDFSPATH}
        sudo -u hdfs hdfs dfs -mkdir ${HDFSPATH}

        for filename in astroII.1.markup.ablet.1 normal.3.markup.ablet.1 astroII.2.markup.ablet.1  oligoastroII.1.markup.ablet.1 gbm0.1.markup.ablet.1 oligoastroII.2.markup.ablet.1 gbm0.2.markup.ablet.1     oligoastroIII.1.markup.ablet.1 gbm1.1.markup.ablet.1 oligoastroIII.2.markup.ablet.1 gbm1.2.markup.ablet.1 oligoII.1.markup.ablet.1 gbm2.1.markup.ablet.1     oligoII.2.markup.ablet.1 gbm2.2.markup.ablet.1     oligoIII.1.markup.ablet.1 normal.2.markup.ablet.1   oligoIII.2.markup.ablet.1
        do
                rm -rf ${NEWPATH}/${filename}
                ./GenDup2 2048 ${counter} ${filename}."stat".${counter} < ${OLDPATH}/${filename} > ${NEWPATH}/${filename}
        sudo -u hdfs hdfs dfs -copyFromLocal ${NEWPATH}/${filename} ${HDFSPATH}/
        done
done
