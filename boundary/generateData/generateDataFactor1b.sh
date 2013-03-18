#! /bin/bash

if [ ! $# == 1 ]; then
    echo "Usage: [factor] "
    exit 0
fi

FACTOR=$1
NEWPATH=/data2/hoang/mjoin1c${FACTOR}
OLDPATH2=/data2/ablimit/algo2
HDFSPATH2=/user/hvo8/mjoin/algo2c1${FACTOR}

mkdir $NEWPATH
chmod 777 $NEWPATH

sudo -u hdfs hdfs dfs -rm -r ${HDFSPATH2}
sudo -u hdfs hdfs dfs -mkdir ${HDFSPATH2}

for filename2 in astroII.1.markup.ablet.2  normal.3.markup.ablet.2 astroII.2.markup.ablet.2  oligoastroII.1.markup.ablet.2 gbm0.1.markup.ablet.2 oligoastroII.2.markup.ablet.2 gbm0.2.markup.ablet.2 oligoastroIII.1.markup.ablet.2 gbm1.1.markup.ablet.2 oligoastroIII.2.markup.ablet.2 gbm1.2.markup.ablet.2 oligoII.1.markup.ablet.2 gbm2.1.markup.ablet.2 oligoII.2.markup.ablet.2 gbm2.2.markup.ablet.2 oligoIII.1.markup.ablet.2 normal.2.markup.ablet.2 oligoIII.2.markup.ablet.2
do
   rm -rf ${NEWPATH}/${filename2}
   ./regen 2048 ${FACTOR} $filename2."stat".${FACTOR} < ${OLDPATH2}/${filename2} > ${NEWPATH}/${filename2}
   sudo -u hdfs hdfs dfs -copyFromLocal ${NEWPATH}/${filename2} ${HDFSPATH2}/
done

