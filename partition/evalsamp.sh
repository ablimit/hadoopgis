#! /bin/bash

hid=$(hostname)
opath=/home/aaji/proj/hadoopgis/partition/sampling/stateval
data=/scratch/ablimit/data/partition/samp/osm

for algo in fg bsp hc slc bos str
do
  if [ ! -e sampling/parr/eval.${algo} ] ;
  then
    touch sampling/parr/eval.${algo} ;
    for f in 01 05 10 15 20 25 
    do
      for k in 864 4322 8644 17288 43220 86441 172882 432206 864412 4322062
      do
        if [ -e ${data}/${algo}/f${f}/c${k}/osm.part ]; then
          echo "${algo},${f},${k}"
          echo -n "${algo},${f},${k}," >> ${opath}/osm.eval.csv.${hid}
          python evalpartition.py 86441255 < ${data}/${algo}/f${f}/c${k}/osm.part >> ${opath}/osm.eval.csv.${hid}
          echo "------------------------------------"
        fi
      done
    done
  fi
done

