#! /bin/bash

usage(){
  echo -e "plotall.sh  [ pais | osm | hilbert ]\n \
    --name \t dataset name to process \n \
    --help \t show this information.
  "
  exit 1
}

name=""
filter=""

while :
do
  case $1 in
    -h | --help | -\?)
      usage;
      #  Call your Help() or usage() function here.
      exit 0      # This is not an error, User asked help. Don't do "exit 1"
      ;;
    -n | --name)
      name=$2     # You might want to check if you really got FILE
      shift 2
      ;;
    --name=*)
      name=${1#*=}        # Delete everything up till "="
      shift
      ;;
    --) # End of all options
      shift
      break
      ;;
    -*)
      echo "WARN: Unknown option (ignored): $1" >&2
      shift
      ;;
    *)  # no more options. Stop while loop
      break
      ;;
  esac
done


if [ ! "$name" ] || [ "$name" == "pais" ]; then
  echo "ERROR: missing option. See --help" >&2
  exit 1
fi

# for metric in min max avg median count stddev

for algo in bsp slc bos # hc str
do
  > /tmp/mydata.csv

  for r in 0001 0005 001 0015 0020 0025 01 25 100
  do
    grep "${algo},${r}," ${name}.eval.csv | cut -d, -f 2,3,4,5,6,7,8,9,10 >> /tmp/mydata.csv
  done
  for metric in stddev ratio
  do
    cat /tmp/mydata.csv | python genPlotData.py "${metric}" > pltdata.dat
    cp pltdata.dat indiv/${algo}.${metric}.dat
    cp template.${name}.plt draw.plt
    perl -p -i -e "s/_chartname_/algo\/${metric}.${name}.${algo}.eps/g" draw.plt
    perl -p -i -e "s/_dataset_/pltdata.dat/g" draw.plt
    if [ "$metric" == "stddev" ]; then 
      perl -p -i -e "s/_keyposition_/left top/g" draw.plt
    else 
      perl -p -i -e "s/_keyposition_/right top/g" draw.plt
    fi
    cp draw.plt indiv/${algo}.${metric}.plt
    gnuplot draw.plt
  done
  # exit 0;
done

