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

for algo in fg bsp hc slc bos str
do
  for metric in stddev ratio
  do
    grep "${algo}" ${name}.samp.stat.csv | cut -d, -f 2,3,4,5,6,7,8,9,10 | python genPlotData.py "${metric}" > pltdata.dat
    cp template.${name}.plt draw.plt
    perl -p -i -e "s/_chartname_/algo\/${metric}.${name}.${algo}.eps/g" draw.plt
    perl -p -i -e "s/_dataset_/pltdata.dat/g" draw.plt
    perl -p -i -e "s/_keyposition_/right top/g" draw.plt
    gnuplot draw.plt
    # exit 0;
  done
done

