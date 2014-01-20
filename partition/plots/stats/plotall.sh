#! /bin/bash

usage(){
    echo -e "plotall.sh  [ pais | osm | hilbert ]\n \
    --name \t dataset name to process \n \
    --help \t show this information.
    "
    exit 1
}

name=""

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


if [ ! "$name" ] ; then
    echo "ERROR: missing option. See --help" >&2
    exit 1
fi

# for metric in min max avg median count stddev
#for metric in stddev ratio
#do
#	python genPlotData.py "${metric}" < ${name}.eval.csv | grep -v "qt" > ${name}.${metric}.dat
#	cp template.${name}.plt ${name}.${metric}.plt
#	perl -p -i -e "s/_chartname_/${name}.${metric}.eps/g" ${name}.${metric}.plt
#	perl -p -i -e "s/_dataset_/${name}.${metric}.dat/g" ${name}.${metric}.plt
#	perl -p -i -e "s/_keyposition_/left top/g" ${name}.${metric}.plt
#	#gnuplot draw.plt
#done
for metric in stddev ratio
do
	grep -v qt ${name}.eval.csv | python genPlotData.py "${metric}" > pltdata.dat
	cp template.${name}.plt draw.plt
	perl -p -i -e "s/_chartname_/${metric}${name}.eps/g" draw.plt
	perl -p -i -e "s/_dataset_/pltdata.dat/g" draw.plt
	perl -p -i -e "s/_keyposition_/right top/g" draw.plt
	gnuplot draw.plt
done

