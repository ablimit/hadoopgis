#! /bin/bash

usage(){
    echo -e "plotall.sh  [ pais | osm ]\n \
    --name \t dataset name to process \n \
    --help \t show this information.
    "
    exit 1
}

name=""
size=""

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
	-s | --size)
	    size=$2     # You might want to check if you really got FILE
	    shift 2
	    ;;
	--size=*)
	    size=${1#*=}        # Delete everything up till "="
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


if [ ! "$name" ] || [ ! "$size" ]; then
  echo "ERROR: missing option. See --help" >&2
  echo "Name: -- ${name} "
  echo "Size: -- ${size} "
  exit 1
fi

  cp template.${name}.plt draw.plt
  perl -p -i -e "s/_chartname_/${name}.medium.${size}.eps/g" draw.plt
  perl -p -i -e "s/_dataset_/${name}.medium.${size}.dat/g" draw.plt
  perl -p -i -e "s/_keyposition_/left top/g" draw.plt
  gnuplot draw.plt

#if [ ! "$name" == "pais" ] ; then
#  echo "Processing OpenStreetMap."
# python genPlotData.py "${metric}" < ${name}.eval.csv > pltdata.dat
#  cp template.pais.plt draw.plt
#  perl -p -i -e "s/_chartname_/${name}.medium.${size}.eps/g" draw.plt
#  perl -p -i -e "s/_dataset_/${name}.medium.${size}.dat/g" draw.plt
#  perl -p -i -e "s/_keyposition_/left top/g" draw.plt
#  gnuplot draw.plt
#fi

