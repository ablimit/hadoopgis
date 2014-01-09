#! /bin/bash

usage(){
    echo -e "sorthilbert.sh  --corner [min|center|max]\n \
    --help \t show this information.
    "
    exit 1
}

infile=/data2/ablimit/Data/spatialdata/bakup/data/partition/osm/hc/osm.mbb.hilbert.dat
outpath=/data2/ablimit/Data/spatialdata/bakup/data/partition/osm/hc
corner=""

while :
do
    case $1 in
	-h | --help | -\?)
	    usage;
	    #  Call your Help() or usage() function here.
	    exit 0      # This is not an error, User asked help. Don't do "exit 1"
	    ;;
	-c | --corner)
	    corner=$2     # You might want to check if you really got FILE
	    shift 2
	    ;;
	--corner=*)
	    corner=${1#*=}        # Delete everything up till "="
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

# Suppose some options are required. Check that we got them.
if [ ! "$corner" ] ; then
    echo "ERROR: parameter CORNER is not specified. See --help" >&2
    exit 1
fi

pos=6

if [ ${corner} == "min" ];
then
  pos=6 
elif [ ${corner} == "center" ]
then 
  pos=7 
elif [ ${corner} == "max" ]
then 
  pos=8 
else
  echo "param corner is NOT correct."
fi

sort -T /dev/shm --numeric-sort --parallel=10 --key=${pos} ${infile} -o ${outpath}/osm.mbb.hc.sorted.${corner}.dat

echo "done."

