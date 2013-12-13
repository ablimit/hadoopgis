#! /bin/bash

usage(){
    echo -e "spjoin-amazonemr.sh  [options]\n \
    --inputa \t Amazon S3 directory path for join input data A (table a )\n \
    --inputb \t Amazon S3 directory path for join input data B (table b )\n \
    --output \t Amazon S3 directory path for output data \n \
    --predicate \t join predicate [contains | intersects | touches | crosses | within | dwithin] \n \
    --geoma \t index of the geometry field of table A\n \
    --geomb \t index of the geometry field \n \
    --worker \t number of reduce tasks to utlize \n \
    --uid \t index of the uid field \n \
    --temppath \t Amazon S3 to store the temporary tiler result (directory) (no trailing slash)
    --help \t show this information.
    "
    exit 1
}

north=90
south=-90
east=180
west=-180

# Reset all variables that might be set
# path to the elastic-mapreduce CLI
#ldpath=/usr/local/lib:/usr/lib:$LD_LIBRARY_PATH

# parameter for number of reducers
reducecount=20
# Number of core nodes to be requested
num_instances=4

temps3path=s3://cciemorypublic/libhadoopgis/tempspace
inputdira=s3://cciemorypublic/libhadoopgis/sampledata/osm.1.dat
inputdirb=s3://cciemorypublic/libhadoopgis/sampledata/osm.2.dat
outputdir=s3://cciemorypublic/libhadoopgis/finaloutput/
predicate=intersects
gidxa=5
gidxb=5
jobid=""
outputtiler1=""
outputtiler2=""

while :
do
    case $1 in
	-h | --help | -\?)
	    usage;
	    #  Call your Help() or usage() function here.
	    exit 0      # This is not an error, User asked help. Don't do "exit 1"
	    ;;
	-n | --worker)
	    reducecount=$2     # You might want to check if you really got FILE
	    shift 2
	    ;;
	--worker=*)
	    reducecount=${1#*=}        # Delete everything up till "="
	    shift
	    ;;
	-p | --predicate)
	    predicate=$2     # You might want to check if you really got FILE
	    shift 2
	    ;;
	--predicate=*)
	    predicate=${1#*=}        # Delete everything up till "="
	    shift
	    ;;
	-t | --temppath)
	    temps3path=$2
	    shift 2
	    ;;
	--temppath=*)
	    temps3path=${1#*=}        # Delete everything up till "="
	    shift
	    ;;
	-i | --inputa)
	    inputdira=$2     # You might want to check if you really got FILE
	    shift 2
	    ;;
	--inputa=*)
	    inputdira=${1#*=}        # Delete everything up till "="
	    shift
	    ;;
	-j | --inputb)
	    inputdirb=$2     # You might want to check if you really got FILE
	    shift 2
	    ;;
	--inputb=*)
	    inputdirb=${1#*=}        # Delete everything up till "="
	    shift
	    ;;
	-o | --output)
	    outputdir=$2     # You might want to check if you really got FILE
	    shift 2
	    ;;
	--output=*)
	    outputdir=${1#*=}        # Delete everything up till "="
	    shift
	    ;;
	-g | --geoma)
	    gidxa=$2     # You might want to check if you really got FILE
	    shift 2
	    ;;
	--geoma=*)
	gidxa=${1#*=}        # Delete everything up till "="
	    shift
	    ;;
	-h | --geomb)
	    gidxb=$2     # You might want to check if you really got FILE
	    shift 2
	    ;;
	--geomb=*)
	gidxb=${1#*=}        # Delete everything up till "="
	    shift
	    ;;
	-x | --xsplit)
	    xsplit=$2     # You might want to check if you really got FILE
	    shift 2
	    ;;
	--xsplit=*)
	    xsplit=${1#*=}        # Delete everything up till "="
	    shift
	    ;;
	-y | --ysplit)
	    ysplit=$2     # You might want to check if you really got FILE
	    shift 2
	    ;;
	--ysplit=*)
	    ysplit=${1#*=}        # Delete everything up till "="
	    shift
	    ;;
	-u | --uid)
	    uidx=$2     # You might want to check if you really got FILE
	    shift 2
	    ;;
	--uid=*)
	    uidx=${1#*=}        # Delete everything up till "="
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
if [ ! "$predicate" ] ; then
    echo "ERROR: join predicate is missing. See --help" >&2
    exit 1
fi

if [ ! "$gidxa" ] || [ ! "$gidxb" ] ; then
    echo "ERROR: geometry field index is missing. See --help" >&2
    exit 1
fi


if [ ! "$inputdira" ] || [ ! "$inputdirb" ] || [ ! "$outputdir" ]; then
    echo "ERROR: missing option. See --help" >&2
    exit 1
fi

outputtiler1=${temps3path}/outputtiler1
outputtiler2=${temps3path}/outputtiler2

reducecmd="resque st_${predicate} ${gidxa} ${gidxb}"

#reducecmd='s3://cciemorypublic/libhadoopgis/program/hgtiler -w '${west}' -s '${south}' -n '${north}' -e ${east} -x ${xsplit} -y ${ysplit} -u ${uidx} -g ${gidxa}'

# Tiling first data file
ruby ./elastic-mapreduce --create --alive --stream --num-instances=10 --instance-type=m1.medium --master-instance-type=m1.medium --name "Joiner"  --mapper 's3://cciemorypublic/libhadoopgis/program/tagmapper.py ' --reducer ${reducecmd} --input ${input1} --output ${outputtiler1}  --jobconf mapred.reduce.tasks=${reducecount} --bootstrap-action "s3://cciemorypublic/libhadoopgis/bootstrap/bootcopygeosspatial.sh" | rev | cut -d " " -f 1 | rev > tmpjobid

read jobid < tmpjobid

# Tiling second data file (adding step)

# Wait for previous step to complete
#ruby ./elastic-mapreduce -j ${jobid} --wait-for-steps

ruby ./elastic-mapreduce --jobflow ${jobid} --stream --mapper 's3://cciemorypublic/libhadoopgis/program/hgdeduplicater.py cat' --reducer "s3://cciemorypublic/libhadoopgis/program/hgtiler -w ${west} -s ${south} -n ${north} -e ${east} -x ${xsplit} -y ${ysplit} -u ${uidx} -g ${gidxb}" --input ${inputdirb} --output ${outputtiler2}/  --jobconf mapred.reduce.tasks=${reducecount}

reducecmd="s3://cciemorypublic/libhadoopgis/program/resque st_${predicate} ${gidxa} ${gidxb}"
# Add the join step
ruby ./elastic-mapreduce --jobflow ${jobid} --stream --args "-input ${outputtiler2}"  --mapper 's3://cciemorypublic/libhadoopgis/program/tagmapper.py outputtiler1 outputtiler2' --reducer ${reducecmd} --input ${outputtiler1} --output ${outputdir}  --jobconf mapred.reduce.tasks=${reducecount}


