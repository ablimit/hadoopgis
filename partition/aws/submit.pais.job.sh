#! /bin/bash

usage(){
    echo -e "submitall.sh  --job [job flow id]\n \
    --job \t Amazon EMR Job Flow ID to submit steps \n \
    --help \t show this information.
    "
    exit 1
}

jobid=""
algo=""

while :
do
    case $1 in
	-h | --help | -\?)
	    usage;
	    #  Call your Help() or usage() function here.
	    exit 0      # This is not an error, User asked help. Don't do "exit 1"
	    ;;
	-j | --job)
	    jobid=$2     # You might want to check if you really got FILE
	    shift 2
	    ;;
	--job=*)
	    jobid=${1#*=}        # Delete everything up till "="
	    shift
	    ;;
	-a | --alg)
	    algo=$2     # You might want to check if you really got FILE
	    shift 2
	    ;;
	--alg=*)
	    algo=${1#*=}        # Delete everything up till "="
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
if [ ! "$jobid" ] ; then
    echo "ERROR: job flow id is missing. See --help" >&2
    exit 1
fi

# argument checking 
if [ "${algo}" != "st" ] && [ "${algo}" != "rt" ] && [ "${algo}" != "rp" ] && [ "${algo}" != "fg" ] ;
then
  echo "Parameter [${algo}] is NOT recognized. Alternatives are [ st | rp | rt | fg ]"
  exit 1;
fi

# jobid="j-2UHSB3ZIF85YA"
echo "Job ID [${jobid}]"

# /usr/local/emrcli/elastic-mapreduce -j ${jobid} --wait-for-steps

# R+ Tree  | R* Tree  | Fixed Grid
if [ "${algo}" != "st" ] ;
then
  for c in 864 4322 8644 17288 43220 86441 # 172882 432206 864412 4322062
  do
    # echo "[${c}] [${algo}]"
    elastic-mapreduce --jobflow ${jobid} --stream --step-name "${algo}.${c}" --step-action CONTINUE --mapper 's3://aaji/scratch/awsjoin/tagmapper.py osm.geom.dat osm.geom.2.dat' --reducer "s3://aaji/scratch/deps/bins/resque st_intersects 1 1" --input "s3://aaji/data/partitions/osm/${algo}/c${c}" --output s3://aaji/scratch/pout/dec18/${algo}c${c} --jobconf mapred.reduce.tasks=1000

    sleep 300 ;
  done

else
  # strip
  for c in 864 4322 8644 17288 43220 86441 # 172882 432206 864412 4322062
  do
    echo "[${c}] [${algo}]"
    elastic-mapreduce --jobflow ${jobid} --stream --step-name "${algo}.${c}" --step-action CONTINUE --mapper 's3://aaji/scratch/awsjoin/tagmapper.py osm.geom.dat osm.geom.2.dat' --reducer "s3://aaji/scratch/deps/bins/resque st_intersects 1 1" --input "s3://aaji/data/partitions/osm/${algo}/x/c${c}" --output s3://aaji/scratch/pout/dec18/${algo}c${c} --jobconf mapred.reduce.tasks=1000

    sleep 300 ;

  done
fi

