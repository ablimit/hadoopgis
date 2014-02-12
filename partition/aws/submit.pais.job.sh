#!/usr/bin/env bash

usage(){
  echo -e "submitall.sh  --job [job flow id]\n \
    --job \t Amazon EMR Job Flow ID to submit steps. \n \
    --alg \t partition algorithm to test. \n \
    --log \t output directory for the results. \n \
    --help \t show this information.
  "
  exit 1
}

jobid=""
algo=""
log=""

while :
do
  case $1 in
    -h | --help | -\?)
      usage;
      #  Call your Help() or usage() function here.
      exit 0      # This is not an error, User asked help. Don't do "exit 1"
      ;;
    -j | --job)
      jobid=$2
      shift 2
      ;;
    --job=*)
      jobid=${1#*=}
      shift
      ;;
    -a | --alg)
      algo=$2 
      shift 2
      ;;
    --alg=*)
      algo=${1#*=}        # Delete everything up till "="
      shift
      ;;
    -l | --log)
      log=$2
      shift 2
      ;;
    --log=*)
      log=${1#*=}
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
if [ "${algo}" != "st" ] && [ "${algo}" != "rt" ] && [ "${algo}" != "rp" ] && [ "${algo}" != "fg" ] && [ "${algo}" != "hc" ];
then
  echo "Parameter [${algo}] is NOT recognized. Alternatives are [ st | rp | rt | fg | hc ]"
  exit 1;
fi

if [ ! "$log" ] ; then
  echo "ERROR: log directoray name is missing. See --help" >&2
  exit 1
fi

echo "Job ID [${jobid}]"
# elastic-mapreduce -j ${jobid} --wait-for-steps

# R+ Tree  | R* Tree  | Fixed Grid | Strip

for c in 20 100 200 400 1000 2000 4000 10000 20000 100000
do
  s3input="s3://aaji/data/partitions/pais/${algo}/c${c}"
  echo -n "job param: [${c}] -- "
  elastic-mapreduce --jobflow ${jobid} --stream --step-name "pais.${algo}.${c}" --step-action CONTINUE --mapper 's3://aaji/scratch/awsjoin/tagmapper.py pais.geom.1.tsv pais.geom.2.tsv' --reducer "s3://aaji/scratch/deps/bins/resque st_intersects 1 1" --input ${s3input} --output s3://aaji/scratch/pout/${log}/pais/${algo}c${c} --jobconf mapred.reduce.tasks=1000

  sleep 15 ;
done
