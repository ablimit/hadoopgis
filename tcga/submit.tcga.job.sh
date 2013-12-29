#!/usr/bin/env bash

usage(){
  echo -e "submitall.sh  --job [job flow id]\n \
    --job \t Amazon EMR Job Flow ID to submit steps. \n \
    --help \t show this information.
  "
  exit 1
}

jobid=""
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

if [ ! "$log" ] ; then
  echo "ERROR: log directoray name is missing. See --help" >&2
  exit 1
fi

echo "Job ID [${jobid}]"

action=CONTINUE  # TERMINATE_JOB_FLOW 

elastic-mapreduce --jobflow ${jobid} --stream --step-name "tcga.st_intersect.full" --step-action ${action} --mapper 's3://aaji/scratch/awsjoin/tcgamapper.py' --reducer "s3://aaji/scratch/deps/bins/resque st_intersects 1 1" --input s3://aaji/data/tcga --output s3://aaji/scratch/pout/${log}/tcga_2 --jobconf mapred.reduce.tasks=2000

