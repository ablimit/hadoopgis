#! /bin/bash

usage(){
  echo -e "benchmark.sh  --algo [ fg | rp | rt | st | all ]\n \
    --algo \t The partition algorithm to benchmark. \n \
    --help \t show this information.
  "
  exit 1
}

algo=""

while :
do
  case $1 in
    -h | --help | -\?)
      usage;
      #  Call your Help() or usage() function here.
      exit 0      # This is not an error, User asked help. Don't do "exit 1"
      ;;
    -a | --algo)
      algo=$2     # You might want to check if you really got FILE
      shift 2
      ;;
    --algo=*)
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
if [ ! "$algo" ] ; then
  echo "ERROR: test algorithm param is missing. See --help" >&2
  exit 1
fi

logid=$(date +%F-%k-%M)
logfile=query.${algo}.${logid}.log

echo "---$(date)---" >> ${logfile}
# exit 0;

udfsinbase=/user/aaji/data/partition/osm
hdfsoutbase=/user/aaji/outputs/osm
streamjar=/usr/share/hadoop/contrib/streaming/hadoop-examples-1.2.1.jar
HCMD="hadoop --config /etc/hadoop jar ${streamjar}"
BASEOPTS=" -verbose -cmdenv LD_LIBRARY_PATH=/home/aaji/softs/lib:$LD_LIBRARY_PATH"

for c in 864 4322 8644 17288 43220 86441 172882 432206 864412 4322062
do
  hdfsout=${hdfsoutbase}/${algo}/c${c}
  hdfsin=${hdfsinbase}/${algo}/c${c}

  hadoop fs -mkdir -p ${hdfsout}

  for reducecount in 20 40 60 80 100 120
  do
    STREAMOPTS="${BASEOPTS} -jobconf mapred.job.name=\"sj_osm_${algo}_${c}_${reducecount}\" -jobconf mapred.task.timeout=36000000"

    START=$(date +%s)
    # join step 
    ${HCMD} -D mapred.child.java.opts=-Xmx4096M -mapper "tagmapper.py osm.geom.1.tsv osm.geom.2.tsv"  -reducer "resque st_intersects 1 1" -file mapper -file reducer -input ${hdfsin} -output ${hdfsout} -numReduceTasks ${reducecount} ${STREAMOPTS}
    END=$(date +%s)
    DIFF=$(( $END - $START ))
    echo "join,c${c},${algo},${reducecount},${DIFF}" >> ${logfile}
    # deduplication step
    START=$(date +%s)
    ${HCMD} -mapper org.apache.hadoop.mapred.lib.IdentityMapper -reducer "/usr/bin/uniq" -input "${hdfsout}" -output "${hdfsoutbase}/temp" -numReduceTasks ${reducecount} -verbose 
    END=$(date +%s)
    DIFF=$(( $END - $START ))

    echo "uniq,c${c},${algo},${reducecount},${DIFF}" >> ${logfile}

    hadoop fs -rmr ${hdfsoutbase}/temp
  done
done

