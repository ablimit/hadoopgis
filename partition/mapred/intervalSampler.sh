#! /bin/bash

usage(){
  echo -e "intervalSampler.sh  [options]\n \
    -n --interval \t [ an integer value ] \n \
    --help \t show this information.
  "
  exit 1
}

n=""

while :
do
  case $1 in
    -h | --help | -\?)
      usage;
      #  Call your Help() or usage() function here.
      exit 0      # This is not an error, User asked help. Don't do "exit 1"
      ;;
    -n | --interval)
      n=$2     # You might want to check if you really got FILE
      shift 2
      ;;
    --interval=*)
      n=${1#*=}        # Delete everything up till "="
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

if [ ! ${n} ] ;then
  echo "ERROR: interval value is missing. See --help" >&2
  exit 1;
fi

sed -n "0~${n}p" /data2/ablimit/Data/spatialdata/bakup/data/osm.mbb.center.sorted.dat | cut -d" " -f6,7 

