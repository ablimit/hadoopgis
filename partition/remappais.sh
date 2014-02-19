#! /bin/bash

usage(){
  echo -e "remappais.sh  options \n \
    --alg \t [ fg | bsp | hc | st | rp | rt ] \n \
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


# argument checking 
if [ "${algo}" != "st" ] && [ "${algo}" != "rt" ] && [ "${algo}" != "rp" ] && [ "${algo}" != "fg" ] && [ "${algo}" != "hc" ] && [ "${algo}" != "bsp" ];
then
  echo "Parameter [${algo}] is NOT recognized. Alternatives are [ st | rp | rt | fg | hc | bsp ]"
  exit 1;
fi
# osm
geompath=/data2/ablimit/Data/spatialdata/pais/geom
opath=/data/ablimit/Data/spatialdata/bakup/data/partition/pais

if  [ ! -e ${opath}/${algo} ] ; then 
  echo "no data is present !"
  exit 1; 
fi


for image in astroII.1 astroII.2 gbm0.1 gbm0.2 gbm1.1 gbm1.2 gbm2.1 gbm2.2 normal.2 normal.3 oligoastroII.1 oligoastroII.2 oligoastroIII.1 oligoastroIII.2 oligoII.1 oligoII.2 oligoIII.1 oligoIII.2
do
  for k in 20 100 200 400 1000 2000 4000 10000 20000 100000
  do

    if [ "${algo}" == "rt" ] || [ "${algo}" == "rp" ] || [ "${algo}" == "fg" ] || [ "${algo}" == "bsp" ]; then
      
      echo "set 1 ---- ${image} --- ${k} --- "
      python remappaistogeom.py ${opath}/${algo}/c${k}/${image}.part < ${geompath}/${image}.1.dat > ${opath}/${algo}/c${k}/${image}.geom.1.dat
      
      echo "set 2 ---- ${image} --- ${k} --- "
      python remappaistogeom.py ${opath}/${algo}/c${k}/${image}.2.part < ${geompath}/${image}.2.dat > ${opath}/${algo}/c${k}/${image}.geom.2.dat
    
    elif [ "${algo}" == "st" ]; then 
      
      echo "[${k}] [${algo}]"
      python remappaistogeom.py ${opath}/${algo}/x/c${k}/${image}.part < ${geompath}/${image}.1.dat > ${opath}/${algo}/x/c${k}/${image}.geom.1.dat
      
      python remappaistogeom.py ${opath}/${algo}/x/c${k}/${image}.2.part < ${geompath}/${image}.2.dat > ${opath}/${algo}/x/c${k}/${image}.geom.2.dat
    
    else 
      echo "Dumb input !"
      exit 1;
    fi
  done
done

