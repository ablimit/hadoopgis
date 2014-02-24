#! /bin/bash

usage(){
  echo -e "prefixpais.sh  options \n \
    --dir \t [ path to the algorithm ] \n \
    --help \t show this information.
  "
  exit 1
}

dir=""

while :
do
  case $1 in
    -h | --help | -\?)
      usage;
      #  Call your Help() or usage() function here.
      exit 0      # This is not an error, User asked help. Don't do "exit 1"
      ;;
    -d | --dir)
      dir=$2     # You might want to check if you really got FILE
      shift 2
      ;;
    --dir=*)
      dir=${1#*=}        # Delete everything up till "="
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

if  [ ! -e ${dir} ] ; then 
  echo "no data is present !"
  exit 1; 
fi


for k in 20 100 200 400 1000 2000 4000 10000 20000 100000
do
  echo "${k} ----------------- "
  for image in astroII.1 astroII.2 gbm0.1 gbm0.2 gbm1.1 gbm1.2 gbm2.1 gbm2.2 normal.2 normal.3 oligoastroII.1 oligoastroII.2 oligoastroIII.1 oligoastroIII.2 oligoII.1 oligoII.2 oligoIII.1 oligoIII.2
  do
    if [ -e ${dir}/c${k}/${image}.geom.1.dat ] ;then 
      sed -i -e "s/^/${image}-/" ${dir}/c${k}/${image}.geom.1.dat
    fi

    if [ -e ${dir}/c${k}/${image}.geom.2.dat ] ;then 
      sed -i -e "s/^/${image}-/" ${dir}/c${k}/${image}.geom.2.dat
    fi

  done
done

