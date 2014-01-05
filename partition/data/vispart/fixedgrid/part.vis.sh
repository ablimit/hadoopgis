#! /bin/bash

fillFactor=0.80

idir=../testdata
opath=meta
# tempPath=/dev/shm
tempPath=meta

if [ ! -e fixedgridPartition ];
then
	echo "executables are not available."
	exit 1;
fi

if [ ! -d ${tempPath} ];
then 
	mkdir -p ${tempPath}
fi


for tid in 2 3 4 5 6 7
do
	ifile="${idir}/test${tid}.obj.ns.txt"

	echo -e "\n--------------${tid}----------------------"
	echo -e "\ngenerating partition region..."
	./fixedgridPartition ${ifile} 4  > ${opath}/regionmbb.${tid}.txt 

done

