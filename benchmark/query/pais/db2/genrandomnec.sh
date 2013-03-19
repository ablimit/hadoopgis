#! /bin/bash


#if [ ! $# == 2 ]; then
#    echo "Usage: $0 [log file] [output file]"
#    exit
#fi


gpdb='node40.clus.cci.emory.edu'

if [ "$HOSTNAME" = ${gpdb} ] ; then

    for dbname in testp20 testp10 testp5
    do
	echo "${dbname}" 
	for query in setrandom
	do

	    if [ -e ${query}.sql ]
	    then
		echo "EXEC ${query}"
		db2 connect to ${dbname}
		db2 -tvf ${query}.sql  
	    fi
	done
    done
else
    echo "You can not execute SQL on this host."
    exit
fi

