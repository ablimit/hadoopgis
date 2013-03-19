#! /bin/bash


#if [ ! $# == 2 ]; then
#    echo "Usage: $0 [log file] [output file]"
#    exit
#fi
logfile=perf

dbname=testdb

gpdb='node40.clus.cci.emory.edu'

if [ "$HOSTNAME" = ${gpdb} ] ; then
    date >> ${logfile}.log

    for dbname in testdb
    #for dbname in testdb testp20 testp10 testp5
    do
	echo "${dbname}" >> ${logfile}.log
	for query in aggregation containment #join
	# for query in join
	do
	    echo "EXEC ${query} :" >> ${logfile}.log

	    if [ -e ${query}.sql ]
	    then
		echo "EXEC ${query}"
		db2 connect to ${dbname}
		date >> ${logfile}.log
		db2 -tf ${query}.sql  > /shared/db2testout/pais/${dbname}/${query}.out 
		date >> ${logfile}.log
	    fi
	done
    done
    echo "" >> ${logfile}.log
else
    echo "You can not execute SQL on this host."
    exit
fi

