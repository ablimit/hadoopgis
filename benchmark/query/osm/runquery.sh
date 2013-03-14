#! /bin/bash


#if [ ! $# == 2 ]; then
#    echo "Usage: $0 [log file] [output file]"
#    exit
#fi
logfile=perf16

gpdb='node40.clus.cci.emory.edu'

if [ "$HOSTNAME" = ${gpdb} ] ; then
    date >> ${logfile}.log

    for class in cont aggr
    do
	echo "EXEC Type: ${class}"
	echo "EXEC ${class} :" >> ${logfile}.log
	for query in 1 2 3 4 5 6
	do
	    if [ -e ${class}/q${query}.sql ]
	    then
		echo -n "Q${query} " >> ${logfile}.log

		echo "EXEC Q${query}"

		psql --dbname=osm --output=/shared/tempdir/osm/${class}/q${query}.out  --file=${class}/q${query}.sql  >> ${logfile}.log
	    fi
	done
	echo "" >> ${logfile}.log
    done

    echo "" >> ${logfile}.log


else
    echo "You can not execute SQL on this host."
    exit
fi

