#! /bin/bash


if [ ! $# == 2 ]; then
    echo "Usage: $0 [log file] [output file]"
    exit
fi

gpdb='node40.clus.cci.emory.edu'

if [ "$HOSTNAME" = ${gpdb} ] ; then
    date >> $1

    for query in containment.sql # join.sql aggregation.sql
    do
	echo "EXEC ${query}"
	psql --dbname=pais --echo-queries  --output=$2  --file=$query  >> $1
    done

    echo -e "\n\n\n" >> $1


else
    echo "You can not execute SQL on this host."
    exit
fi

