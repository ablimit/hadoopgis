#! /bin/bash


if [ ! $# == 2 ]; then
    echo "Usage: $0 [log file] [output file]"
    exit
fi

gpdb='node40.clus.cci.emory.edu'

if [ "$HOSTNAME" = ${gpdb} ] ; then
    date >> $1

    for query in 1 2 3 4 5
    do
	echo "EXEC ${query}"

	echo "EXEC ${query}" >> $1
	
	psql --dbname=pais --output=$2  --file=q${query}.sql  >> $1
    done

    echo -e "\n\n" >> $1


else
    echo "You can not execute SQL on this host."
    exit
fi

