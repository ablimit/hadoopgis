#! /bin/bash


gpdb='node40.clus.cci.emory.edu'

if [ "$HOSTNAME" = ${gpdb} ] ; then

    for query in 1 2 3 4
    do
	echo "EXEC q${query}.sql"

	psql --tuples-only --dbname=pais --file=q${query}.sql | sort -n > q${query}.out

    done

    echo -e "\n\n\n" 


else
    echo "You can not execute SQL on this host."
    exit
fi

