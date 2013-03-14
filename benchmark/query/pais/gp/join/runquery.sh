#! /bin/bash


#if [ ! $# == 2 ]; then
#    echo "Usage: $0 [log file] [output file]"
#    exit
#fi
logfile=res

gpdb='node40.clus.cci.emory.edu'

if [ "$HOSTNAME" = ${gpdb} ] ; then
    date >> ${logfile}.log

    echo "EXEC Type: ${class}"
    echo "EXEC ${class} :" >> ${logfile}.log
    for query in q3.no.centroid q3.opt
    do
        if [ -e ${class}/${query}.sql ]
        then
            echo -n "${query} " >> ${logfile}.log

            echo "EXEC ${query}"

            psql --dbname=pais --output=/shared/tempdir/pais/${class}/${query}.out  --file=${class}/${query}.sql  >> ${logfile}.log
        fi
    done
    echo "" >> ${logfile}.log

else
    echo "You can not execute SQL on this host."
    exit
fi

