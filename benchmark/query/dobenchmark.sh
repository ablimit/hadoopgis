#! /bin/bash


#if [ ! $# == 2 ]; then
#    echo "Usage: $0 [log file] [output file]"
#    exit
#fi
logfile=perf8

gpdb='node40.clus.cci.emory.edu'

if [ "$HOSTNAME" = ${gpdb} ] ; then

    echo "benchmarking PAIS"
    cd pais/gp
    sh runquery.sh

    cd -

    echo "benchmarking OSM"
    cd osm
    sh runquery.sh

else
    echo "You can not execute SQL on this host."
    exit
fi

