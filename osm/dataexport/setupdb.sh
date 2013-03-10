#! /bin/bash


if [ ! $# == 1 ]; then
    echo "Usage: $0 [DB name]"
    exit
fi

gphost='node40.clus.cci.emory.edu'

if [ "$HOSTNAME" = ${gphost} ] ; then
    
    echo "Creating database $1 ...."
    psql --dbname=postgres -c "CREATE DATABASE $1;"
    echo "Adding spatial extension to $1 ...."
    psql --dbname=$1 --file=$GPHOME/share/postgresql/contrib/postgis.sql
    echo "Uploading data to $1 ...."
    psql --dbname=$1 --file=create_load_gp_schema.sql

fi

