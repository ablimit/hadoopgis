#! /bin/bash

if [ ! $# == 1 ]; then
    echo "Usage: $0 [DB name]"
    exit
fi

postbin=/home/aaji/softs/postgresqlgpu/bin

${postbin}/createdb --host localhost --maintenance-db=postgres --echo $1


if [ $? -eq 0 ]
then
    ${postbin}/psql --host=localhost --dbname=$1 --echo-queries  --file=paisdb.sql
else
    echo "Database creation is NOT successful.."
fi
