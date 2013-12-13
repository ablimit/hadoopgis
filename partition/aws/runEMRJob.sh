#! /bin/bash

jobid="j-1J7S4TEJ0TP0R"
c=43220
algo=rp

/usr/local/emrcli/elastic-mapreduce --jobflow ${jobid} --stream --mapper 's3://aaji/scratch/awsjoin/tagmapper.py osm.geom.dat osm.geom.2.dat' --reducer "s3://aaji/scratch/deps/bins/resque st_intersects 1 1" --input "s3://aaji/data/partitions/osm/${algo}/c${c}" --output s3://aaji/temp/${algo}c${c} --jobconf mapred.reduce.tasks=160 
