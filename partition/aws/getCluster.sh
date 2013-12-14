#! /bin/bash

c=43220
algo=rp

# Tiling first data file
/usr/local/emrcli/elastic-mapreduce --create --alive --stream --enable-debugging --num-instances=14 --instance-type=c1.xlarge --master-instance-type=m1.medium --name 'partitionrunner'  --bootstrap-action 's3://aaji/scratch/awsjoin/bootcopy.sh' --region us-east-1 --log-uri 's3://aaji/scratch/logs' --with-termination-protection --key-pair aaji --mapper 's3://aaji/scratch/awsjoin/tagmapper.py osm.geom.dat osm.geom.2.dat' --reducer 's3://aaji/scratch/deps/bins/resque st_intersects 1 1' --input "s3://aaji/data/partitions/osm/${algo}/c${c}" --output "s3://aaji/temp/${algo}c${c}" --jobconf mapred.reduce.tasks=160 


#elastic-mapreduce --create --alive --stream --num-instances=20 --instance-type=c1.xlarge --master-instance-type=m1.medium --name "partitionrunner"  --mapper 's3://aaji/scratch/awsjoin/tagmapper.py ' --reducer "s3://aaji/scratch/deps/bins/resque st_intersects 1 1" --input "s3://aaji/data/partitions/osm/${algo}/c${c}" --output s3://aaji/temp/${algo}c${c} --jobconf mapred.reduce.tasks=160 --bootstrap-action "s3://aaji/scratch/awsjoin/bootcopy.sh"

# Wait for previous step to complete
#ruby ./elastic-mapreduce -j ${jobid} --wait-for-steps
