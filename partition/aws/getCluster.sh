#! /bin/bash

c=864
algo=rp

# Tiling first data file
# --ami-version 3.0.2 
# --num-instances=18 --instance-type=c1.xlarge --master-instance-type=m1.medium 
/usr/local/emrcli/elastic-mapreduce --create --alive --stream --enable-debugging --instance-group master --instance-type m1.medium --instance-count 1 --instance-group core --instance-type m1.medium --instance-count 18 --instance-group task --instance-type c1.xlarge --instance-count 80 --name 'partitionrunner'  --bootstrap-action 's3://aaji/scratch/awsjoin/bootcopy.sh' --region us-east-1 --log-uri 's3://aaji/scratch/logs' --with-termination-protection --key-pair aaji --mapper 's3://aaji/scratch/awsjoin/tagmapper.py osm.geom.dat osm.geom.2.dat' --reducer 's3://aaji/scratch/deps/bins/resque st_intersects 1 1' --input "s3://aaji/data/partitions/osm/${algo}/c${c}" --output "s3://aaji/scratch/pout/dec17/${algo}c${c}" --jobconf mapred.reduce.tasks=1000


# --step-name "${algo}.${c}"

# Wait for previous step to complete
# elastic-mapreduce -j ${jobid} --wait-for-steps


# --instance-group master --instance-type m1.medium --instance-count 1 \
# --instance-group core   --instance-type m1.medium --instance-count 5  --bid-price 0.028
# --instance-group task   --instance-type c1.xlarge --instance-count 14 --bid-price 0.028

