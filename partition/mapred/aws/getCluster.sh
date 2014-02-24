#! /bin/bash

# Tiling first data file
# --num-instances=18 --instance-type=c1.xlarge --master-instance-type=m1.medium 
# --ami-version 3.0.4

maxred=2
maxmem=1024
coretype=m1.medium
corecount=1
tasktype=m1.medium
taskcount=1
optmem="-m,mapred.reduce.child.java.opts=-Xmx${maxmem}m"

elastic-mapreduce --create --alive --enable-debugging --instance-group master --instance-type m1.medium --instance-count 1 --instance-group core --instance-type ${coretype} --instance-count ${corecount}  --instance-group task --instance-type ${tasktype}  --instance-count ${taskcount} --name "ppart.${corecount}.${taskcount}" --bootstrap-action 's3://aaji/scratch/awsjoin/bootcopy.sh' --region 'us-east-1' --log-uri 's3://aaji/scratch/logs' --key-pair aaji --with-termination-protection 

# --bootstrap-action s3://elasticmapreduce/bootstrap-actions/configure-hadoop --args "-m,mapred.tasktracker.reduce.tasks.maximum=${maxred}" 


# Wait for previous step to complete
# elastic-mapreduce -j ${jobid} --wait-for-steps


# --instance-group master --instance-type m1.medium --instance-count 1 \
# --instance-group core   --instance-type m1.medium --instance-count 5  --bid-price 0.028
# --instance-group task   --instance-type c1.xlarge --instance-count 14 --bid-price 0.028

