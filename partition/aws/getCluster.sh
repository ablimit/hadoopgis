#! /bin/bash

# Tiling first data file
# --ami-version 3.0.2 
# --num-instances=18 --instance-type=c1.xlarge --master-instance-type=m1.medium 

maxred=2
maxmem=1024
coretype=m1.medium
corecount=49
tasktype=m1.medium
taskcount=50
optmem="-m,mapred.reduce.child.java.opts=-Xmx${maxmem}m"

elastic-mapreduce --create --alive --enable-debugging --instance-group master --instance-type m1.medium --instance-count 1 --instance-group core --instance-type ${coretype} --instance-count ${corecount}  --instance-group task --instance-type ${tasktype}  --instance-count ${taskcount} --name "partitionrunner.${corecount}.${taskcount}"  --bootstrap-action 's3://aaji/scratch/awsjoin/bootcopy.sh' --bootstrap-action s3://elasticmapreduce/bootstrap-actions/configure-hadoop --args "-m,mapred.tasktracker.reduce.tasks.maximum=${maxred}" --region 'us-east-1' --log-uri 's3://aaji/scratch/logs' --with-termination-protection --key-pair aaji


# Wait for previous step to complete
# elastic-mapreduce -j ${jobid} --wait-for-steps


# --instance-group master --instance-type m1.medium --instance-count 1 \
# --instance-group core   --instance-type m1.medium --instance-count 5  --bid-price 0.028
# --instance-group task   --instance-type c1.xlarge --instance-count 14 --bid-price 0.028

