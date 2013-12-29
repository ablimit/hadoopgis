#! /bin/bash

# Tiling first data file
# --ami-version 3.0.2 
# --num-instances=18 --instance-type=c1.xlarge --master-instance-type=m1.medium 
elastic-mapreduce --create --stream --enable-debugging --instance-group master --instance-type m1.medium --instance-count 1 --instance-group core --instance-type c1.xlarge --instance-count 23 --instance-group task --instance-type c1.xlarge --instance-count 25 --name 'TCGA'  --bootstrap-action 's3://aaji/scratch/awsjoin/bootcopy.sh' --bootstrap-action s3://elasticmapreduce/bootstrap-actions/configure-hadoop --bootstrap-name "Set Maximum Cuncurrent Reduce Task" --args "-m,mapred.tasktracker.reduce.tasks.maximum=8,-m,mapred.reduce.child.java.opts=-Xmx768m" --bootstrap-action s3://elasticmapreduce/bootstrap-actions/configure-hadoop --bootstrap-name "Disable reducer speculative execution" --args "-m,mapred.reduce.tasks.speculative.execution=false" --region us-east-1 --log-uri 's3://aaji/scratch/logs' --key-pair aaji --mapper 's3://aaji/scratch/awsjoin/tcgamapper.py' --reducer "s3://aaji/scratch/deps/bins/resque st_intersects 1 1" --input s3://aaji/data/tcga --output s3://aaji/scratch/pout/dec27/tcga_1 --jobconf mapred.reduce.tasks=2000

# --step-action TERMINATE_JOB_FLOW
# --step-name "tcga.st_intersect.full" 

