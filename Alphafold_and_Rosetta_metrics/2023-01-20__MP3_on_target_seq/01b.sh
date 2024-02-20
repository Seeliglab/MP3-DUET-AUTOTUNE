#!/bin/bash
job_name=01b
out_dir=out/$job_name
input=$job_name.fasta

#--msa-mode single_sequence --model-type AlphaFold2-multimer-v2
CMD="/home/aljubetic/AF2/CF2/bin/colabfold_batch --num-recycle 6 --sort-queries-by length --model-type AlphaFold2-multimer-v2 $input $out_dir"

sbatch << EOF 
#!/bin/bash

#SBATCH --job-name=$job_name
#SBATCH --output=$job_name.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A40:1

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

out_dir=$out_dir

echo RUNNING AS \${USER}@\${HOSTNAME} JOB_NAME: \$JOB_NAME JOB_ID: \$JOB_ID TASK_ID: \$TASK_ID

#activate AF2 enviorment
. /home/aljubetic/bin/set_up_AF2.sh


mkdir -pv $out_dir

echo $CMD
$CMD
EOF

#sbatch $job_name-template.sh
echo "Submited done"