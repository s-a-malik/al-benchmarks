#!/bin/bash

#$ -S /bin/bash
#$ -l gpu=true,gpu_type=!(gtx1080ti|rtx2080ti)
#$ -l tmem=14G
#$ -l h_rt=08:00:00
#$ -l tscratch=20G
#$ -wd /home/smalik/output_files

#$ -N al_hypers_sweep
#$ -j y

#$ -t 1-20

echo "We're Starting!"
hostname
date

CODE_DIR=/home/smalik/AL_benchmarks
export PYTHONPATH=$PYTHONPATH:$CODE_DIR
 
source /share/apps/source_files/python/python-3.8.5.source
source /share/apps/source_files/cuda/cuda-10.1.source

# get to correct directory
cd $CODE_DIR/

# set up wandb
wandb login <INSERT API KEY>

# create scratch dir
SCRATCH_DIR=/scratch0/smalik/$JOB_ID.$SGE_TASK_ID 
mkdir -p $SCRATCH_DIR
# export to make available in python
export SCRATCH_DIR

# run experiment (the config files are set up in the sweep yaml)
# TODO change the SWEEP_ID to the initialised one
wandb agent --count 1 <SWEEP_ID>	

# delete scratch data
function finish {
    rm -rf $SCRATCH_DIR
}
echo "deleting scratch"
trap finish EXIT ERR INT TERM

date
echo "We're done!"
