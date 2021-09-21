#!/bin/bash
echo 'Setting up the repo...'

module purge
module load 2019
module load Python/3.7.5-foss-2019b
module load CUDA/10.1.243
module load cuDNN/7.6.5.32-CUDA-10.1.243
module load NCCL/2.5.6-CUDA-10.1.243
module load Anaconda3/2018.12


# Download and untar data
wget http://www.cs.cmu.edu/~glai1/data/race/RACE.tar.gz
tar -xvf RACE.tar.gz
rm RACE.tar.gz

# Download models
git lfs install
# Uncomment which model you need
git lfs clone https://huggingface.co/bert-base-uncased
#git lfs clone https://huggingface.co/bert-large-uncased
#git lfs clone https://huggingface.co/bert-base-cased
#git lfs clone https://huggingface.co/bert-large-cased

# Create directory for output logs
mkdir output

echo 'Its done mate cheers'