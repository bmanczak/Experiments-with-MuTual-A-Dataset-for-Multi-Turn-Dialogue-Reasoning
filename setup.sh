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
conda create -n "ocn" -y Python==3.7
source activate ocn

echo 'install requirements.txt...'
pip install -q -r requirements.txt
echo 'Its done mate cheers'

:'
python run.py \
  --do_train \
  --do_eval \
  --do_lower_case \
  --race_dir RACE \
  --model_dir bert-base-uncased \
  --max_doc_len 400 \
  --max_query_len 30 \
  --max_option_len 16 \
  --train_batch_size 24 \
  --eval_batch_size 24 \
  --learning_rate 1.5e-5 \
  --num_train_epochs 5 \
  --gradient_accumulation_steps 2 \
  --output_dir output
'