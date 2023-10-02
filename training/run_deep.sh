#!/bin/bash -l
#SBATCH --job-name=mtl-small
#SBATCH --time=48:00:00
#SBATCH --partition ica100
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --qos=qos_gpu
#SBATCH --mem=48GB
#SBATCH -A mdredze80_gpu
#SBATCH --job-name="finetune_askdocs"
#SBATCH --output="/home/vtiyyal1/askdocs/output/finetune_askdocs.out"
#SBATCH --export=ALL

module load cuda/12.1.0
#module load anaconda

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/data/apps/linux-centos8-cascadelake/gcc-9.3.0/anaconda3-2020.07-i7qavhiohb2uwqs4eqjeefzx3kp5jqdu/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/data/apps/linux-centos8-cascadelake/gcc-9.3.0/anaconda3-2020.07-i7qavhiohb2uwqs4eqjeefzx3kp5jqdu/etc/profile.d/conda.sh" ]; then
        . "/data/apps/linux-centos8-cascadelake/gcc-9.3.0/anaconda3-2020.07-i7qavhiohb2uwqs4eqjeefzx3kp5jqdu/etc/profile.d/conda.sh"
    else
        export PATH="/data/apps/linux-centos8-cascadelake/gcc-9.3.0/anaconda3-2020.07-i7qavhiohb2uwqs4eqjeefzx3kp5jqdu/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<


echo "Printing Conda env info..."
conda info --envs
conda activate jupyter

# init virtual environment if needed
# conda create -n toy_classification_env python=3.7

#pip install -r requirements.txt # install Python dependencies
pip install -r requirements.txt

# Set your Weights & Biases API key
echo "Setting W&B API key..."
export WANDB_API_KEY='777501c1a468cab3359a9d2ee89293c06605a76e'
export HUGGINGFACE_TOKEN='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'

# runs your code
echo "Running python script..."



accelerate launch --config_file "configs/deepspeed_config.yaml"  train.py \
--model_name "meta-llama/Llama-2-7b-chat-hf" \
--dataset_name "vtiyyal1/AskDocs-QA" \
--max_seq_len 2048 \
--num_train_epochs 2 \
--max_steps 1000 \
--logging_steps 25 \
--eval_steps 100 \
--save_steps 500 \
--bf16 True \
--packing True \
--output_dir "/scratch4/mdredze1/vtiyyal1/models/askdocsproject" \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 1 \
--dataset_text_field "content" \
--use_gradient_checkpointing \
--use_peft_lora True \
--learning_rate 5e-5  \
--lr_scheduler_type "cosine" \
--weight_decay 0.01 \
--warmup_ratio 0.03 \
--use_flash_attn True