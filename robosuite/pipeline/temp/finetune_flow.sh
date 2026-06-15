cd ~/Documents/DAggar/robosuite

export CUDA_VISIBLE_DEVICES=0

CKPT="checkpoints/multitask_6/flow_multi_ep0100.pt"

# finetune model
python -m robosuite.pipeline.temp.train_flow_offline \
    --init-checkpoint "${CKPT}" \
    --env NutAssemblyRound \
    --num-trajectories 10 \
    --steps 10000 \
    --save-interval 2000 \
    --train-scope all \
    --output-root "./outputs/flow-sft" \
    --output-postfix "pretrain-data" \
    --data-type "pretrain_data-20260615_174814"
