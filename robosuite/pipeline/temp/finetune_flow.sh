cd ~/Documents/DAggar/robosuite

python -m robosuite.pipeline.temp.train_flow_offline \
    --env PickPlaceBread \
    --num-trajectories 20 \
    --steps 10000 \
    --device cuda:0 \
    --save-interval 2000 \
    --output-postfix "expert-pretrain-data" \
    --data-type "expert_pretrain_data"