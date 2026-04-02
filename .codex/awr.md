follow the [AGENTS.md](.codex/AGENTS.md) to perform the task. Implement AWR algorithm under folder ./robosuite/pipeline/algorithms/awr.

# Background

## pipeline
The pipeline folder is a human-in-the-loop style, roboisuite simulation based robotic algorithm codebase. directory: ./robosuite/pipeline
1. It use [factory.py](robosuite/pipeline/factory.py) to initialize algorithm
2. it uses spacemouse for human online correction and intervention. Note that env async rendering logic is very complicated, so please 100% follow the logic of existing implementation for ./robosuite/pipeline/algorithms/flow_dagger (bash entrance: [train_flow_dagger.sh](robosuite/pipeline/scripts/train_flow_dagger.sh) ), whose environment rendering logic is correct.
3. it uses mutli-thread / process to simultaneously perform rollout inference and model training. new model will be published occasionally.

## discriminator
I have implemented and trained a dynamic model based failure discriminator under folder ./robosuite/discriminator/lpb_new
1. discriminator logic: first train a dynamic model using pretrained multi-tasks policy net encoder, then use the dynamic model to predict next state. By comparing the real next states and predicted next states (and other intuitive metrics), it will output a lambda. If this lambda is higher than threshold, we denote this state as failure / OOD
2. you can use trained discriminator (dynamic model) checkpoint: ./checkpoints/multitask_6/lpb_new/lpb_new_20260324_232921/lpb_new_20260324_232921_ep0050.pt

## base policy
base policy for multi-task is under ./robosuite/policy/flow_multi, which is a flow policy. You should refer to ./robosuite/policy/flow_multi, which is original implementation. You may also need to read ./robosuite/pipeline/algorithms/flow_dagger because under this folder, a identical flow policy is being adapted to pipeline folder api.  
1. you should use checkpoint of ./checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_20260320_114720.pt as base policy
2. note that base policy is for multi-tasks, but when using dipole online tuning, please use fixed one task (language can be changed as long as it described the same task)

## Dipole algorithm
dipole is a new generative model RL algorithm. You should read [dipole.md](.codex/dipole.md) carefully and understand content inside. Also, you should read dipole implementation under robosuite/pipeline/algorithms/dipole, with bash entrance of [train_dipole.py](robosuite/pipeline/scripts/train_dipole.sh)

## data
for each task, we have expert data under folder ./data/<task_name>/expert, and success/failure rollout data under ./data/<task_name>/fail_rollout and ./data/<task_name>/fail_rollout.

# TODO

You should implement standard online version of AWR algorithm under folder ./robosuite/pipeline/algorithms/awr.

Note that:
1. you should warm up value function using both failure and success rollout trajectories before online training; Use IQL to do this, no policy update is required;
2. maintain two buffer during online steps: demo buffer and online buffer; keep human intervention data into both demo buffer and online buffer, while policy rollout data should be only put into online buffer. During sampling, you should sample 1:1 from demo data buffer and online data buffer.
3. simultaneously doing online discriminator and policy infer (rendering), using multi-thread; for this logic, you should refer to dipole implementation
4. note that policy is a flow policy, so you should refer to dipole implementation which can update generative network in RL. 
    $$
    \theta^\star = \arg\min_\theta
    \mathbb E_{(s,a)\sim\mathcal D,\epsilon,t}
    \left[
    \exp!\left(\frac{1}{\beta}A(s,a)\right)
    \left|v_\theta(x_t,t,s)-u_t(a,\epsilon)\right|^2
    \right]
    $$
5. after each rollout, you should start value and policy update for `NUM_UPDATE` times, and publish new policy


make detailed plan first before implement, ask me if there is something unclear.