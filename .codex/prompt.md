follow the [AGENTS.md](.codex/AGENTS.md) to perform the task. You check the codebase for discriminator-based dipole algorithm under ./robosuite/pipeline/algorithms/dipole, debug and optimize implementation. Also, explain the code logic in detail, in Chinese (you should reasoning in English)

---

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
dipole is a new generative model RL algorithm. You should read [dipole.md](.codex/dipole.md) carefully and understand content inside. Also, you can refer to [dipole.py](.codex/dipole.py) for original JAX version (I want you to write in pytorch)

---

# TODO
your should check, debug and optimize dipole-style algorithm for human-in-the-loop training pipeline under folder ./robosuite/pipeline/algorithms/dipole, following the API ./robosuite/pipeline folder. 

## algorithm
main algorithm is dipole. while original dipole uses advantage-based $w(s, a)$, I now want to use discriminator output $\lambda$ to be $G(s, a)$.
1. for base policy, you should use pretrained flow policy checkpoint ./checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_20260320_114720.pt
2. $w(s, a) = \sigma(\beta G(s, a))$, where $G(s, a) = \lambda$

## Logic
1. using different process / thread to run discriminator and policy inference
2. collected data should be used to update policy using dipole algorithm
3. publish new policy occasionally

---


now start. ask me anything if you are not sure.