# Context

You should read files under @robosuite/pipeline folder. Follow @robosuite/pipeline/docs/DIPOLE.md and @robosuite/pipeline/docs/README.md

# Task

I want to build a pipeline for DIPOLE-based human-in-the-loop robosuite RL training. Steps are as follows:

1. train a base policy using expert data
2. using that base policy to collect success rollout and failure rollout data
3. train a feature extractor (WAM-style), denote as encoder: $f(o,s)$. Freeze. see [here](#note) for more detail.
4. based on that encoder, train a base discriminator $d(f(o,s))$ using BCE target. Note that according to RL theory and occupancy measure view, this discriminator is a intrinsic reward $r'(s,a)$ for current policy. Also, this discriminator tells the human about when to intervene and when to stop.
5. warmup a Q for online RL on the offline dataset, note that here Q and V need to use latent from encoder $f(o,s)$, and reward comes from -1/0 reward + discrminator intrinsic reward both.
6. use warmup-ed Q/V, discriminator to start online dipole RL training: 
    - inference: dipole policy
    - training: dipole policy, discriminator (online training, use human-intervention to train discriminator. note that discrminator is also conditioned on current policy), Q/V (online learning, start from warmup-ed network, reward comes from -1/0 reward + current discrminator intrinsic reward both)

# Background

I have implemented base DIPOLE algorithm and discrimiantor:
- [done]: (1), (2), (3), (4) and discriminator based-dipole human-in-the-loop architecture.
- [tbd]: (5) (6). Q/V part(warmup and online update) + online discriminator update + Q/V based-dipole

# NOTE

1. discrminator is under @robosuite/discriminator/lpb_v2 folder. discriminator for dipole should be implemented under @robosuite/pipeline/algorithms/discriminator
2. For Q/V learning and warmup, you should use Cal-QL algorithm and its implementation should be under @robosuite/pipeline/algorithms/q_learning
3. current dipole use discriminator output as G, but actually is should be G=Advantage.
4. feature extractor (WAM-style) encoder takes image+state+action as input, but here you can directly repeat s into a to degrade this encoder as $f(o,s)$ rather than $f(o,s,a)$. For Q-network, action should be add into network additionally.
5. all the module should be update online. So write efficient code and infra, ensure high simulation fps, using parallel computing and programming. We have 2 4090 GPU, make good use of this. (difficuly lies in real time flow-policy infer + Q/V and discriminator + policy training should be execute together)

# TODO

1. Make detailed prompts, this prompts should specify plan for all subagents. Write prompts into markdown file under @robosuite/pipeline/docs
2. decide apis and interfaces between modules like Q/V, discriminator... . 
3. create files, you do NOT need to write code, but you should make files, create base class, decide interface(input output protoacol)

communicate with me in Chinese, ask me anything if you are unclear. 