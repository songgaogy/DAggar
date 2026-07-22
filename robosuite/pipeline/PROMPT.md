## Task-1 [done]

Refactor [pipeline](../pipeline/) codebase.   
现在整个codebase充满了冗长的legacy features. 请你重构codebase, 删除原来 update while inference 的逻辑，加上现在 batch online 的"多次offline"逻辑.

### Legacy

1. 所有的 train_dipole 相关的逻辑，入口为 [train_dipole](./scripts/train_dipole.sh), [train_dipole_rl](./scripts/train_dipole_rl.sh) and [eval_dipole](./scripts/eval_dipole.sh).
2. 这些都是使用 online "边infer边update"的方法，现在可以直接删除了。但是注意，[inference](./offline/scripts/collect_data.sh) 还是需要的，因为我们需要 batch online, 使用更新好的policy采集数据。

### Description

目标 [pipeline](../pipeline/) 中包含整个 batch online dipole with vast and discriminator 的逻辑。总体算法流程：collect data -> offline update discriminator+vast+dipole_policy -> collect data -> ... 如此循环。其中，单次循环已经写好，见 [offline](./offline/)

其中"多轮循环"需要按照下面的 "algorithm" 逻辑, 也请遵守 "other" 中的要求.

#### Algorithm

1. 确保第一次循环的结果与当前 [offline](./offline/) 的相同（假设数据一样，不考虑seed或其他的随机性）。
    - 使用现在 3-term loss + g regularization 的算法
    - 在数据的使用上，$L_{nnPU}$ 仍使用原来的 pretrain data; $L_P$ 与 $L_{gt-fail}$ 使用 moving average 方法 $\beta D^{old} + (1-\beta) D^{new}$ 的方法采样， $\beta$ 可调
3. 对于 VAST 的多轮更新: 数据直接混合，与一轮更新相同
4. 对于 dipole 的多轮更新：数据直接混合，与一轮更新相同
5. 数据存储与run result格式：
    - output `outputs/dipole/<env>/<run_name_with_time_stamp>/`
    - 所有data都存储在这个目录下，按照轮次存储，注意格式与内容需要方便读取
    - checkpoint也都存储在下面
    - 每次启动程序的时候只需要使用这个run的根文件夹即可

#### Others

1. refactor [configs](./config/), 尽可能整合为一个 .yaml 文件(合并重复依赖，重命名不规范的变量名)，同时不同 task 用自己的 yaml. 注意，公用的 config 请使用公用的config "overall.yaml"
2. 这次 refactor 变动较大，请复核大型 python 项目规范，依据常见深度学习项目的规范。可以完全打乱文件排列及其依赖逻辑(也就是不需要一定要有 offline 这个folder了)


## Task-2 [tbd]

Restore or re-add visualization logic for discriminator and VAST. Then refactor the files under [pipeline](../pipeline).  
1. 在上一个commit中，本来有的visualization逻辑被当成legacy删除了，现在请你恢复，请注意代码规范以及存放的位置规范
2. 需要重新重构。目前主要逻辑已经完全正确，每个文件/code也都完整，但是整体files/scripts排布仍然混乱，需要调整一下文件/文件夹关系以及命名等(具体的code内容基本上不用变了，除了import等依赖关系)

### Restore Vis

参考 git branch "dipole-rl/v3-vast-disc_reward", 其中包含了所有visualization相关的逻辑与具体代码(只不过是refactor之前的). 

你需要完成/恢复的相关代码逻辑:
1. Discriminator: evaluation+visualization. see [eval_disc](./robosuite/pipeline/offline/scripts/eval_disc_finetuned.sh). 
2. VAST: visualization. see [vis_init](./robosuite/pipeline/scripts/utils/vis_init_vast.sh)

注意:
1. output 存储在 `<run_root>/rounds/<round_idx>/eval_vis` 中, 创建 `./disc` or `vast`
2. 输入是 `<run_root>` 以及对应的 `round_idx`

### Refactor & Rename

1. 现在 [offline](./offline/) 看着很多余，是不是应该拆掉里面的内容散开？
2. [env](./envs/) and [data](./data) 里面就一个文件，不用单独放 (也许放入 common?)
3. [orchestration](./orchestration) 命名有点奇怪
