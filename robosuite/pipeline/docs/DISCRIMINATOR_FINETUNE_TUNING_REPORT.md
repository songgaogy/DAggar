# PickPlaceCereal Discriminator Finetune Tuning Report

Date: 2026-07-16<br>
Task: `PickPlaceCereal`<br>
Seed: `0`<br>
Compute: two independent RTX 4090 CUDA processes (`CUDA_VISIBLE_DEVICES=0/1`, process-local `cuda:0`)

## Outcome

The strict acceptance target was **not met** after all 50 approved new trials. The closest trial satisfied 9 of 10 numeric gates. It missed only event precision: `0.2066` versus the required parent value `0.2109375`. This report therefore does not claim a successful finetune.

The best observed near-miss is `stage3/PickPlaceCereal_20260716_040503_s3_next_w0025`. Its configuration has been written to `robosuite/pipeline/config/finetune_disc.yaml` so the checked-in defaults reproduce the best observed trade-off:

- epochs `10`, learning rate `3e-5`, weight decay `1e-4`;
- supervised GT-BCE weight `0.025`, positive/negative class weights `0.5/0.5`;
- GT window `(pre, post, pre_end)=(1,1,0)`;
- nnPU replay weight `1.0`, batch size `512`, frozen encoder and fixed parent nnPU method.

This is a diagnostic best candidate, not an accepted model.

## Protocol and gates

Every new trial used the same parent action-chunk checkpoint, encoder `model_10.pth`, fixed `delta=10`, fixed nnPU configuration, seed `0`, and no generated data. Each checkpoint was evaluated on all 45 failure and 50 success validation trajectories. Success false alarms use only frames before the first `is_success=true`, excluding post-completion padding. The offline GT diagnostic is a train-set diagnostic and is not treated as held-out evidence.

The ten gates were: GT pre detection `>=0.90`; trajectory AUROC/AUPRC `>=0.979333/0.978404`; frame AUROC/AUPRC/F1 `>=0.912063/0.961426/0.871924`; event recall `=1.0`; event precision `>=0.2109375`; valid success-frame FPR `<=0.158967`; and success trajectory alarm rate `<=0.75`.

## Baselines and existing trials

| Model | Configuration | GT pre | Traj AUROC/AUPRC | Frame AUROC/AUPRC/F1 | Event R/P | Success FPR/alarm | Result |
|---|---|---:|---:|---:|---:|---:|---|
| Parent action-chunk | Frozen parent; no finetune | N/A | 0.9893/0.9884 | 0.9221/0.9714/0.8819 | 1.000/0.2109 | 0.1490/0.70 | Reference |
| Existing `20260715_172530` | 50 ep, LR 3e-5, legacy offline objective | 0.6128 | 0.9858/0.9852 | 0.9231/0.9711/0.8790 | 1.000/0.2038 | 0.1564/0.76 | Fail |
| Existing `20260716_005542` | 20 ep, LR 3e-5, GT w 1.0, window 3/3/3 | 1.0000 | 0.9867/0.9858 | 0.8918/0.9545/0.8812 | 1.000/0.2033 | 0.1304/0.62 | Fail |

Load-only baseline outputs are under `outputs/discriminator-finetune/tuning_20260716/baselines/`. The parent rerun exactly reproduced the saved parent benchmark metrics.

## All 50 new trials

All output paths below are relative to `outputs/discriminator-finetune/tuning_20260716/<stage>/`. Every run directory contains resolved configuration, checkpoint, TensorBoard events, GT diagnostic, `benchmark.json`, and `success_false_alarm.json`. `Final loss` and `Clamp` are the final TensorBoard epoch aggregates; clamp is the nnPU clamp fraction. Window is `pre/post/pre_end`; CW is positive/negative class weight.

### Stage 1: 24-trial grid

| Run suffix | Ep | LR | GT w | Window | CW | Final loss | Clamp | GT pre | Traj AUC/AP | Frame AUC/AP/F1 | Event R/P | Success FPR/alarm | Gates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 022433_s1_w005_lr3e6_e5 | 5 | 3e-6 | .05 | 3/3/3 | .5/.5 | .0853 | .97 | .3994 | .9827/.9818 | .9438/.9788/.9015 | 1/.2000 | .1854/.74 | 7/10 |
| 022856_s1_w005_lr1e5_e5 | 5 | 1e-5 | .05 | 3/3/3 | .5/.5 | .0648 | 1 | .5666 | .9876/.9864 | .9429/.9783/.9007 | 1/.1959 | .1822/.74 | 7/10 |
| 022857_s1_w005_lr3e6_e10 | 10 | 3e-6 | .05 | 3/3/3 | .5/.5 | .0766 | .96 | .4835 | .9858/.9846 | .9432/.9785/.9008 | 1/.1958 | .1831/.74 | 7/10 |
| 023249_s1_w005_lr3e5_e5 | 5 | 3e-5 | .05 | 3/3/3 | .5/.5 | .0303 | 1 | .9575 | .9938/.9933 | .9318/.9741/.8940 | 1/.2015 | .1662/.66 | 8/10 |
| 023301_s1_w005_lr1e5_e10 | 10 | 1e-5 | .05 | 3/3/3 | .5/.5 | .0413 | 1 | .8942 | .9884/.9874 | .9406/.9774/.8992 | 1/.2227 | .1778/.68 | 8/10 |
| 023649_s1_w010_lr3e6_e5 | 5 | 3e-6 | .10 | 3/3/3 | .5/.5 | .1392 | .97 | .3957 | .9791/.9783 | .9453/.9793/.9039 | 1/.2017 | .1911/.78 | 4/10 |
| 023705_s1_w005_lr3e5_e10 | 10 | 3e-5 | .05 | 3/3/3 | .5/.5 | .0205 | 1 | .9962 | .9951/.9947 | .9125/.9664/.8829 | 1/.1733 | .1522/.70 | 9/10 |
| 024042_s1_w010_lr1e5_e5 | 5 | 1e-5 | .10 | 3/3/3 | .5/.5 | .1075 | 1 | .5656 | .9876/.9864 | .9436/.9785/.9012 | 1/.1926 | .1850/.74 | 7/10 |
| 024109_s1_w010_lr3e6_e10 | 10 | 3e-6 | .10 | 3/3/3 | .5/.5 | .1245 | .96 | .4825 | .9853/.9842 | .9442/.9788/.9018 | 1/.2000 | .1862/.74 | 7/10 |
| 024435_s1_w010_lr3e5_e5 | 5 | 3e-5 | .10 | 3/3/3 | .5/.5 | .0525 | 1 | .9622 | .9938/.9933 | .9292/.9730/.8934 | 1/.1963 | .1629/.68 | 8/10 |
| 024506_s1_w010_lr1e5_e10 | 10 | 1e-5 | .10 | 3/3/3 | .5/.5 | .0691 | 1 | .9037 | .9898/.9890 | .9399/.9771/.8988 | 1/.2232 | .1787/.68 | 9/10 |
| 024826_s1_w025_lr3e6_e5 | 5 | 3e-6 | .25 | 3/3/3 | .5/.5 | .2695 | .99 | .3305 | .9724/.9720 | .9471/.9800/.9103 | 1/.2070 | .1845/.80 | 4/10 |
| 024901_s1_w010_lr3e5_e10 | 10 | 3e-5 | .10 | 3/3/3 | .5/.5 | .0356 | 1 | .9981 | .9956/.9951 | .9119/.9658/.8836 | 1/.1806 | .1580/.70 | 8/10 |
| 025218_s1_w025_lr1e5_e5 | 5 | 1e-5 | .25 | 3/3/3 | .5/.5 | .2171 | .99 | .5401 | .9871/.9861 | .9441/.9787/.9020 | 1/.1967 | .1861/.76 | 6/10 |
| 025257_s1_w025_lr3e6_e10 | 10 | 3e-6 | .25 | 3/3/3 | .5/.5 | .2416 | .91 | .4589 | .9836/.9826 | .9456/.9793/.9039 | 1/.2000 | .1929/.78 | 6/10 |
| 025610_s1_w025_lr3e5_e5 | 5 | 3e-5 | .25 | 3/3/3 | .5/.5 | .1205 | 1 | .9556 | .9938/.9933 | .9284/.9725/.8945 | 1/.2070 | .1618/.66 | 8/10 |
| 025654_s1_w025_lr1e5_e10 | 10 | 1e-5 | .25 | 3/3/3 | .5/.5 | .1535 | 1 | .8480 | .9911/.9905 | .9394/.9768/.8988 | 1/.2097 | .1782/.68 | 7/10 |
| 030003_s1_w050_lr3e6_e5 | 5 | 3e-6 | .50 | 3/3/3 | .5/.5 | .4536 | .99 | .2908 | .9604/.9624 | .9482/.9804/.9147 | 1/.2222 | .1751/.82 | 5/10 |
| 030050_s1_w025_lr3e5_e10 | 10 | 3e-5 | .25 | 3/3/3 | .5/.5 | .0637 | .99 | .9981 | .9947/.9940 | .9043/.9625/.8782 | 1/.1971 | .1537/.70 | 8/10 |
| 030356_s1_w050_lr1e5_e5 | 5 | 1e-5 | .50 | 3/3/3 | .5/.5 | .3761 | .97 | .5052 | .9871/.9861 | .9445/.9788/.9023 | 1/.1967 | .1895/.76 | 6/10 |
| 030446_s1_w050_lr3e6_e10 | 10 | 3e-6 | .50 | 3/3/3 | .5/.5 | .4063 | .90 | .4287 | .9809/.9804 | .9468/.9798/.9077 | 1/.2069 | .1891/.80 | 6/10 |
| 030749_s1_w050_lr3e5_e5 | 5 | 3e-5 | .50 | 3/3/3 | .5/.5 | .2429 | 1 | .9188 | .9938/.9934 | .9308/.9732/.8960 | 1/.2070 | .1640/.66 | 8/10 |
| 030841_s1_w050_lr1e5_e10 | 10 | 1e-5 | .50 | 3/3/3 | .5/.5 | .2988 | 1 | .7120 | .9911/.9905 | .9400/.9769/.8997 | 1/.2107 | .1777/.72 | 7/10 |
| 031236_s1_w050_lr3e5_e10 | 10 | 3e-5 | .50 | 3/3/3 | .5/.5 | .0956 | .99 | .9981 | .9933/.9927 | .8914/.9571/.8728 | 1/.1935 | .1390/.68 | 7/10 |

### Stage 2: 18 variants

| Run suffix | Ep | LR | GT w | Window | CW | Final loss | Clamp | GT pre | Traj AUC/AP | Frame AUC/AP/F1 | Event R/P | Success FPR/alarm | Gates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 031722_s2_b1_win110 | 10 | 1e-5 | .10 | 1/1/0 | .5/.5 | .0558 | 1 | .9280 | .9902/.9895 | .9382/.9766/.8955 | 1/.2167 | .1711/.70 | 9/10 |
| 031723_s2_b1_win310 | 10 | 1e-5 | .10 | 3/1/0 | .5/.5 | .0658 | 1 | .9122 | .9898/.9890 | .9399/.9772/.8973 | 1/.2158 | .1784/.70 | 9/10 |
| 032112_s2_b1_win510 | 10 | 1e-5 | .10 | 5/1/0 | .5/.5 | .0713 | 1 | .8974 | .9898/.9890 | .9388/.9767/.8966 | 1/.2119 | .1847/.74 | 8/10 |
| 032114_s2_b1_win330 | 10 | 1e-5 | .10 | 3/3/0 | .5/.5 | .0721 | 1 | .9056 | .9898/.9890 | .9398/.9770/.8987 | 1/.2170 | .1771/.68 | 9/10 |
| 032504_s2_b1_cw2575 | 10 | 1e-5 | .10 | 3/3/3 | .25/.75 | .0667 | 1 | .9452 | .9880/.9870 | .9408/.9775/.8997 | 1/.2114 | .1783/.66 | 9/10 |
| 032508_s2_b1_cw1090 | 10 | 1e-5 | .10 | 3/3/3 | .10/.90 | .0523 | 1 | .9726 | .9862/.9852 | .9418/.9780/.9004 | 1/.2031 | .1814/.66 | 8/10 |
| 032857_s2_b2_win110 | 10 | 3e-5 | .05 | 1/1/0 | .5/.5 | .0154 | 1 | 1.0000 | .9947/.9941 | .9135/.9674/.8780 | 1/.1993 | .1525/.74 | 9/10 |
| 032900_s2_b2_win310 | 10 | 3e-5 | .05 | 3/1/0 | .5/.5 | .0185 | 1 | .9981 | .9956/.9951 | .9170/.9682/.8828 | 1/.2007 | .1569/.72 | 9/10 |
| 033249_s2_b2_win510 | 10 | 3e-5 | .05 | 5/1/0 | .5/.5 | .0222 | 1 | .9982 | .9951/.9947 | .9202/.9691/.8847 | 1/.1978 | .1762/.74 | 8/10 |
| 033251_s2_b2_win330 | 10 | 3e-5 | .05 | 3/3/0 | .5/.5 | .0212 | 1 | .9962 | .9951/.9947 | .9089/.9652/.8804 | 1/.1765 | .1490/.68 | 8/10 |
| 033645_s2_b2_cw1090 | 10 | 3e-5 | .05 | 3/3/3 | .10/.90 | .0114 | 1 | 1.0000 | .9942/.9938 | .9153/.9680/.8837 | 1/.1761 | .1505/.70 | 9/10 |
| 033646_s2_b2_cw2575 | 10 | 3e-5 | .05 | 3/3/3 | .25/.75 | .0173 | 1 | .9981 | .9951/.9947 | .9116/.9663/.8831 | 1/.1711 | .1471/.68 | 8/10 |
| 034040_s2_b3_win110 | 5 | 3e-5 | .25 | 1/1/0 | .5/.5 | .0891 | 1 | .9861 | .9938/.9932 | .9250/.9715/.8878 | 1/.2070 | .1604/.72 | 8/10 |
| 034045_s2_b3_win310 | 5 | 3e-5 | .25 | 3/1/0 | .5/.5 | .1112 | 1 | .9632 | .9942/.9938 | .9294/.9729/.8912 | 1/.2061 | .1657/.70 | 8/10 |
| 034429_s2_b3_win510 | 5 | 3e-5 | .25 | 5/1/0 | .5/.5 | .1262 | 1 | .9523 | .9942/.9938 | .9300/.9731/.8900 | 1/.2072 | .1801/.74 | 8/10 |
| 034435_s2_b3_win330 | 5 | 3e-5 | .25 | 3/3/0 | .5/.5 | .1251 | 1 | .9490 | .9938/.9933 | .9281/.9724/.8937 | 1/.2078 | .1623/.66 | 8/10 |
| 034822_s2_b3_cw2575 | 5 | 3e-5 | .25 | 3/3/3 | .25/.75 | .1034 | 1 | .9802 | .9933/.9929 | .9264/.9718/.8932 | 1/.1971 | .1599/.66 | 8/10 |
| 034828_s2_b3_cw1090 | 5 | 3e-5 | .25 | 3/3/3 | .10/.90 | .0673 | 1 | .9915 | .9938/.9933 | .9262/.9719/.8937 | 1/.1930 | .1583/.66 | 9/10 |

### Stage 3: 8 local-neighborhood trials

| Run suffix | Ep | LR | GT w | Window | CW | Final loss | Clamp | GT pre | Traj AUC/AP | Frame AUC/AP/F1 | Event R/P | Success FPR/alarm | Gates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 035320_s3_top_w0025 | 10 | 3e-5 | .025 | 3/1/0 | .5/.5 | .0114 | 1 | .9943 | .9956/.9951 | .9183/.9690/.8825 | 1/.1957 | .1562/.74 | 9/10 |
| 035320_s3_top_w0100 | 10 | 3e-5 | .10 | 3/1/0 | .5/.5 | .0295 | .99 | .9981 | .9956/.9951 | .9188/.9685/.8835 | 1/.1925 | .1693/.74 | 8/10 |
| 035713_s3_top_lr1e5 | 10 | 1e-5 | .05 | 3/1/0 | .5/.5 | .0399 | 1 | .9037 | .9880/.9869 | .9407/.9775/.8976 | 1/.2128 | .1813/.72 | 9/10 |
| 035717_s3_top_lr9e5 | 10 | 9e-5 | .05 | 3/1/0 | .5/.5 | .0173 | 1 | .9981 | .9956/.9951 | .9134/.9668/.8785 | 1/.2008 | .1558/.72 | 9/10 |
| 040107_s3_top_e20 | 20 | 3e-5 | .05 | 3/1/0 | .5/.5 | .0148 | 1 | .9981 | .9956/.9951 | .9125/.9663/.8779 | 1/.1915 | .1581/.72 | 9/10 |
| 040113_s3_top_e5 | 5 | 3e-5 | .05 | 3/1/0 | .5/.5 | .0283 | 1 | .9622 | .9938/.9933 | .9316/.9741/.8906 | 1/.2143 | .1689/.74 | 9/10 |
| 040458_s3_next_w0100 | 10 | 3e-5 | .10 | 1/1/0 | .5/.5 | .0232 | 1 | 1.0000 | .9951/.9945 | .9122/.9666/.8775 | 1/.1978 | .1561/.72 | 9/10 |
| 040503_s3_next_w0025 | 10 | 3e-5 | .025 | 1/1/0 | .5/.5 | .0092 | 1 | 1.0000 | .9951/.9946 | .9141/.9676/.8796 | 1/.2066 | .1530/.74 | 9/10 |

## Ranking and visualization review

Using the approved ordering (hard-gate count, then normalized violation), the top three were:

1. `040503_s3_next_w0025`: failed only event precision by `0.0043`; all ranking, GT, and success false-alarm gates passed.
2. `040113_s3_top_e5`: failed only success-frame FPR (`0.1689` versus `0.1590`).
3. `035717_s3_top_lr9e5`: failed only event precision (`0.2008`).

For each of these three, `vis_disc_finetuned.sh` produced the complete benchmark/offline PDFs and 30 sampled MP4s (10 failure, 10 success, 10 offline) under `<run>/visualization/val-seed0/`. The sampled plots/videos show the intended score direction in many regions, but they also fail the qualitative gate. In representative top-1 and high-LR plots, alarms begin 67--70 frames before the labeled failure; one sampled successful-terminal offline episode is alarmed for 474/477 frames. The top and high-LR models also produce fragmented or early failure segments, matching their low event precision. The five-epoch model is more conservative on event segmentation but has systematic held-out success false alarms above the allowed frame rate. No inspected candidate showed a global score-direction inversion, but obvious long-duration/early alarms reinforce rejection rather than override it.

## Failure analysis and likely causes

- **A stable Pareto trade-off appeared.** Lower learning rate or shorter training preserved event precision but left success FPR around `0.17-0.18`. Stronger adaptation reduced success FPR to about `0.15-0.16` and drove GT detection toward `1.0`, but split predictions into more event segments and lowered event precision.
- **nnPU was clamped for almost all training.** Most final clamp fractions are `1.00`. The negative-risk term therefore contributes no gradient over much of finetuning; optimization is dominated by positive risk plus supervised GT-BCE. This is consistent with rapid calibration/segmentation shifts and limited control over the parent decision geometry.
- **GT detection alone is insufficient.** The existing GT-weight-1.0 run achieved perfect train-set GT detection but degraded frame AUROC/AUPRC substantially. Larger GT weights similarly improve the local diagnostic while moving away from the parent benchmark.
- **Class weights mainly shift calibration.** The `0.25/0.75` and `0.1/0.9` variants did not resolve the joint gate. Because the threshold is recalibrated after training, a mostly uniform logit shift is largely cancelled and does not reliably improve event topology.
- **Window labels are sparse and imperfect.** Intervention-onset windows are proxy failure labels, include truncated boundaries, and cover a small subset of behavior. Short windows reduced label contamination and gave the best near-miss, but cannot fully constrain success behavior or segment continuity.
- **Frozen representation/head limitation is plausible.** With the encoder and discriminator method fixed, a single linear-head update and recalibrated scalar threshold may not have enough freedom to improve intervention-window recall while preserving parent event segmentation and success calibration simultaneously.

The most plausible attainable result under the fixed method is therefore the recorded 9/10 near-miss, not a verified improvement. Recommended follow-up experiments, requiring a new decision, are threshold selection with an explicit held-out success/event constraint, temporal hysteresis or minimum-segment post-processing, a loss term targeting success false alarms, or revisiting the nnPU clamp behavior. None was applied here because each changes the fixed evaluation or discriminator method.

## Implementation and artifact notes

- `eval_disc_checkpoint.py` and `eval_disc_finetuned.sh` provide CUDA-only, load-only parent/finetuned evaluation with checkpoint contract validation.
- `finetune_disc.sh` now treats YAML as the only source of default epochs, LR, weight decay, encode batch size, log interval, and seed; explicitly set environment variables remain overrides.
- Every benchmark JSON covers all 95 validation trajectories. Every success false-alarm JSON includes aggregate and per-trajectory valid-frame results.
- TensorBoard was the only logger; WandB was disabled.
- No CUDA OOM, CPU fallback, or missing/NaN aggregate metric occurred across the 50 new trials.
