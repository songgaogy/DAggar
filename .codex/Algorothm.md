# Algorithm

## Representation Layer
input observation $(o_t, l, a_t)$ first pass shared pretrained encoder from policy network:
$$
u_t = E_{\text{shared}}(o_{t-n:t}, l, p_{t-n:t-1})
$$
note that we may input chunk

then lead out three heads:
$$
h_t^{sh} = B(P_{sh}(u_t))
$$
$$
h_t^{occ,p} = P_{occ}(u_t)
$$
$$
h_t^{dyn,p,m} = P_{dyn}^{(m)}(u_t), \quad m=1,\dots,M
$$

here:
* $h_t^{sh}$：shared bottleneck representation
* $h_t^{occ,p}$：occupancy latent
* $h_t^{dyn,p,m}$：the $m$-th ensemble dynamics latents

## Occupancy 
Occupancy inputs are
$$
[h_t^{sh}, h_t^{occ,p}]
$$

define occupancy risk:
$$
s_t^{occ} = D_{occ}(h_t^{sh}, h_t^{occ,p})
$$


## Dynamics
for each ensemble member $m$, it predicts latent state after ground-truth action is given:
$$
(\hat h_{t,m}^{dyn}, \sigma_{t,m}^2) = M^{(m)}(h_{t-1}^{sh}, h_{t-1}^{dyn,p,m}, a_{t})
$$

then use EMA target branch to provide fair comparision:
$$
h_t^{dyn,\text{EMA}}
$$

1. difference: we use average over ensembles to predict
    $$
    \bar h_t^{dyn} = \frac{1}{M}\sum_{m=1}^M \hat h_{t,m}^{dyn}
    $$

    we define residual-based dynamics anomaly：
    $$
    s_t^{dyn} = \mathrm{Calib}
    \left(
    \left|\bar h_t^{dyn} - \mathrm{SG}(h_t^{dyn,\text{EMA}})
    \right|_2
    \right)
    $$

2. deviation: define ensemble disagreement：
    $$
    s_t^{epi} = \mathrm{Calib}
    \left(
    \frac{1}{M-1}\sum_{m=1}^{M}
    \left|
    \hat h_{t,m}^{dyn} - \bar h_t^{dyn}
    \right|^2
    \right)
    $$


## Calibration and Detachment
1. Stop-gradient
    $$
    e_t^{occ} = \mathrm{SG}(\mathrm{Calib}(s_t^{occ}))
    $$
    $$
    e_t^{dyn} = \mathrm{SG}(\mathrm{Calib}(s_t^{dyn}))
    $$
    $$
    e_t^{epi} = \mathrm{SG}(\mathrm{Calib}(s_t^{epi}))
    $$
2. Calibration: map raw score into comparable range:
    $$
    e_t^{occ}, e_t^{dyn}, e_t^{epi} \in [0,1]
    $$

## Judge / Fusion Layer
Input for Judge is three detached scalar evidence：
$$
e_t^{occ},\quad e_t^{dyn},\quad e_t^{epi}
$$

use a light-weight network to output threshold
$$
\hat y_t = F(e_t^{occ}, e_t^{dyn}, e_t^{epi})
$$

## Training Objective
The total loss of this is not a simple end-to-end black box loss, but rather **each witness is trained separately + fusion is trained separately**.

1. Occupancy witness loss:  
    using PU learning (BCE) on (expert data + success rollout) vs fail rollout, since fail rollout data still has success / correct transition:
    $$
    \mathcal L_{occ}
    $$

2. Dynamics witness loss
    $$
    \mathcal L_{dyn} = \mathcal L_{\beta\text{-NLL}}
    $$

3. shared-private decoupling auxiliary items:  
    decorrelation / covariance penalty on private branch, if one dimension of occ-private is highly linear-correlated with dyn-private in a batch, this loss will be large. Most simply, you can use cross-branch covariance matrix.
    $$
    \mathcal L_{decor}
    $$

4. Judge loss
    train fusion judge on detached evidence
    $$
    \mathcal L_{fuse}
    $$

Total loss:
$$
\mathcal L
=
\mathcal L_{occ}
+
\alpha \mathcal L_{dyn}
+
\eta \mathcal L_{decor}
+
\xi \mathcal L_{fuse}
$$

Note:
* `L_occ` trains the occupancy witness
* `L_dyn` trains the dynamics witness
* `L_fuse` only trains the judge
* Fusion does not reverse-engineer the witness

 