I already have a strong pretrained WAM-style latent representation. This latent space already contains action-related information, so the discriminator does not need to explicitly take raw actions as input. Instead of modeling the success distribution with a unimodal Gaussian density, I want to directly learn an expert-vs-other occupancy ratio in the latent space.

Let $z$ denote the pretrained latent representation of a robot frame or state-action context. Define two datasets:

- $D_e$: expert-like latent samples, including successful rollout frames and optionally the prefix of failed trajectories before the annotated failure point.
- $D_o$: other/failure-like latent samples, including the suffix of failed trajectories after the annotated failure point.

Train a binary discriminator d(z) using the objective:
$$min_d E_{z ~ D_e}[-log d(z)] + E_{z ~ D_o}[-log(1 - d(z))]$$

Equivalently, let the network output a logit $g(z)$, where $d(z) = sigmoid(g(z))$, and train with the numerically stable loss:
$$ L = E_{z ~ D_e}[softplus(-g(z))] + E_{z ~ D_o}[softplus(g(z))] $$

The theoretical motivation is that the optimal discriminator satisfies:
$$ d_(z) = rho_e(z) / (rho_e(z) + rho_o(z)) $$

and therefore its logit estimates the latent occupancy density ratio:
$$g_(z) = log rho_e(z) - log rho_o(z)$$

Thus, g(z) can be interpreted as an expert-likeness reward, while $-g(z)$ can be interpreted as a failure cost.

Please help me turn this idea into a concrete method for robot failure detection. Discuss the theoretical formulation, dataset construction, model architecture, loss function, reward interpretation, calibration strategy, evaluation protocol, and possible extensions such as KNN-style local structure, prototype-based heads, or distillation from a two-bank KNN baseline.