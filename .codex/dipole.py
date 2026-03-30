import copy
import random
from typing import Any, Dict, Tuple
import flax
import jax
import jax.numpy as jnp
from jax.nn import sigmoid
import ml_collections
import optax
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value

def f_func(value: jnp.ndarray, config: Dict[str, Any]) -> jnp.ndarray:
    """Compute advantage-based weighting function."""
    if config["f_func"] == "sigmoid":
        return sigmoid(config["beta"] * value + config["k"])
    else:
        raise NotImplementedError(f"Unknown f_func: {config['f_func']}")


def expectile_loss(diff: jnp.ndarray, expectile: float) -> jnp.ndarray:
    """Expectile regression loss."""
    weight = jnp.where(diff >= 0, expectile, 1.0 - expectile)
    return weight * diff ** 2


class DIPOLEAgent(flax.struct.PyTreeNode):
    """DIPOLE agent."""
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def value_loss(self, batch, params):
        """IQL-style expectile value loss."""
        qs = self.network.select("target_critic")(
            batch["observations"], actions=batch["actions"]
        )
        q = qs.min(axis=0) if self.config["q_agg"] == "min" else qs.mean(axis=0)

        v = self.network.select("value")(batch["observations"], params=params)
        loss = expectile_loss(q - v, self.config["expectile"]).mean()

        return loss, {
            "loss": loss,
            "v_mean": v.mean(),
            "v_max": v.max(),
            "v_min": v.min(),
        }

    def critic_loss(self, batch, params, rng):
        """TD critic loss using flow-based action sampling."""
        rng, act_rng = jax.random.split(rng)

        next_actions = self.sample_actions_fortrain(
            batch["next_observations"],
            seed=act_rng,
            cfg=self.config["cfg_fortrain"],
        )
        target_q = self.network.select("target_critic")(
            batch["next_observations"], actions=next_actions
        )

        q_target = batch["rewards"] + self.config["discount"] * batch["masks"] * target_q

        q1, q2 = self.network.select("critic")(
            batch["observations"], actions=batch["actions"], params=params
        )
        loss = ((q1 - q_target) ** 2 + (q2 - q_target) ** 2).mean()

        return loss, {
            "loss": loss,
            "q_mean": q_target.mean(),
            "q_max": q_target.max(),
            "q_min": q_target.min(),
        }

    def _flow_actor_loss(
        self,
        batch,
        params,
        rng,
        weight: jnp.ndarray,
        actor_name: str,
    ):
        """Flow-matching actor loss (shared by pos / neg actor)."""
        batch_size, action_dim = batch["actions"].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        x0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x1 = batch["actions"]
        t = jax.random.uniform(t_rng, (batch_size, 1))

        x_t = (1.0 - t) * x0 + t * x1
        vel = x1 - x0

        pred = self.network.select(actor_name)(
            batch["observations"], x_t, t, params=params
        )

        loss = ((pred - vel) ** 2).mean(axis=-1) * weight
        return loss.mean(), {"loss": loss.mean()}

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        v_loss, v_info = self.value_loss(batch, grad_params)
        info.update({f"value/{k}": v for k, v in v_info.items()})

        rng, critic_rng = jax.random.split(rng)
        c_loss, c_info = self.critic_loss(batch, grad_params, critic_rng)
        info.update({f"critic/{k}": v for k, v in c_info.items()})

        v = self.network.select("value")(batch["observations"])
        qs = self.network.select("target_critic")(
            batch["observations"], actions=batch["actions"]
        )
        q = qs.min(axis=0)
        value_pos = f_func(q - v, self.config)

        rng, a_rng = jax.random.split(rng)
        pos_loss, _ = self._flow_actor_loss(
            batch, grad_params, a_rng, value_pos, "actor_flow_pos"
        )
        neg_loss, _ = self._flow_actor_loss(
            batch, grad_params, a_rng, 1 - value_pos, "actor_flow_neg"
        )

        total_loss = v_loss + c_loss + pos_loss + neg_loss
        return total_loss, info

    @jax.jit
    def update(self, batch):
        rng, step_rng = jax.random.split(self.rng)

        def loss_fn(params):
            return self.total_loss(batch, params, step_rng)

        new_network, info = self.network.apply_loss_fn(loss_fn)
        self._soft_update(new_network, "critic")

        return self.replace(network=new_network, rng=rng), info

    def _soft_update(self, network, name: str):
        """Polyak update for target networks."""
        params = network.params
        params[f"modules_target_{name}"] = jax.tree_util.tree_map(
            lambda p, tp: self.config["tau"] * p + (1.0 - self.config["tau"]) * tp,
            params[f"modules_{name}"],
            params[f"modules_target_{name}"],
        )

    @jax.jit
    def sample_actions_fortrain(self, observations, seed, cfg=1.0):
        """Sample a single action per state for TD learning."""
        rng = seed
        obs_pos = obs_neg = observations

        if self.config["encoder"] is not None:
            obs_pos = self.network.select("actor_flow_pos_encoder")(observations)
            obs_neg = self.network.select("actor_flow_neg_encoder")(observations)

        rng, noise_rng = jax.random.split(rng)
        actions = jax.random.normal(
            noise_rng,
            (
                observations.shape[0],
                self.config["num_samples_fortrain"],
                self.config["action_dim"],
            ),
        )

        for i in range(self.config["flow_steps"]):
            t = jnp.full(
                (observations.shape[0], self.config["num_samples_fortrain"], 1),
                i / self.config["flow_steps"],
            )
            v_pos = self.network.select("actor_flow_pos")(obs_pos[:, None, :], actions, t, is_encoded=True)
            v_neg = self.network.select("actor_flow_neg")(obs_neg[:, None, :], actions, t, is_encoded=True)
            actions = actions + (v_neg + cfg * (v_pos - v_neg)) / self.config["flow_steps"]

        actions = jnp.clip(actions, -1.0, 1.0)
        q = self.network.select("critic")(observations[:, None, :], actions=actions).min(axis=0)

        best = jnp.argmax(q, axis=1)
        return actions[jnp.arange(actions.shape[0]), best]

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1, cfg = 1.):
        """Sample actions for environment interaction or evaluation."""
        orig_observations = observations
        observations_pos = observations_neg = observations
        if self.config['encoder'] is not None:
            observations_pos = self.network.select('actor_flow_pos_encoder')(observations)
            observations_neg = self.network.select('actor_flow_neg_encoder')(observations)
        action_seed, _ = jax.random.split(seed)

        # Sample `num_samples` noises and propagate them through the flow.
        actions = jax.random.normal(
            action_seed,
            (
                *observations.shape[:-1],
                self.config['num_samples'],
                self.config['action_dim'],
            ),
        )
        n_observations_pos = jnp.repeat(jnp.expand_dims(observations_pos, 0), self.config['num_samples'], axis=0)
        n_observations_neg = jnp.repeat(jnp.expand_dims(observations_neg, 0), self.config['num_samples'], axis=0)
        n_orig_observations = jnp.repeat(jnp.expand_dims(orig_observations, 0), self.config['num_samples'], axis=0)
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], self.config['num_samples'], 1), i / self.config['flow_steps'])
            vels_pos = self.network.select('actor_flow_pos')(n_observations_pos, actions, t, is_encoded=True)
            vels_neg = self.network.select('actor_flow_neg')(n_observations_neg, actions, t, is_encoded=True)
            vels = vels_neg + cfg * (vels_pos - vels_neg)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)

        # Pick the action with the highest Q-value.
        q = self.network.select('critic')(n_orig_observations, actions=actions).min(axis=0)
        actions = actions[jnp.argmax(q)]
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['value'] = encoder_module()
            encoders['critic'] = encoder_module()
            encoders['actor_flow_pos'] = encoder_module()
            encoders['actor_flow_neg'] = encoder_module()

        # Define networks.
        value_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=1,
            encoder=encoders.get('value'),
        )
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        actor_flow_pos_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_flow_pos'),
        )
        actor_flow_neg_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_flow_neg'),
        )

        network_info = dict(
            value=(value_def, (ex_observations,)),
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_flow_pos=(actor_flow_pos_def, (ex_observations, ex_actions, ex_times)),
            actor_flow_neg=(actor_flow_neg_def, (ex_observations, ex_actions, ex_times)),
        )
        if encoders.get('actor_flow_pos') is not None:
            # Add actor_flow_encoder to ModuleDict to make it separately callable.
            network_info['actor_flow_pos_encoder'] = (encoders.get('actor_flow_pos'), (ex_observations,))
            network_info['actor_flow_neg_encoder'] = (encoders.get('actor_flow_neg'), (ex_observations,))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_critic'] = params['modules_critic']

        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='dipole',  # Agent name.
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.995,  # Discount factor.
            tau=0.005,  # Target network update rate.
            expectile=0.9,  # IQL expectile.
            num_samples=32,  # Number of action samples for rejection sampling.
            num_samples_fortrain=1,  # Number of action samples for TD update.
            cfg_fortrain=1.,
            flow_steps=10,  # Number of flow steps.
            beta=1.0,  # Number of flow steps.
            k=0.0,  # Number of flow steps.
            f_func='sigmoid',
            q_agg='min',  # Aggregation method for target Q values.
            seed=random.randint(0, 1000000),
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config