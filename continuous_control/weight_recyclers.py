# coding=utf-8
# Copyright 2023 ReDo authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This class implements recycling of dead neurons."""
import functools
import logging
import flax
from flax import linen as nn
import gin
import jax
from jax import random
import jax.numpy as jnp
import optax



def reset_momentum(momentum, mask):
  new_momentum = momentum if mask is None else momentum * (1.0 - mask)
  return new_momentum



def weight_reinit_zero(param, mask):
  if mask is None:
    return param
  else:
    new_param = jnp.zeros_like(param)
    param = jnp.where(mask == 1, new_param, param)
    return param



def weight_reinit_random(param,
                         mask,
                         key,
                         weight_scaling=False,
                         scale=1.0,
                         weights_type='incoming',
                         sp_weight=0.5):
  """Randomly reinit recycled weights and may scale its norm.

  If scaling applied, the norm of recycled weights equals
  the average norm of non recycled weights per neuron multiplied by a scalar.

  Args:
    param: current param
    mask: incoming/outgoing mask for recycled weights
    key: random key to generate new random weights
    weight_scaling: if true scale recycled weights with the norm of non recycled
    scale: scale to multiply the new weights norm.
    weights_type: incoming or outgoing weights

  Returns:
  params: new params after weight recycle.
  """
  if mask is None or key is None:
    return param

  new_param = nn.initializers.xavier_uniform()(key, shape=param.shape)

  if weight_scaling:
    axes = list(range(param.ndim))
    if weights_type == 'outgoing':
      del axes[-2]
    else:
      del axes[-1]

    neuron_mask = jnp.mean(mask, axis=axes)

    non_dead_count = neuron_mask.shape[0] - jnp.count_nonzero(neuron_mask)
    norm_per_neuron = _get_norm_per_neuron(param, axes)
    non_recycled_norm = jnp.sum(norm_per_neuron *
                                (1 - neuron_mask)) / non_dead_count
    non_recycled_norm = non_recycled_norm * scale

    normalized_new_param = _weight_normalization_per_neuron_norm(
        new_param, axes)
    new_param = normalized_new_param * non_recycled_norm

  param = jnp.where(mask == 1, (1-sp_weight) * new_param + sp_weight * param, param)
  return param



def _weight_normalization_per_neuron_norm(param, axes):
  norm_per_neuron = _get_norm_per_neuron(param, axes)
  norm_per_neuron = jnp.expand_dims(norm_per_neuron, axis=axes)
  normalized_param = param / norm_per_neuron
  return normalized_param



def _get_norm_per_neuron(param, axes):
  return jnp.sqrt(jnp.sum(jnp.power(param, 2), axis=axes))



@functools.partial(jax.jit, static_argnames=('dead_neurons_threshold'))
def score2mask(activation, dead_neurons_threshold):
  # del key, param, next_param
  reduce_axes = list(range(activation.ndim - 1))
  score = jnp.mean(jnp.abs(activation), axis=reduce_axes)
  # Normalize so that all scores sum to one.
  score /= jnp.mean(score) + 1e-9
  return score <= dead_neurons_threshold



@jax.jit
def create_mask_helper(neuron_mask, current_param, next_param):
  """generate incoming and outgoing weight mask given dead neurons mask.

  Args:
    neuron_mask: mask of size equals the width of a layer.
    current_param: incoming weights of a layer.
    next_param: outgoing weights of a layer.

  Returns:
    incoming_mask
    outgoing_mask
  """
  def mask_creator(expansion_axis, expansion_axes, param, neuron_mask):
    """create a mask of weight matrix given 1D vector of neurons mask.

    Args:
      expansion_axis: List contains 1 axis. The dimension to expand the mask
        for dense layers (weight shape 2D).
      expansion_axes: List conrtains 3 axes. The dimensions to expand the
        score for convolutional layers (weight shape 4D).
      param: weight.
      neuron_mask: 1D mask that represents dead neurons(features).

    Returns:
      mask: mask of weight.
    """
    if param.ndim == 2:
      axes = expansion_axis
      # flatten layer
      # The size of neuron_mask is the same as the width of last conv layer.
      # This conv layer will be flatten and connected to dense layer.
      # we repeat each value of a feature map to cover the spatial dimension.
      if axes[0] == 1 and (param.shape[0] > neuron_mask.shape[0]):
        num_repeatition = int(param.shape[0] / neuron_mask.shape[0])
        neuron_mask = jnp.repeat(neuron_mask, num_repeatition, axis=0)
    elif param.ndim == 4:
      axes = expansion_axes
    mask = jnp.expand_dims(neuron_mask, axis=tuple(axes))
    for i in range(len(axes)):
      mask = jnp.repeat(mask, param.shape[axes[i]], axis=axes[i])
    return mask

  incoming_mask = mask_creator([0], [0, 1, 2], current_param, neuron_mask)
  outgoing_mask = mask_creator([1], [0, 1, 3], next_param, neuron_mask)
  return incoming_mask, outgoing_mask



# @jax.jit
# def compute_effective_rank(intermediates):
#   activations_dict = flax.traverse_util.flatten_dict(
#         intermediates, sep='/')
#   activation = activations_dict['projection/net_act/__call__'][0]
#   sv = jnp.linalg.svd(activation, compute_uv=False)
#   norm_sv = sv / jnp.sum(jnp.abs(sv))
#   entropy = 0
#   for p in norm_sv:
#       res = jax.lax.cond(p > 0.0, lambda x: x * jnp.log(x), lambda x: 0.0, p)
#       entropy -= res

#   effective_rank = jnp.e ** entropy
#   return effective_rank



@functools.partial(
    jax.jit,
    static_argnames=(
      'dead_neurons_threshold',
      'init_method_outgoing',
    )
)
def create_masks(
  param_dict,
  activations_dict,
  key,
  # current_count,
  total_count,
  dead_neurons_threshold,
  init_method_outgoing,
  ):
  reset_layers = ['DoubleCritic_0/Critic_0/MLP_0/Dense_0', 
                  'DoubleCritic_0/Critic_1/MLP_0/Dense_0',
                  ]
  next_layers = {'DoubleCritic_0/Critic_0/MLP_0/Dense_0': 'DoubleCritic_0/Critic_0/MLP_0/Dense_1',
                 'DoubleCritic_0/Critic_0/MLP_0/Dense_1': 'DoubleCritic_0/Critic_0/MLP_0/Dense_2',
                 'DoubleCritic_0/Critic_1/MLP_0/Dense_0': 'DoubleCritic_0/Critic_1/MLP_0/Dense_1',
                 'DoubleCritic_0/Critic_1/MLP_0/Dense_1': 'DoubleCritic_0/Critic_1/MLP_0/Dense_2',}
  incoming_mask_dict = {
      k: jnp.zeros_like(p) if p.ndim != 1 else None
      for k, p in param_dict.items()
  }
  outgoing_mask_dict = {
      k: jnp.zeros_like(p) if p.ndim != 1 else None
      for k, p in param_dict.items()
  }
  ingoing_random_keys_dict = {k: None for k in param_dict}
  outgoing_random_keys_dict = {
      k: None for k in param_dict
  } if init_method_outgoing == 'random' else {}

  # prepare mask of incoming and outgoing recycled connections
  for k in reset_layers:
    param_key = k + '/kernel'
    param = param_dict[param_key]
    # This won't work for DRQ, since returned keys can be a list.
    # We don't support that at the moment.
    next_key = next_layers[k]
    if isinstance(next_key, list):
      next_key = next_key[0]
    next_param = param_dict[next_key + '/kernel']
    activation = activations_dict[k + '_act/__call__'][0]
    # TODO(evcu) Maybe use per_layer random keys here.
    neuron_mask = score2mask(activation, dead_neurons_threshold)

    # current_count[k] += neuron_mask
    total_count[k] += neuron_mask
    # neuron_mask = jnp.where(current_count[k] >= 5, True, False)
    # current_count[k] = jnp.where(neuron_mask, 0, current_count[k])

    # the for loop handles the case where a layer has multiple next layers
    # like the case in DrQ where the output layer has multihead.
    next_keys = (
        next_layers[k]
        if isinstance(next_layers[k], list) else [next_layers[k]])
    for next_k in next_keys:
      next_param_key = next_k + '/kernel'
      next_param = param_dict[next_param_key]
      incoming_mask, outgoing_mask = create_mask_helper(
          neuron_mask, param, next_param)
      incoming_mask_dict[param_key] = incoming_mask
      outgoing_mask_dict[next_param_key] = outgoing_mask
      key, subkey = random.split(key)
      ingoing_random_keys_dict[param_key] = subkey
      if init_method_outgoing == 'random':
        key, subkey = random.split(key)
        outgoing_random_keys_dict[next_param_key] = subkey

    # reset bias
    bias_key = k + '/bias'
    new_bias = jnp.zeros_like(param_dict[bias_key])
    param_dict[bias_key] = jnp.where(neuron_mask, new_bias,
                                      param_dict[bias_key])

  return (incoming_mask_dict, outgoing_mask_dict, ingoing_random_keys_dict,
          outgoing_random_keys_dict, param_dict, total_count)



@functools.partial(
  jax.jit,
  static_argnames=(
    'dead_neurons_threshold',
    'init_method_outgoing',
  )
)
def jit_rsp(
  model,
  intermediates,
  rng,
  # current_count,
  total_count,
  dead_neurons_threshold,
  init_method_outgoing,
  sp_weight,
):
  activations_score_dict = flax.traverse_util.flatten_dict(
        intermediates, sep='/')
  param_dict = flax.traverse_util.flatten_dict(model.params, sep='/')
  # create incoming and outgoing masks and reset bias of dead neurons.
  (
      incoming_mask_dict,
      outgoing_mask_dict,
      incoming_random_keys_dict,
      outgoing_random_keys_dict,
      param_dict,
      # current_count,
      total_count,
  ) = create_masks(
      param_dict,
      activations_score_dict,
      rng,
      # current_count,
      total_count,
      dead_neurons_threshold,
      init_method_outgoing,)
  
  params = flax.core.freeze(
      flax.traverse_util.unflatten_dict(param_dict, sep='/'))
  incoming_random_keys = flax.core.freeze(
      flax.traverse_util.unflatten_dict(incoming_random_keys_dict, sep='/'))
  if init_method_outgoing == 'random':
    outgoing_random_keys = flax.core.freeze(
        flax.traverse_util.unflatten_dict(outgoing_random_keys_dict, sep='/'))
  # reset incoming weights
  incoming_mask = flax.core.freeze(
      flax.traverse_util.unflatten_dict(incoming_mask_dict, sep='/'))
  reinit_fn = functools.partial(
      weight_reinit_random,
      weight_scaling=False,
      scale=1,
      weights_type='incoming',
      sp_weight=sp_weight)
  weight_random_reset_fn = jax.jit(functools.partial(jax.tree_map, reinit_fn))
  params = weight_random_reset_fn(params, incoming_mask, incoming_random_keys)

  # reset outgoing weights
  outgoing_mask = flax.core.freeze(
      flax.traverse_util.unflatten_dict(outgoing_mask_dict, sep='/'))
  if init_method_outgoing == 'random':
    reinit_fn = functools.partial(
        weight_reinit_random,
        weight_scaling=False,
        scale=1,
        weights_type='outgoing',
        sp_weight=sp_weight)
    weight_random_reset_fn = jax.jit(
        functools.partial(jax.tree_map, reinit_fn))
    params = weight_random_reset_fn(params, outgoing_mask,
                                    outgoing_random_keys)
  elif init_method_outgoing == 'zero':
    weight_zero_reset_fn = jax.jit(
        functools.partial(jax.tree_map, weight_reinit_zero))
    params = weight_zero_reset_fn(params, outgoing_mask)
  # else:
  #   raise ValueError(f'Invalid init method: {self.init_method_outgoing}')
  # reset mu, nu of adam optimizer for recycled weights.
  reset_momentum_fn = jax.jit(functools.partial(jax.tree_map, reset_momentum))
  opt_state = model.opt_state_head
  incoming_mask, _ = incoming_mask.pop('SharedEncoder')
  outgoing_mask, _ = outgoing_mask.pop('SharedEncoder')
  new_mu = reset_momentum_fn(opt_state[0][1], incoming_mask)
  new_mu = reset_momentum_fn(new_mu, outgoing_mask)
  new_nu = reset_momentum_fn(opt_state[0][2], incoming_mask)
  new_nu = reset_momentum_fn(new_nu, outgoing_mask)
  opt_state_list = list(opt_state)
  opt_state_list[0] = optax.ScaleByAdamState(
      opt_state[0].count, mu=new_mu, nu=new_nu)
  opt_state = tuple(opt_state_list)
  return params, opt_state, total_count



@jax.jit
def get_intermediates(model, batch):
  # TODO(gsokar) add a check if batch_size equals batch_size_statistics
  # then no need to sample a new batch from buffer.
  filter_rep = lambda l, _: l.name is not None and 'act' in l.name
  _, state = model(batch.observations, 
                  batch.actions,        
                  capture_intermediates=filter_rep,
                  mutable=['intermediates'])
  return state['intermediates']



# @jax.jit
# def get_intermediates2(model, batch):
#   # TODO(gsokar) add a check if batch_size equals batch_size_statistics
#   # then no need to sample a new batch from buffer.
#   filter_rep = lambda l, _: l.name is not None and 'act' in l.name
#   _, critic_state = model.critic(batch.observations, 
#                                 batch.actions,        
#                                 capture_intermediates=filter_rep,
#                                 mutable=['intermediates'])
#   _, actor_state = model.actor(batch.observations,
#                                capture_intermediates=filter_rep,
#                                mutable=['intermediates'])
#   return critic_state['intermediates'], actor_state['intermediates']



def estimate_neuron_score(activation, sub_mean_score=True, is_cbp=False):
  """Calculates neuron score based on absolute value of activation.

  The score of feature map is the normalized average score over
  the spatial dimension.

  Args:
    activation: intermediate activation of each layer
    is_cbp: if true, subtracts the mean and skips normalization.

  Returns:
    element_score: score of each element in feature map in the spatial dim.
    neuron_score: score of feature map
  """
  reduce_axes = list(range(activation.ndim - 1))
  if sub_mean_score or is_cbp:
    activation = activation - jnp.mean(activation, axis=reduce_axes)

  score = jnp.mean(jnp.abs(activation), axis=reduce_axes)
  if not is_cbp:
    # Normalize so that all scores sum to one.
    score /= jnp.mean(score) + 1e-9

  return score



def log_dead_neurons_count(intermediates, threshold=0.0):
  """log dead neurons in each layer.

  For conv layer we also log dead elements in the spatial dimension.

  Args:
    intermediates: intermidate activation in each layer.

  Returns:
    log_dict_elements_per_neuron
    log_dict_neurons
  """

  def log_dict(score, score_type):
    total_neurons, total_deadneurons = 0., 0.
    score_dict = flax.traverse_util.flatten_dict(score, sep='/')

    log_dict = {}
    for k, m in score_dict.items():
      if 'final_layer' in k:
        continue
      m = m[0]
      layer_size = float(jnp.size(m))
      deadneurons_count = jnp.count_nonzero(m <= threshold)
      total_neurons += layer_size
      total_deadneurons += deadneurons_count
      log_dict[f'dead_{score_type}_percentage/{k[:-9]}'] = (
          float(deadneurons_count) / layer_size) * 100.
      log_dict[f'dead_{score_type}_count/{k[:-9]}'] = float(deadneurons_count)
    log_dict[f'{score_type}/total'] = total_neurons
    log_dict[f'{score_type}/deadcount'] = float(total_deadneurons)
    log_dict[f'dead_{score_type}_percentage'] = (float(total_deadneurons) /
                                                  total_neurons) * 100.
    return log_dict

  neuron_score = jax.tree_map(estimate_neuron_score, intermediates)
  log_dict_neurons = log_dict(neuron_score, 'feature')

  return log_dict_neurons


import csv
def log_stats(intermediates, threshold, base_dir):
  log_dict = log_dead_neurons_count(intermediates, threshold)
  if log_dict is None:
    return
  stats = []
  for k, v in log_dict.items():
    if 'percentage' in k:
      stats.append(v)
  with open(base_dir+'/inter.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(stats)