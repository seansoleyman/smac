from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gym import spaces
import torch
from torch import nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.torch_ops import FLOAT_MIN
from ray.rllib.utils.torch_ops import FLOAT_MAX


class MaskedActionsModel(TorchModelV2, nn.Module):
    """Custom RLlib model that emits -inf logits for invalid actions.

    This is used to handle the variable-length StarCraft action space.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        obs_len = obs_space.shape[0]-action_space.n
        orig_obs_space = spaces.Box(shape=(obs_len,), low=obs_space.low[:obs_len], high=obs_space.high[:obs_len])

        self.action_embed_model = TorchFC(orig_obs_space, action_space, action_space.n, model_config, name + "_action_embed")


    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the unmasked action logits.
        action_logits, _ = self.action_embed_model({
            "obs": input_dict["obs"]["obs"]
        })

        # Perform masking.
        #inf_mask = torch.clamp(torch.log(action_mask), -1e10, FLOAT_MAX)
        #return action_logits + inf_mask, state
        action_mask_boolean = action_mask > 0.5
        masked_value = torch.tensor(FLOAT_MIN)
        masked_logits = torch.where(action_mask_boolean, action_logits, masked_value)
        return masked_logits, state


    def value_function(self):
        return self.action_embed_model.value_function()
