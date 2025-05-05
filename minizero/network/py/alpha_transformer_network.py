import torch
import torch.nn as nn
import torch.nn.functional as F
from .network_unit import ResidualBlock, PolicyNetwork, ValueNetwork, DiscreteValueNetwork
from .network_transformer import Torso,PolicyHead,ValueHead

class AlphaTransNetwork(nn.Module):
    def __init__(self,
                 game_name,
                 num_input_channels,
                 input_channel_height,
                 input_channel_width,
                 num_hidden_channels,
                 hidden_channel_height,
                 hidden_channel_width,
                 num_blocks,
                 action_size,
                 num_value_hidden_channels,
                 discrete_value_size,
                 network_type= "transformer"): #* conv / transformer
        super(AlphaTransNetwork, self).__init__()
        self.game_name = game_name
        self.num_input_channels = num_input_channels
        self.input_channel_height = input_channel_height
        self.input_channel_width = input_channel_width
        self.num_hidden_channels = num_hidden_channels
        self.hidden_channel_height = hidden_channel_height
        self.hidden_channel_width = hidden_channel_width
        self.num_blocks = num_blocks
        self.action_size = action_size
        self.num_value_hidden_channels = num_value_hidden_channels
        self.discrete_value_size = discrete_value_size
        self.network_type = network_type
        #* transformer
        self.input_shape  = (hidden_channel_height,hidden_channel_width,hidden_channel_height)
        # print("....",hidden_channel_height,hidden_channel_width,hidden_channel_height)
        self.torso = Torso(self.input_shape,num_hidden_channels,num_attn_models=1) #num_blocks 
        # self.torso = Torso((4,4,4),256,num_attn_models=1) 
        #     #*(16,4,4), 256
        self.policy_head = PolicyHead(action_size,num_hidden_channels,num_hidden_channels)
        self.value_head = ValueHead(num_hidden_channels,num_hidden_channels,discrete_value_size)
        
    @torch.jit.export
    def get_type_name(self):return "alphazero"
    @torch.jit.export
    def get_game_name(self):return self.game_name
    @torch.jit.export
    def get_num_input_channels(self):return self.num_input_channels
    @torch.jit.export
    def get_input_channel_height(self):return self.input_channel_height
    @torch.jit.export
    def get_input_channel_width(self):return self.input_channel_width
    @torch.jit.export
    def get_num_hidden_channels(self):return self.num_hidden_channels
    @torch.jit.export
    def get_hidden_channel_height(self):return self.hidden_channel_height
    @torch.jit.export
    def get_hidden_channel_width(self):return self.hidden_channel_width
    @torch.jit.export
    def get_num_blocks(self):return self.num_blocks
    @torch.jit.export
    def get_action_size(self):return self.action_size
    @torch.jit.export
    def get_num_value_hidden_channels(self):return self.num_value_hidden_channels
    @torch.jit.export
    def get_discrete_value_size(self):return self.discrete_value_size
    def forward(self, state):
      assert state.shape[-3:] == self.input_shape
      e = self.torso(state) #*(1024,16,44) -> (1024,256)  
      policy_logit = self.policy_head(e)
      # policy
      policy = torch.softmax(policy_logit, dim=1)
      # value
      assert self.discrete_value_size != 1
      value_logit = self.value_head(e)
      value = torch.softmax(value_logit, dim=1)
      return {"policy_logit": policy_logit,
              "policy": policy,
              "value_logit": value_logit,
              "value": value}