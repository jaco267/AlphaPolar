import torch
from typing import *
from torch import nn


class TranBlock(nn.Module):  #* if I use nn.Module  I got error I guess this is some jit bug  
    # def __init__(self):
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, mlp_ratio: int = 4):
        super(TranBlock,self).__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_ratio * embed_dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * embed_dim, embed_dim)
        )
    def forward(self, x):
        x = self.norm1(x)
        x = x + self.attn(x, x, x, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x

class AttentiveModels(nn.Module):
    def __init__(self, embed_dim: int = 256):
        super(AttentiveModels,self).__init__()

        self.block1 = TranBlock(embed_dim)
        self.block2 = TranBlock(embed_dim)
        self.block3 = TranBlock(embed_dim)

    def forward(self, grids: List[torch.Tensor]):#todo the annotation here is super important for jit  https://discuss.pytorch.org/t/error-with-my-custom-concat-class-with-torchscript/112440
        grids = [g.clone() for g in grids]
        '''
        for i, j, block in [(0, 1, self.block1), (2, 0, self.block2), (1, 2, self.block3)]:
            x = torch.cat((grids[i], grids[j].transpose(1, 2)), dim=2)  # (bs, S, 2S, c)
            # raise Exception('hihiahhaahhaaherror...') 
            x = block(x.flatten(0, 1)).unflatten(0, x.shape[:2])  # (bs, S, 2S, c)
            S = grids[i].shape[2]
            assert grids[j].shape[1] == S
            assert x.shape[2] == 2 * S
            grids[i] = x[:, :, :S]
            grids[j] = x[:, :, S:].transpose(1, 2)
        '''
        i=0; j=1;  block = self.block1
        x = torch.cat((grids[i], grids[j].transpose(1, 2)), dim=2)  # (bs, S, 2S, c)
        # raise Exception('hihiahhaahhaaherror...') 
        x = block(x.flatten(0, 1)).unflatten(0, x.shape[:2])  # (bs, S, 2S, c)
        S = grids[i].shape[2]
        assert grids[j].shape[1] == S
        assert x.shape[2] == 2 * S
        grids[i] = x[:, :, :S]
        grids[j] = x[:, :, S:].transpose(1, 2)
        #*------------------------------------------
        i=2; j=0;  block = self.block2
        x = torch.cat((grids[i], grids[j].transpose(1, 2)), dim=2)  # (bs, S, 2S, c)
        # raise Exception('hihiahhaahhaaherror...') 
        x = block(x.flatten(0, 1)).unflatten(0, x.shape[:2])  # (bs, S, 2S, c)
        S = grids[i].shape[2]
        assert grids[j].shape[1] == S
        assert x.shape[2] == 2 * S
        grids[i] = x[:, :, :S]
        grids[j] = x[:, :, S:].transpose(1, 2)
        #*------------------------------------------
        i=1; j=2;  block = self.block3
        x = torch.cat((grids[i], grids[j].transpose(1, 2)), dim=2)  # (bs, S, 2S, c)
        # raise Exception('hihiahhaahhaaherror...') 
        x = block(x.flatten(0, 1)).unflatten(0, x.shape[:2])  # (bs, S, 2S, c)
        S = grids[i].shape[2]
        assert grids[j].shape[1] == S
        assert x.shape[2] == 2 * S
        grids[i] = x[:, :, :S]
        grids[j] = x[:, :, S:].transpose(1, 2)
        #*------------------------------------------

        return grids

class Torso(nn.Module):
    def __init__(self, input_size: Tuple[int, int, int], embed_dim: int = 256, num_attn_models: int = 4):
        super().__init__()

        self.fc1 = nn.Linear(input_size[-1], embed_dim)
        self.fc2 = nn.Linear(input_size[-2], embed_dim)
        self.fc3 = nn.Linear(input_size[-3], embed_dim)

        self.attn_models = nn.ModuleList([AttentiveModels() for _ in range(num_attn_models)])

        self.input_size = input_size
        self.embed_dim = embed_dim

    def forward(self, s: torch.Tensor):
        assert s.ndim == 4
        assert s.shape[-3:] == self.input_size

        grids = self.fc1(s), self.fc2(s.permute((0, 3, 1, 2))), self.fc3(s.permute((0, 2, 3, 1)))
        for attn_models in self.attn_models:
            grids = attn_models(grids)

        e = torch.stack(grids, dim=1).flatten(1, -2)  # (bs, 3 * S * S, c)
        return torch.max(e, dim=1).values  # (bs, c), global max pool instead of cross attn



class PolicyHead(nn.Module):
    def __init__(self, num_actions: int, embed_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class ValueHead(nn.Module):
    def __init__(self, embed_dim: int = 256, hidden_dim: int = 256, num_quantiles: int = 8):
        super().__init__()

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_quantiles)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))  # (bs, num_quantiles)

