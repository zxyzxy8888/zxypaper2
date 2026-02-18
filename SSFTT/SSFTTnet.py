from einops import rearrange
from torch import nn
import torch.nn.init as init
import torch
import torch.nn.functional as F


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


# 等于 PreNorm
class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# 等于 FeedForward
class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):

    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)  # Wq,Wk,Wv for each vector, thats why *3
        # torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        # torch.nn.init.zeros_(self.to_qkv.bias)

        self.nn1 = nn.Linear(dim, dim)
        # torch.nn.init.xavier_uniform_(self.nn1.weight)
        # torch.nn.init.zeros_(self.nn1.bias)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # split into multi head attentions

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper

        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block
        out = self.nn1(out)
        out = self.do1(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attention, mlp in self.layers:
            x = attention(x, mask=mask)  # go to attention
            x = mlp(x)  # go to MLP_Block
        return x


NUM_CLASS = 2


class ModifiedSSFTTnet(nn.Module):
    def __init__(self, in_channels=2, num_classes=2, num_tokens=4, dim=64, depth=1, heads=8, mlp_dim=8,
                 dropout=0.1, emb_dropout=0.1):
        super(ModifiedSSFTTnet, self).__init__()
        self.L = num_tokens
        self.cT = dim

        # 3D卷积特征提取
        self.conv3d_features = nn.Sequential(
            # 输入: B*3*128*128*128
            nn.Conv3d(in_channels, 16, kernel_size=3, stride=2, padding=1),  # -> B*16*64*64*64
            nn.BatchNorm3d(16),
            nn.ReLU(),

            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),  # -> B*32*32*32*32
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),  # -> B*64*16*16*16
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),  # -> B*128*8*8*8
            nn.BatchNorm3d(128),
            nn.ReLU(),
        )

        # 2D卷积特征融合
        self.conv2d_features = nn.Sequential(
            nn.Conv2d(128 * 8, 256, kernel_size=3, stride=1, padding=1),  # 保持空间维度
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),  # 输出通道降到64以匹配后续token处理
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Tokenization参数
        self.token_wA = nn.Parameter(torch.empty(1, self.L, 64))  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, 64, self.cT))  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV)

        # Transformer相关组件
        self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens + 1), dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Transformer
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_cls_token = nn.Identity()

        # 分类头
        self.pred1 = nn.Sequential(
            nn.Linear(dim, 128),
            nn.LayerNorm(128),
            nn.ReLU())
        self.pred2 =nn.Linear(128, num_classes)

    def forward(self, x, mask=None):
        B = x.shape[0]
        # 插值到128*128*128
        # x = F.interpolate(x, size=[128, 128, 128], mode='trilinear', align_corners=False)

        # 3D卷积特征提取
        x = self.conv3d_features(x)  # -> B*128*8*8*8

        # 重排张量用于2D处理
        x = rearrange(x, 'b c h w d -> b (c h) w d')  # 将channel和第一个空间维度合并

        # 2D卷积处理
        x = self.conv2d_features(x)  # -> B*64*8*8

        # 准备token化
        x = rearrange(x, 'b c h w -> b (h w) c')  # -> B*(8*8)*64

        # Token化处理
        wa = rearrange(self.token_wA, 'b h w -> b w h')
        A = torch.einsum('bij,bjk->bik', x, wa)
        A = rearrange(A, 'b h w -> b w h')
        A = A.softmax(dim=-1)

        VV = torch.einsum('bij,bjk->bik', x, self.token_wV)
        T = torch.einsum('bij,bjk->bik', A, VV)

        # 添加分类token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, T), dim=1)

        # 位置编码和Transformer处理
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x, mask)

        # 分类
        x = self.to_cls_token(x[:, 0])
        futures = self.pred1(x)
        x = self.pred2(futures)

        return x,futures