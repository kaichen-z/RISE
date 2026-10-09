import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
import pdb

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)
        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb), nn.Linear(d_model, dim),
            Swish(), nn.Linear(dim, dim),)
        self.initialize()
    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
    def forward(self, t):
        emb = self.timembedding(t)
        return emb

class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()
    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)
    def forward(self, x, temb):
        x = self.main(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()
    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)
    def forward(self, x, temb):
        _, _, H, W = x.shape
        x = F.interpolate(
            x, scale_factor=2, mode='nearest')
        x = self.main(x)
        return x

class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()
    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)
        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)
        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)
        return x + h

class ResBlock_nor(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch), Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),)
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch), Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),)
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()
    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)
    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        h = h + self.shortcut(x)
        h = self.attn(h)
        return h

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes,in_planes // ratio, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(in_planes // ratio, in_planes, bias = False))
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    def forward(self, in_feature):
        x = in_feature
        b, c, _, _ = in_feature.size()
        avg_out = self.fc(self.avg_pool(x).view(b,c)).view(b, c, 1, 1)
        out = avg_out
        return self.sigmoid(out).expand_as(in_feature) * in_feature
    
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),)
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),)
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),)
        if in_ch != out_ch: self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else: self.shortcut = nn.Identity()
        if attn: self.attn = AttnBlock(out_ch)
        else: self.attn = nn.Identity()
        self.initialize()
    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)
    def forward(self, x, temb):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)
        h = h + self.shortcut(x)
        h = self.attn(h)
        return h

class UNet(nn.Module):
    def __init__(self, T, input_ch, ch, ch_mult, attn, num_res_blocks, dropout, image_cond1=False, image_cond2=False):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)
        self.image_cond1 = image_cond1; self.image_cond2 = image_cond2
        self.image_ch = 16
        if self.image_cond1 and not self.image_cond2:
            self.cond_conv_in_cond1 = nn.Conv2d(in_channels=input_ch, out_channels=self.image_ch, kernel_size=1, bias=False)
            self.head = nn.Conv2d(input_ch + self.image_ch, ch, kernel_size=3, padding=1)
            self.channel = ChannelAttention(ch)
        elif self.image_cond1 and self.image_cond2:
            self.cond_conv_in_cond1 = nn.Conv2d(in_channels=input_ch, out_channels=self.image_ch, kernel_size=1, bias=False)
            self.cond_conv_in_cond2 = nn.Conv2d(in_channels=input_ch, out_channels=self.image_ch, kernel_size=1, bias=False)
            self.head = nn.Conv2d(input_ch + self.image_ch + self.image_ch, ch, kernel_size=3, padding=1)
            self.channel = ChannelAttention(ch)
        elif not self.image_cond1 and not self.image_cond2: 
            self.head = nn.Conv2d(input_ch, ch, kernel_size=3, stride=1, padding=1)
            self.channel = ChannelAttention(ch)
        # (the `github` release also built a `condition_proj` ModuleList here; it was
        #  never referenced in forward(), so it only cost parameters and optimiser state)
        self.downblocks = nn.ModuleList(); chs = [ch]; now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(in_ch=now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout, attn=(i in attn)))
                now_ch = out_ch; chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch)); chs.append(now_ch)
        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),])
        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0
        self.tail = nn.Sequential(nn.GroupNorm(32, now_ch), Swish(),
            nn.Conv2d(now_ch, input_ch, 3, stride=1, padding=1))
        self.initialize()
    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)
    def forward(self, x, t, cond_input1=None, cond_input2=None):
        # cond -> Observation 
        # cond -> Prediction
        temb = self.time_embedding(t)
        if self.image_cond1 and not self.image_cond2:
            cond_output1 = self.cond_conv_in_cond1(cond_input1)
            h = torch.cat([x, cond_output1], dim=1)
            h = self.channel(self.head(h))
        elif self.image_cond1 and self.image_cond2:
            cond_output1 = self.cond_conv_in_cond1(cond_input1)
            cond_output2 = self.cond_conv_in_cond2(cond_input2)
            h = torch.cat([x, cond_output1, cond_output2], dim=1)
            h = self.channel(self.head(h))
        elif not self.image_cond1 and not self.image_cond2: 
            h = self.channel(self.head(x))
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb); hs.append(h)
        for layer in self.middleblocks: h = layer(h, temb) 
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        h = self.tail(h)
        assert len(hs) == 0
        return h

if __name__ == '__main__':
    batch_size = 8
    model = UNet(
        T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
        num_res_blocks=2, dropout=0.1)
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(1000, (batch_size, ))
    y = model(x, t)