import torch
import torch.nn as nn
from einops import rearrange
import math



class DepMHSA(nn.Module):
    def __init__(self, n_dims, n_frames=2, width=14, height=14, heads=4, pos_enc = True, q_kernel='133',k_kernel='311',v_kernel='311133'):
        super(DepMHSA, self).__init__()
        self.scale = (n_dims // heads) ** -0.5
        self.heads = heads
        self.pos_enc = pos_enc
        q_kernel = [int(x) for x in list(q_kernel)]
        k_kernel = [int(x) for x in list(k_kernel)]
        v_kernel = [int(x) for x in list(v_kernel)]

        self.query = nn.Conv3d(in_channels=n_dims, out_channels=n_dims, kernel_size=(q_kernel[0], q_kernel[1], q_kernel[2]), stride=1, padding='same',
                            bias=False)
        self.key = nn.Conv3d(in_channels=n_dims, out_channels=n_dims, kernel_size=(k_kernel[0], k_kernel[1], k_kernel[2]), stride=1, padding='same',
                            bias=False)
        self.value = nn.Sequential(nn.Conv3d(in_channels=n_dims, out_channels=n_dims, kernel_size=(v_kernel[0], v_kernel[1], v_kernel[2]), stride=1, padding='same',
                            bias=False),
                                nn.Conv3d(in_channels=n_dims, out_channels=n_dims, kernel_size=(v_kernel[3], v_kernel[4], v_kernel[5]), stride=1, padding='same',
                            bias=False))
        
        # 1, h, dim, 1, height, 1
        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height, 1]), requires_grad=True)
        # 1, h, dim, 1, 1, w
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, 1, width]), requires_grad=True)
        # 1, h, dim , f, 1, 1
        self.rel_t = nn.Parameter(torch.randn([1, heads, n_dims // heads, n_frames, 1, 1]), requires_grad=True)
        
        # 1, h, dim, 1, height, 1
        self.rel_h_2 = nn.Parameter(torch.randn([1, n_dims, 1, height, 1]), requires_grad=True)
        # 1, h, dim, 1, 1, w
        self.rel_w_2 = nn.Parameter(torch.randn([1, n_dims, 1, 1, width]), requires_grad=True)
        # 1, h, dim , f, 1, 1
        self.rel_t_2 = nn.Parameter(torch.randn([1, n_dims, n_frames, 1, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, F, height, width = x.size()        
        q = self.query(x).reshape(n_batch, self.heads, F, C // self.heads, -1) # B, Head, F, C, HW
        k = self.key(x).reshape(n_batch, self.heads, F, C // self.heads, -1)
        v = self.value(x).reshape(n_batch, self.heads, F, C // self.heads, -1)
        
        qk_weights = torch.matmul(q.permute(0, 1, 2, 4, 3), k) # B Head F C HW -> B Head F H*W C x B Head F C H*W -> B Head F H*W * H*W
        if self.pos_enc:
            # 1, Head, dim, F, H, W = 1, 4, 128, 2, 12, 12
            relative_positions = self.rel_h + self.rel_w + self.rel_t
            
            # 1, Head, dim, F, H*W -> 1, Head, F, H*W, dim | 1, 4, 128, 2, 1024 -> 1, 4, 2, 144, 128
            multihead_relative_positions = relative_positions.reshape(1, self.heads, C // self.heads, F, -1).permute(0, 1, 3, 4, 2)
            multihead_q_times_pos = torch.matmul(multihead_relative_positions, q)

            # # content_content: B, h, F, H, W 4, 4, 2, 256, 256
            energy = (qk_weights + multihead_q_times_pos) * self.scale
        else:
            energy = qk_weights * self.scale
            
        attention = self.softmax(energy)
        mhsa_out = torch.matmul(v, attention.permute(0, 1, 2, 4, 3)) # B Head F C H*W x B head F H*W H*W -> B Head F C H*W
        # out = out.reshape(n_batch, C, F, height, width)
        mhsa_out = mhsa_out.reshape(n_batch, C, F, height, width)
        out = mhsa_out + (self.rel_h_2 + self.rel_t_2 + self.rel_w_2)
        return out
    
class MHSA3D(nn.Module):
    "modified from https://github.com/SanoScience/BabyNet"
    def __init__(self, n_dims, n_frames=2, width=14, height=14, heads=4, pos_enc = True):
        super(MHSA3D, self).__init__()
        self.scale = (n_dims // heads) ** -0.5
        self.heads = heads
        self.pos_enc = pos_enc
        self.query = nn.Conv3d(in_channels=n_dims, out_channels=n_dims, kernel_size=(1, 1, 1), stride=1, padding=0,
                               bias=False)
        self.key = nn.Conv3d(in_channels=n_dims, out_channels=n_dims, kernel_size=(1, 1, 1), stride=1, padding=0,
                             bias=False)
        self.value = nn.Conv3d(in_channels=n_dims, out_channels=n_dims, kernel_size=(1, 1, 1), stride=1, padding=0,
                               bias=False)
        # 1, h, dim, 1, height, 1
        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height, 1]), requires_grad=True)
        # 1, h, dim, 1, 1, w
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, 1, width]), requires_grad=True)
        # 1, h, dim , f, 1, 1
        self.rel_t = nn.Parameter(torch.randn([1, heads, n_dims // heads, n_frames, 1, 1]), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # cab_output = self.cab(x)
        n_batch, C, F, height, width = x.size()
        q = self.query(x).reshape(n_batch, self.heads, F, C // self.heads, -1) # B, Head, F, C, H*W
        k = self.key(x).reshape(n_batch, self.heads, F, C // self.heads, -1) # B, Head, F, C, H*W
        v = self.value(x).reshape(n_batch, self.heads, F, C // self.heads, -1) # B, Head, F, C, H*W

        spatio_attention_QK = torch.matmul(q.permute(0, 1, 2, 4, 3), k) # B Head F C HW -> B Head F H*W C x B Head F C H*W -> B Head F H*W * H*W
        
        if self.pos_enc:
            content_rel_pos = self.rel_h + self.rel_w + self.rel_t
            content_position = content_rel_pos.reshape(1, self.heads, C // self.heads, F, -1).permute(0, 1, 3, 4, 2)
            content_position2 = torch.matmul(content_position, q)
            energy = (spatio_attention_QK + content_position2) * self.scale
        else:
            energy = spatio_attention_QK * self.scale
        attention = self.softmax(energy)
        mhsa_out = torch.matmul(v, attention.permute(0, 1, 2, 4, 3)) # B Head F C H*W x B head F H*W H*W -> B Head F C H*W
        mhsa_out = mhsa_out.reshape(n_batch, C, F, height, width)
        out = mhsa_out
        return out
    
    
class SingleModalQKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    1 dim data
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        H = head
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)
    
    
class SingleModalAtten(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=4,
        num_head_channels=-1,
        use_checkpoint=False,
        normalization_type = 'groupnorm'
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        if normalization_type == 'groupnorm':
            self.norm = normalization(channels)
        else:
            self.norm = nn.LayerNorm(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = SingleModalQKVAttention(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return x + h.reshape(b, c, *spatial)