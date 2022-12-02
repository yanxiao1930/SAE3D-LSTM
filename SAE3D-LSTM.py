import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.nn.modules.utils import _pair
from torch.autograd import Variable
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from functools import reduce
from utils import nice_print, mem_report, cpu_stats
import copy
import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from torch import nn, einsum



class SAE3DLSTM(nn.Module):
    '''
    tau:timstamp-1
    input:(n,b,c,t,h,w) n:fragments;t:frames in the fragments;
    output:(b,c,t,h,w)'''

    def __init__(self, input_shape, hidden_size, num_layers, kernel_size,reduce_channel):
        super().__init__()
        self.agg = nn.Parameter(torch.randn(1, input_shape[0], input_shape[1], input_shape[2], input_shape[3])) #c,t,h,w
        # self._tau = n
        self._cells = []

        input_shape = list(input_shape)
        for i in range(num_layers):
            cell = SASAE3DLSTMCell(input_shape, hidden_size, kernel_size,self.agg,reduce_channel)
            # NOTE hidden state becomes input to the next cell
            input_shape[0] = hidden_size
            self._cells.append(cell)
            # Hook to register submodule
            setattr(self, "cell{}".format(i), cell)

    def forward(self, input):
        # NOTE (seq_len, batch, input_shape)
        batch_size = input.size(1)
        c_history_states = []
        h_states = []
        outputs = []
        seq_len=input.shape[0]
        # self.position_emb = _get_sinusoid_encoding_table(seq_len-1,)

        for step, x in enumerate(input):
            for cell_idx, cell in enumerate(self._cells):
                if step == 0: #
                    c_history, m, h = self._cells[cell_idx].init_hidden(
                        batch_size,seq_len-1, input.device
                    )
                    c_history_states.append(c_history)
                    h_states.append(h)

                # NOTE c_history and h are coming from the previous time stamp, but we iterate over cells
                c_history, m, h = cell(
                    x, c_history_states[cell_idx], m, h_states[cell_idx]
                )
                c_history_states[cell_idx] = c_history
                h_states[cell_idx] = h
                # NOTE hidden state of previous LSTM is passed as input to the next one
                x = h
            outputs.append(h)

        return outputs[-1]


class SASAE3DLSTMCell(nn.Module):
    def __init__(self, input_shape, hidden_size, kernel_size,agg,reduce_channel):
        super().__init__()
        #input shape [c,t,h,w]
        in_channels = input_shape[0]
        self._input_shape = input_shape
        self._hidden_size = hidden_size

        # memory gates: input, cell(input modulation), forget
        self.weight_xi = ConvDeconv3d(in_channels, hidden_size, kernel_size)
        self.weight_hi = ConvDeconv3d(hidden_size, hidden_size, kernel_size, bias=False)

        self.weight_xg = copy.deepcopy(self.weight_xi)
        self.weight_hg = copy.deepcopy(self.weight_hi)


        '''my defination'''
        self.agg=agg
        self.reduce_channel=reduce_channel
        self.conv=nn.Conv3d(input_shape[0],self.reduce_channel,kernel_size=(input_shape[1],1,1))
        dim=self.reduce_channel*input_shape[2]*input_shape[3] #c,h,w
        self.to_qk = nn.Linear(dim, dim * 2, bias = False)
        self.scale=dim** -0.5

        memory_shape = list(input_shape)
        memory_shape[0] = hidden_size

        self.layer_norm = nn.LayerNorm(memory_shape)

        # for spatiotemporal memory
        self.weight_xi_prime = copy.deepcopy(self.weight_xi)
        self.weight_mi_prime = copy.deepcopy(self.weight_hi)

        self.weight_xg_prime = copy.deepcopy(self.weight_xi)
        self.weight_mg_prime = copy.deepcopy(self.weight_hi)

        self.weight_xf_prime = copy.deepcopy(self.weight_xi)
        self.weight_mf_prime = copy.deepcopy(self.weight_hi)

        self.weight_xo = copy.deepcopy(self.weight_xi)
        self.weight_ho = copy.deepcopy(self.weight_hi)
        self.weight_co = copy.deepcopy(self.weight_hi)
        self.weight_mo = copy.deepcopy(self.weight_hi)

        self.weight_111 = nn.Conv3d(hidden_size + hidden_size, hidden_size, 1)

    def attn(self,q, k, v):
        sim = einsum('b i d, b j d -> b i j', q, k)
        attn = sim.softmax(dim=-1)
        attn=attn.squeeze(1)
        out = einsum('b j, j b c t h w -> b c t h w', attn, v)
        return out

    def self_attention_qk(self, agg,x,h,c_history):
        b,_,_,_,_=x.shape
        #value
        agg_batch=agg.repeat(b,1,1,1,1)
        agg_value=agg_batch.unsqueeze(0)
        x_value=x.unsqueeze(0)
        h_value = h.unsqueeze(0)
        value=torch.cat((agg_value,x_value,h_value,c_history),0)# n, b ,c ,t ,h ,w

        #reduce dimension
        x_reduce=self.conv(x)
        x_reduce = torch.squeeze(x_reduce,2)
        h_reduce=self.conv(h)
        h_reduce = torch.squeeze(h_reduce,2)
        agg_reduce=self.conv(agg_batch)
        agg_reduce = torch.squeeze(agg_reduce,2)
        c_history=rearrange(c_history,'n b c t h w -> (n b) c t h w')
        c_reduce=self.conv(c_history)
        c_reduce = torch.squeeze(c_reduce,2)
        c_reduce=rearrange(c_reduce,'(n b) c h w -> b n (c h w)',b=b)

        #flatten
        x_reduce = torch.flatten(x_reduce,1)#b,chw
        h_reduce = torch.flatten(h_reduce, 1)  # b,chw
        agg_reduce = torch.flatten(agg_reduce, 1)  # b,chw
        x_reduce = x_reduce.unsqueeze(1)
        h_reduce = h_reduce.unsqueeze(1)
        agg_reduce = agg_reduce.unsqueeze(1)
        #cat
        sequence=torch.cat((agg_reduce,x_reduce,h_reduce,c_reduce),1)
        #q,k
        q, k= self.to_qk(sequence).chunk(2, dim=-1)
        q = q * self.scale
        (agg_q, q_), (agg_k, k_) = map(lambda t: (t[:, :1], t[:, 1:]), (q, k))
        s= self.attn(agg_q, k, value)
        s = torch.squeeze(s)
        return s

    def forward(self, x, c_history, m, h):
        # Normalized shape for LayerNorm is CxT×H×W
        # print(x.shape,h.shape,'x h')
        normalized_shape = list(h.shape[-3:])

        def LR(input):
            return F.layer_norm(input, normalized_shape)

        i = torch.sigmoid(LR(self.weight_xi(x) + self.weight_hi(h)))
        g = torch.tanh(LR(self.weight_xg(x) + self.weight_hg(h)))

        s =self.self_attention_qk(self.agg,x, h,c_history)
        # nice_print(**locals())
        # mem_report()
        # cpu_stats()
        # print(s.shape,1)
        c = i * g + self.layer_norm(c_history[-1] + s) #distill useful information

        i_prime = torch.sigmoid(LR(self.weight_xi_prime(x) + self.weight_mi_prime(m)))
        g_prime = torch.tanh(LR(self.weight_xg_prime(x) + self.weight_mg_prime(m)))
        f_prime = torch.sigmoid(LR(self.weight_xf_prime(x) + self.weight_mf_prime(m)))

        m = i_prime * g_prime + f_prime * m
        o = torch.sigmoid(
            LR(
                self.weight_xo(x)
                + self.weight_ho(h)
                + self.weight_co(c)
                + self.weight_mo(m)
            )
        )
        h = o * torch.tanh(self.weight_111(torch.cat([c, m], dim=1)))

        # TODO is it correct FIFO?
        c_history = torch.cat([c_history[1:], c[None, :]], dim=0)
        # nice_print(**locals())
        return (c_history, m, h)

    def init_hidden(self, batch_size, n, device=None):
        memory_shape = list(self._input_shape)
        memory_shape[0] = self._hidden_size
        c_history = torch.zeros(n, batch_size, *memory_shape, device=device)
        m = torch.zeros(batch_size, *memory_shape, device=device)
        h = torch.zeros(batch_size, *memory_shape, device=device)

        return (c_history, m, h)


class ConvDeconv3d(nn.Module):
    def __init__(self, in_channels, out_channels, *vargs, **kwargs):
        super().__init__()

        self.conv3d = nn.Conv3d(in_channels, out_channels, *vargs, **kwargs)
        # self.conv_transpose3d = nn.ConvTranspose3d(out_channels, out_channels, *vargs, **kwargs)

    def forward(self, input):
        # print(self.conv3d(input).shape, input.shape)
        # return self.conv_transpose3d(self.conv3d(input))
        return F.interpolate(self.conv3d(input), size=input.shape[-3:], mode="nearest")


class Model(nn.Module):
    def __init__(self,window=3,frame=6,reduce_channel=2,width=12,height=16,hidden_units=32,out_channel=2):
        super().__init__()
        self.window=window
        self.width=width
        self.height=height
        self.out_channel=out_channel
        self.sae3d = nn.Sequential(
            SAE3DLSTM(input_shape=(hidden_units, frame, width, height), hidden_size=hidden_units, num_layers=2, kernel_size=(3, 3, 3),reduce_channel=reduce_channel),
            nn.ReLU(),
        )

        self.conv3d_encoder = nn.Sequential(
            nn.Conv3d(2, hidden_units, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(hidden_units, hidden_units, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU()
        )

        self.conv3d_decoder = nn.Sequential(
            nn.Conv3d(hidden_units, hidden_units, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(hidden_units, 2, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
        )

        self.vector = nn.Sequential(
            nn.Linear(56, 10),
            nn.ReLU(),
            nn.Linear(10, self.out_channel * width * height),
        )

        self.map= nn.Sequential(
            nn.Conv2d(2*frame,2,1)
        )

    def forward(self, video, x2,y=None): #x2:external factors
        # encoder
        video = rearrange(video, 'b (n t) c h w -> (n b) c t h w ', n=self.window)
        video= self.conv3d_encoder(video)
        video = rearrange(video, '(n b) c t h w -> n b c t h w ', n=self.window)
        video = self.sae3d(video)
        video=self.conv3d_decoder(video)
        video = rearrange(video, 'b c t h w -> b (c t) h w')
        video = self.map(video)
        # external factors
        x2 = self.vector(x2)
        x2=torch.reshape(x2,(-1,self.out_channel,self.width,self.height))
        out = video+x2
        if y is not None:
            loss=F.mse_loss(out, y)
            return loss
        else:
            return out

if __name__ == '__main__':
    video = np.ones((2, 12, 2,12, 16))  # b (n t) c h w。t:frames in the fragment; n:fragments,i.e.,closeness,period,trend; c:channel; h:height; h:width; b: batch
    video = torch.tensor(video, dtype=torch.float)
    x2 = np.ones((2, 56))  #x2:external factors
    x2 = torch.tensor(x2, dtype=torch.float)
    model = Model(3,4)
    y = model(video, x2)
    print(y.shape)

