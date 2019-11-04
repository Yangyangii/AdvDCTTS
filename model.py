from config import ConfigArgs as args
import torch
import torch.nn as nn
from network import TextEncoder, AudioEncoder, AudioDecoder, DotProductAttention
from torch.nn.utils import weight_norm as norm

import layers as ll
import module as mm


class Text2Mel(nn.Module):
    """
    Text2Mel
    Args:
        L: (N, Tx) text
        S: (N, Ty/r, n_mels) previous audio
    Returns:
        Y: (N, Ty/r, n_mels)
    """
    def __init__(self):
        super(Text2Mel, self).__init__()
        self.name = 'Text2Mel'
        self.embed = nn.Embedding(len(args.vocab), args.Ce, padding_idx=0)
        self.TextEnc = TextEncoder()
        self.AudioEnc = AudioEncoder()
        self.Attention = DotProductAttention()
        self.AudioDec = AudioDecoder()
    
    def forward(self, L, S):
        L = self.embed(L).transpose(1,2) # -> (N, Cx, Tx) for conv1d
        S = S.transpose(1,2) # (N, n_mels, Ty/r) for conv1d
        K, V = self.TextEnc(L) # (N, Cx, Tx) respectively
        Q = self.AudioEnc(S) # -> (N, Cx, Ty/r)
        R, A = self.Attention(K, V, Q) # -> (N, Cx, Ty/r)
        R_ = torch.cat((R, Q), 1) # -> (N, Cx*2, Ty/r)
        Y = self.AudioDec(R_) # -> (N, n_mels, Ty/r)
        return Y.transpose(1, 2), A # (N, Ty/r, n_mels)

    def synthesize(self, L, prev_mels):
        L = self.embed(L).transpose(1,2) # -> (N, Cx, Tx) for conv1d
        K, V = self.TextEnc(L) # (N, Cx, Tx) respectively

        S = prev_mels.transpose(1,2) # (N, n_mels, Ty/r) for conv1d
        for t in range(args.max_Ty-1):
            Q = self.AudioEnc(S) # -> (N, Cx, Ty/r)
            R, A = self.Attention(K, V, Q) # -> (N, Cx, Ty/r)
            R_ = torch.cat((R, Q), 1) # -> (N, Cx*2, Ty/r)
            Y = self.AudioDec(R_) # -> (N, n_mels, Ty/r)
            S[:, :, t+1] = Y[:, :, t]
        return Y.transpose(1, 2), A # (N, Ty/r, n_mels)

class SSRN(nn.Module):
    """
    SSRN
    Args:
        Y: (N, Ty/r, n_mels)
    Returns:
        Z: (N, Ty, n_mags)
    """
    def __init__(self):
        super(SSRN, self).__init__()
        self.name = 'SSRN'
        # (N, n_mels, Ty/r) -> (N, Cs, Ty/r)
        self.hc_blocks = nn.ModuleList([norm(mm.Conv1d(args.n_mels, args.Cs, 1, activation_fn=torch.relu))])
        self.hc_blocks.extend([norm(mm.HighwayConv1d(args.Cs, args.Cs, 3, dilation=3**i))
                               for i in range(2)])
        # (N, Cs, Ty/r*2) -> (N, Cs, Ty/r*2)
        self.hc_blocks.extend([norm(mm.ConvTranspose1d(args.Cs, args.Cs, 4, stride=2, padding=1))])
        self.hc_blocks.extend([norm(mm.HighwayConv1d(args.Cs, args.Cs, 3, dilation=3**i))
                               for i in range(2)])
        # (N, Cs, Ty/r*2) -> (N, Cs, Ty/r*4==Ty)
        self.hc_blocks.extend([norm(mm.ConvTranspose1d(args.Cs, args.Cs, 4, stride=2, padding=1))])
        self.hc_blocks.extend([norm(mm.HighwayConv1d(args.Cs, args.Cs, 3, dilation=3**i))
                               for i in range(2)])
        # (N, Cs, Ty) -> (N, Cs*2, Ty)
        self.hc_blocks.extend([norm(mm.Conv1d(args.Cs, args.Cs*2, 1))])
        self.hc_blocks.extend([norm(mm.HighwayConv1d(args.Cs*2, args.Cs*2, 3, dilation=1))
                               for i in range(2)])
        # (N, Cs*2, Ty) -> (N, n_mags, Ty)
        self.hc_blocks.extend([norm(mm.Conv1d(args.Cs*2, args.n_mags, 1))])
        self.hc_blocks.extend([norm(mm.Conv1d(args.n_mags, args.n_mags, 1, activation_fn=torch.relu))
                               for i in range(2)])
        self.hc_blocks.extend([norm(mm.Conv1d(args.n_mags, args.n_mags, 1))])

    def forward(self, Y):
        Y = Y.transpose(1, 2) # -> (N, n_mels, Ty/r)
        Z = Y
        # -> (N, n_mags, Ty)
        for i in range(len(self.hc_blocks)):
            Z = self.hc_blocks[i](Z)
        Z = torch.sigmoid(Z)
        return Z.transpose(1, 2) # (N, Ty, n_mags)


class ConditionalDiscriminatorBlock(nn.Module):
    def __init__(self):
        super(ConditionalDiscriminatorBlock, self).__init__()
        self.c_net = nn.Sequential(
            # (N, 80, Tmel)
            ll.CustomConv1d(80, 256, kernel_size=1, stride=1, padding='same', lrelu=True),
            ll.LeakyReLU(),
            # (N, 256, Tmel)
        )
        self.net = nn.ModuleList([
            # (N, n_mags, Tmel*4)
            ll.CustomConv1d(args.n_mags, 64, kernel_size=3, stride=1, padding='same', lrelu=False),
            ll.LeakyReLU(),
            # (N, 16, Tmel*4)
            ll.CustomConv1d(64, 128, kernel_size=5, stride=2, padding='same', lrelu=False),
            ll.LeakyReLU(),
            # (N, 64, Tmel*2)
            ll.CustomConv1d(128, 256, kernel_size=5, stride=2, padding='same', lrelu=False),
            ll.LeakyReLU(),
            # (N, 256, Tmel)
        ])
        self.postnet = nn.ModuleList([
            ll.CustomConv1d(256, 256, kernel_size=3, stride=1, padding='same', lrelu=False),
            ll.LeakyReLU(),
            ll.CustomConv1d(256, 128, kernel_size=3, stride=1, padding='same', lrelu=False),
            ll.LeakyReLU(),
            ll.CustomConv1d(128, 1, kernel_size=3, stride=1, padding='same', lrelu=False),
        ])

    def forward(self, x, c):
        features_k = []
        y = x
        for idx in range(len(self.net)):
            y = self.net[idx](y)
            if idx % 2 == 1:
                features_k.append(y) # append only after activation
        c = self.c_net(c)
        y = y + c # (N, 256, Twav//256)
        for idx in range(len(self.postnet)):
            y = self.postnet[idx](y)
            if idx % 2 == 1:
                features_k.append(y) # append only after activation
        return y, features_k
