from config import ConfigArgs as args
import os, sys, glob
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

import numpy as np
import pandas as pd
from model import Text2Mel, SSRN
from data import TextDataset, synth_collate_fn, load_vocab
import utils
from scipy.io.wavfile import write


def synthesize(t2m, ssrn, data_loader, batch_size=100):
    '''
    DCTTS Architecture
    Text --> Text2Mel --> SSRN --> Wav file
    '''
    # Text2Mel
    idx2char = load_vocab()[-1]
    with torch.no_grad():
        print('='*10, ' Text2Mel ', '='*10)
        for step, (texts, _, _) in tqdm(enumerate(data_loader), total=len(data_loader), ncols=70):
            texts = texts.to(DEVICE)
            prev_mel_hats = torch.zeros([len(texts), args.max_Ty, args.n_mels]).to(DEVICE)
            total_mel_hats, A = t2m.synthesize(texts, prev_mel_hats)
            alignments = A.cpu().detach().numpy()
            visual_texts = texts.cpu().detach().numpy()
            # Mel --> Mag
            mags = ssrn(total_mel_hats) # mag: (N, Ty, n_mags)
            mags = mags.cpu().detach().numpy()
            for idx in range(len(mags)):
                fname = step*batch_size + idx
                text = [idx2char[ch] for ch in visual_texts[idx]]
                utils.plot_att(alignments[idx], text, args.global_step, path=os.path.join(args.sampledir, 'A'), name='{:02d}.png'.format(fname))
                wav = utils.spectrogram2wav(mags[idx])
                write(os.path.join(args.sampledir, '{:02d}.wav'.format(fname)), args.sr, wav)

    return None

def main():
    testset = TextDataset(args.testset)
    test_loader = DataLoader(dataset=testset, batch_size=args.test_batch, drop_last=False,
                             shuffle=False, collate_fn=synth_collate_fn, pin_memory=True)

    t2m = Text2Mel().to(DEVICE)
    ssrn = SSRN().to(DEVICE)
    
    mname = type(t2m).__name__
    ckpt = sorted(glob.glob(os.path.join(args.logdir, mname, '{}-*k.pth'.format(mname))))
    state = torch.load(ckpt[-1])
    t2m.load_state_dict(state['model'])
    args.global_step = state['global_step']

    mname = type(ssrn).__name__
    ckpt = sorted(glob.glob(os.path.join(args.logdir, mname, '{}-*k.pth'.format(mname))))
    state = torch.load(ckpt[-1])
    ssrn.load_state_dict(state['model'])

    print('All of models are loaded.')

    t2m.eval()
    ssrn.eval()
    
    if not os.path.exists(os.path.join(args.sampledir, 'A')):
        os.makedirs(os.path.join(args.sampledir, 'A'))
    synthesize(t2m, ssrn, test_loader, args.test_batch)

if __name__ == '__main__':
    gpu_id = int(sys.argv[1])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
