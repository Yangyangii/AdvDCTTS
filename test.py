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
from tensorboardX import SummaryWriter

import numpy as np
import pandas as pd
from collections import deque
from models import *
from data import SpeechDataset, collate_fn
import utils
from scipy.io.wavfile import write


def evaluate(model, data_loader, batch_size=100):
    # valid_loss = 0.
    with torch.no_grad():
        for step, (texts, mels, mags) in tqdm(enumerate(data_loader), total=len(data_loader)):
            texts, mels = texts.to(DEVICE), mels.to(DEVICE)
            mags_hat = model(mels)  # Predict
            mags_hat = mags_hat.cpu().numpy()
            mags = mags.numpy()
            # import pdb; pdb.set_trace()
            for idx in range(len(mags)):
                fname = step*batch_size + idx
                wav = utils.spectrogram2wav(mags_hat[idx])
                write(os.path.join(args.testdir, '{:03d}-gen.wav'.format(fname)), args.sr, wav)
                wav = utils.spectrogram2wav(mags[idx])
                write(os.path.join(args.testdir, '{:03d}-gt.wav'.format(fname)), args.sr, wav)
            # # You can adjust # of test samples
            # if step > 2:
            #     break
    # print()

def main():
    ssrn = SSRN().to(DEVICE)

    mname = type(ssrn).__name__
    ckpt = sorted(glob.glob(os.path.join(args.logdir, mname, '{}-*k.pth.tar'.format(mname))))
    state = torch.load(ckpt[-1])
    ssrn.load_state_dict(state['model'])

    if not os.path.exists(args.testdir):
        os.makedirs(args.testdir)

    validset = SpeechDataset(args.data_path, args.meta_eval, type(ssrn).__name__, mem_mode=args.mem_mode)
    valid_loader = DataLoader(dataset=validset, batch_size=args.test_batch,
                              shuffle=False, collate_fn=collate_fn)

    evaluate(ssrn, valid_loader, args.test_batch)
    return None

if __name__ == '__main__':
    gpu_id = int(sys.argv[1])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set random seem for reproducibility
    seed = 999
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    main()
    
