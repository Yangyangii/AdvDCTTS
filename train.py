from config import ConfigArgs as args
import os, sys, glob
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter

import numpy as np
import pandas as pd
from collections import deque
from models import Text2Mel, SSRN
from data import SpeechDataset, collate_fn, t2m_collate_fn, t2m_ga_collate_fn, load_vocab
from utils import att2img, spectrogram2wav, plot_att

def train(model, data_loader, valid_loader, optimizer, scheduler, batch_size=32, ckpt_dir=None, writer=None, mode='1'):
    epochs = 0
    global_step = args.global_step
    l1_criterion = nn.L1Loss() # default average
    bd_criterion = nn.BCELoss()
    first_frames = torch.zeros([batch_size, 1, args.n_mels]).to(DEVICE) # (N, Ty/r, n_mels)
    idx2char = load_vocab()[-1]
    while global_step < args.max_step:
        epoch_loss = 0
        for step, (texts, mels, extras) in tqdm(enumerate(data_loader), total=len(data_loader), unit='B', ncols=70, leave=False):
            optimizer.zero_grad()
            if type(model).__name__ == 'Text2Mel':
                if args.ga_mode:
                    texts, mels, gas = texts.to(DEVICE), mels.to(DEVICE), extras.to(DEVICE)
                else:
                    texts, mels = texts.to(DEVICE), mels.to(DEVICE)
                prev_mels = torch.cat((first_frames, mels[:, :-1, :]), 1)
                mels_hat, A = model(texts, prev_mels)  # mels_hat: (N, Ty/r, n_mels), A: (N, Tx, Ty/r)
                if args.ga_mode:
                    l1_loss = l1_criterion(mels_hat, mels)
                    bd_loss = bd_criterion(mels_hat, mels)
                    att_loss = torch.mean(A*gas)
                    loss = l1_loss + bd_loss + att_loss
                else:
                    l1_loss = l1_criterion(mels_hat, mels)
                    bd_loss = bd_criterion(mels_hat, mels)
                    loss = l1_loss + bd_loss
            elif type(model).__name__ == 'SSRN':
                texts, mels, mags = texts.to(DEVICE), mels.to(DEVICE), extras.to(DEVICE)
                mags_hat = model(mels)  # mags_hat: (N, Ty, n_mags)
                l1_loss = l1_criterion(mags_hat, mags)
                bd_loss = bd_criterion(mags_hat, mags)
                loss = 10*(l1_loss + bd_loss)
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            if args.lr_decay:
                scheduler.step()
            optimizer.step()
            epoch_loss += l1_loss.item()
            global_step += 1
            if global_step % args.save_term == 0:
                model.eval()
                val_loss = evaluate(model, valid_loader, l1_criterion, writer, global_step, args.test_batch)
                save_model(model, optimizer, scheduler, val_loss, global_step, ckpt_dir) # save best 5 models
                model.train()
        if args.log_mode:
            # Summary
            avg_loss = epoch_loss / (len(data_loader))
            writer.add_scalar('train/loss', avg_loss, global_step)
            writer.add_scalar('train/lr', scheduler.get_lr()[0], global_step)
            if type(model).__name__ == 'Text2Mel':
                alignment = A[0:1].clone().cpu().detach().numpy()
                writer.add_image('train/alignments', att2img(alignment), global_step) # (Tx, Ty)
                if args.ga_mode:
                    writer.add_scalar('train/loss_att', att_loss, global_step)
                text = texts[0].cpu().detach().numpy()
                text = [idx2char[ch] for ch in text]
                plot_att(alignment[0], text, global_step, path=os.path.join(args.logdir, type(model).__name__, 'A', 'train'))
                mel_hat = mels_hat[0:1].transpose(1,2)
                mel = mels[0:1].transpose(1, 2)
                writer.add_image('train/mel_hat', mel_hat, global_step)
                writer.add_image('train/mel', mel, global_step)
            else:
                mag_hat = mags_hat[0:1].transpose(1, 2)
                mag = mags[0:1].transpose(1, 2)
                writer.add_image('train/mag_hat', mag_hat, global_step)
                writer.add_image('train/mag', mag, global_step)
            # print('Training Loss: {}'.format(avg_loss))
        epochs += 1
    print('Training complete')

def evaluate(model, data_loader, criterion, writer, global_step, batch_size=100):
    valid_loss = 0.
    A = None 
    model_name = type(model).__name__
    with torch.no_grad():
        for step, (texts, mels, extras) in enumerate(data_loader):
            if model_name == 'Text2Mel':
                first_frames = torch.zeros([mels.shape[0], 1, args.n_mels]).to(DEVICE) # (N, Ty/r, n_mels)
                texts, mels = texts.to(DEVICE), mels.to(DEVICE)
                prev_mels = torch.cat((first_frames, mels[:, :-1, :]), 1)
                mels_hat, A = model(texts, prev_mels)  # mels_hat: (N, Ty/r, n_mels), A: (N, Tx, Ty/r)
                loss = criterion(mels_hat, mels)
            elif model_name == 'SSRN':
                texts, mels, mags = texts.to(DEVICE), mels.to(DEVICE), extras.to(DEVICE)
                mags_hat = model(mels)  # Predict
                loss = criterion(mags_hat, mags)
            valid_loss += loss.item()
        avg_loss = valid_loss / (len(data_loader))
        writer.add_scalar('eval/loss', avg_loss, global_step)
        if model_name == 'Text2Mel':
            alignment = A[0:1].clone().cpu().detach().numpy()
            writer.add_image('eval/alignments', att2img(alignment), global_step) # (Tx, Ty)
            text = texts[0].cpu().detach().numpy()
            text = [load_vocab()[-1][ch] for ch in text]
            plot_att(alignment[0], text, global_step, path=os.path.join(args.logdir, type(model).__name__, 'A'))
            mel_hat = mels_hat[0:1].transpose(1,2)
            mel = mels[0:1].transpose(1, 2)
            writer.add_image('eval/mel_hat', mel_hat, global_step)
            writer.add_image('eval/mel', mel, global_step)
        else:
            mag_hat = mags_hat[0:1].transpose(1, 2)
            mag = mags[0:1].transpose(1, 2)
            writer.add_image('eval/mag_hat', mag_hat, global_step)
            writer.add_image('eval/mag', mag, global_step)
    return avg_loss

def save_model(model, optimizer, scheduler, val_loss, global_step, ckpt_dir):
    fname = '{}-{:03d}k.pth'.format(type(model).__name__, global_step//1000)
    state = {
        'global_step': global_step,
        'model': model.state_dict(),
        'loss': val_loss,
        'optimizer': optimizer.state_dict(),
        # 'scheduler': scheduler.state_dict(),
    }
    torch.save(state, os.path.join(ckpt_dir, fname))

def main(network):
    if network == 'text2mel':
        model = Text2Mel().to(DEVICE)
    elif network == 'ssrn':
        model = SSRN().to(DEVICE)
    else:
        print('Wrong network. {text2mel, ssrn}')
        return
    print('Model {} is working...'.format(type(model).__name__))
    print('{} threads are used...'.format(torch.get_num_threads()))
    ckpt_dir = os.path.join(args.logdir, type(model).__name__)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[50000, 150000, 300000], gamma=0.5) #

    if not os.path.exists(ckpt_dir):
        os.makedirs(os.path.join(ckpt_dir, 'A', 'train'))
    else:
        print('Already exists. Retrain the model.')
        ckpt = sorted(glob.glob(os.path.join(ckpt_dir, '*k.pth.tar')))[-1]
        state = torch.load(ckpt)
        model.load_state_dict(state['model'])
        args.global_step = state['global_step']
        optimizer.load_state_dict(state['optimizer'])
        # scheduler.load_state_dict(state['scheduler'])

    # model = torch.nn.DataParallel(model, device_ids=list(range(args.no_gpu))).to(DEVICE)
    if type(model).__name__ == 'Text2Mel':
        if args.ga_mode:
            cfn_train, cfn_eval = t2m_ga_collate_fn, t2m_collate_fn
        else:
            cfn_train, cfn_eval = t2m_collate_fn, t2m_collate_fn
    else:
        cfn_train, cfn_eval = collate_fn, collate_fn

    dataset = SpeechDataset(args.data_path, args.meta_train, type(model).__name__, mem_mode=args.mem_mode, ga_mode=args.ga_mode)
    validset = SpeechDataset(args.data_path, args.meta_eval, type(model).__name__, mem_mode=args.mem_mode)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                             shuffle=True, collate_fn=cfn_train,
                             drop_last=True, pin_memory=True)
    valid_loader = DataLoader(dataset=validset, batch_size=args.test_batch,
                              shuffle=False, collate_fn=cfn_eval, pin_memory=True)
    
    writer = SummaryWriter(ckpt_dir)
    train(model, data_loader, valid_loader, optimizer, scheduler,
          batch_size=args.batch_size, ckpt_dir=ckpt_dir, writer=writer)
    return None

if __name__ == '__main__':
    network = sys.argv[1]
    gpu_id = int(sys.argv[2])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set random seem for reproducibility
    seed = 999
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    main(network=network)
    
