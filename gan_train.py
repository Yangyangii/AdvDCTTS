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
from utils import att2img, spectrogram2wav, plot_att

def feature_maching_loss(feature_inputs, feature_labels, criterion=None):
    l_tot = 0.
    l_weight = 4./5.
    for f_input, f_label in zip(feature_inputs, feature_labels):
        l_tot = l_tot + l_weight*criterion(f_input, f_label.detach())
    return l_tot

def train(G, D, data_loader, valid_loader, G_optim, D_optim, batch_size=32, ckpt_dir=None, writer=None, mode='1'):
    epochs = 0
    global_step = args.global_step
    l1_criterion = nn.L1Loss() # default average
    bd_criterion = nn.BCELoss()
    mse = nn.MSELoss()
    # torch.backends.cudnn.benchmark = True # turn-off if you have old GPU and drivers
    while global_step < args.max_step:
        epoch_loss = 0
        for step, (texts, mels, extras) in tqdm(enumerate(data_loader), total=len(data_loader), unit='B', ncols=70, leave=False):
            texts, mels, mags = texts.to(DEVICE), mels.to(DEVICE), extras.to(DEVICE)
            
            ## Training D
            mags_hat = G(mels)  # mags_hat: (N, Ty, n_mags)
            mels = mels.transpose(1, 2)
            mags, mags_hat = mags.transpose(1, 2), mags_hat.transpose(1, 2)
            
            if global_step > args.begin_gan:
                for c in range(args.n_critic):
                    d_real, _ = D(mags, mels)
                    d_fake, _ = D(mags_hat.detach(), mels)
                    
                    # LSGAN loss
                    # d_loss_r = torch.mean((d_real-1)**2) # D(x)
                    # d_loss_f = torch.mean(d_fake**2) # D(G(z))
                    d_loss = 0.
                    for k, (real, fake) in enumerate(zip(d_real, d_fake)):
                        real_label = torch.ones_like(real)
                        fake_label = torch.zeros_like(fake)
                        d_loss_r = mse(real, real_label)
                        d_loss_f = mse(fake, fake_label)
                        d_loss = d_loss + d_loss_r + d_loss_f

                    D_optim.zero_grad()
                    d_loss.backward()
                    D_optim.step()

            ## Training G
            # recon loss
            l1_loss = l1_criterion(mags_hat, mags)
            bd_loss = bd_criterion(mags_hat, mags)
            recon_loss = l1_loss + bd_loss

            gan_loss = 0.
            if global_step > args.begin_gan:
                # G loss
                g_fake, f_fake = D(mags_hat, mels)
                # D_g_loss = torch.mean((g_fake-1)**2)

                d_real, f_real = D(mags, mels)

                D_G_loss = 0.
                for fake in g_fake:
                    real_label = torch.ones_like(fake)
                    # import pdb; pdb.set_trace()
                    d_loss_f = mse(fake, real_label)
                    D_G_loss = D_G_loss + d_loss_f
                
                fm_loss = feature_maching_loss(f_fake, f_real, criterion=l1_criterion)

                # fm_loss = feature_maching_loss(f_fake, f_real, criterion=l1_criterion)
                gan_loss = 10*fm_loss + D_G_loss

            g_loss = 10*recon_loss + gan_loss

            G_optim.zero_grad()
            g_loss.backward()
            G_optim.step()
            epoch_loss += l1_loss.item()
            global_step += 1
            if global_step % args.save_term == 0:
                G.eval()
                val_loss = evaluate(G, valid_loader, l1_criterion, writer, global_step, args.test_batch)
                save_model(G, G_optim, global_step, ckpt_dir)
                save_model(D, D_optim, global_step, ckpt_dir)
                G.train()
        if args.log_mode:
            # Summary
            avg_loss = epoch_loss / (len(data_loader))
            writer.add_scalar('train/recon_loss', avg_loss, global_step)
            if global_step > args.begin_gan:
                writer.add_scalar('train/d_loss_r', d_loss_r, global_step)
                writer.add_scalar('train/d_loss_f', d_loss_f, global_step)
                writer.add_scalar('train/g_loss', gan_loss, global_step)
                writer.add_scalar('train/fm_loss', fm_loss, global_step)
                writer.add_scalar('train/d_G_loss', D_G_loss, global_step)
            # writer.add_scalar('train/lr', scheduler.get_lr()[0], global_step)
            mag_hat = mags_hat[0:1]
            mag = mags[0:1]
            writer.add_image('train/mag_hat', mag_hat, global_step)
            writer.add_image('train/mag', mag, global_step)
            # print('Training Loss: {}'.format(avg_loss))
        epochs += 1
    print('Training complete')

def evaluate(model, data_loader, criterion, writer, global_step, batch_size=100):
    valid_loss = 0.
    with torch.no_grad():
        for step, (texts, mels, extras) in enumerate(data_loader):
            texts, mels, mags = texts.to(DEVICE), mels.to(DEVICE), extras.to(DEVICE)
            mags_hat = model(mels)  # Predict
            loss = criterion(mags_hat, mags)
            valid_loss += loss.item()
        avg_loss = valid_loss / (len(data_loader))
        writer.add_scalar('eval/loss', avg_loss, global_step)
        mag_hat = mags_hat[0:1].transpose(1, 2)
        mag = mags[0:1].transpose(1, 2)
        writer.add_image('eval/mag_hat', mag_hat, global_step)
        writer.add_image('eval/mag', mag, global_step)
    return avg_loss

def save_model(model, optimizer, global_step, ckpt_dir):
    fname = '{}-{:03d}k.pth.tar'.format(type(model).__name__, global_step//1000)
    state = {
        'global_step': global_step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, os.path.join(ckpt_dir, fname))

def main():
    G = SSRN().to(DEVICE)
    D = MultiScaleDiscriminator().to(DEVICE)
    
    print('{} threads are used...'.format(torch.get_num_threads()))
    ckpt_dir = os.path.join(args.logdir, type(G).__name__)
    G_optim = torch.optim.Adam(G.parameters(), lr=args.lr)
    D_optim = torch.optim.Adam(D.parameters(), lr=args.lr)
    # scheduler = MultiStepLR(optimizer, milestones=[100000, 200000], gamma=0.5)

    if not os.path.exists(ckpt_dir):
        os.makedirs(os.path.join(ckpt_dir, 'A', 'train'))
    else:
        print('Already exists. Retrain the model.')
        import pdb; pdb.set_trace()
        ckpt = sorted(glob.glob(os.path.join(ckpt_dir, '{}-*k.pth.tar'.format(type(G).__name__))))
        state = torch.load(ckpt[-1])
        args.global_step = state['global_step']
        G.load_state_dict(state['model'])
        G_optim.load_state_dict(state['optimizer'])
        # ckpt = sorted(glob.glob(os.path.join(ckpt_dir, '{}-*k.pth'.format(type(D).__name__))))
        # state = torch.load(ckpt[-1])
        # D.load_state_dict(state['model'])
        # D_optim.load_state_dict(state['optimizer'])

    dataset = SpeechDataset(args.data_path, args.meta_train, type(G).__name__, mem_mode=args.mem_mode)
    validset = SpeechDataset(args.data_path, args.meta_eval, type(G).__name__, mem_mode=args.mem_mode)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                             shuffle=True, collate_fn=collate_fn,
                             drop_last=True, pin_memory=True, num_workers=args.n_workers)
    valid_loader = DataLoader(dataset=validset, batch_size=args.test_batch,
                              shuffle=False, collate_fn=collate_fn)
    
    writer = SummaryWriter(ckpt_dir)
    train(G, D, data_loader, valid_loader, G_optim, D_optim,
          batch_size=args.batch_size, ckpt_dir=ckpt_dir, writer=writer)
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
    
