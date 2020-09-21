# coding=utf-8
import glob
import json
import logging
import math
import os
import os.path as osp
import pickle
import random
import re
import subprocess
import sys

import lmdb
import msgpack
import numpy as np
import torch
import torch.functional as F
import torch.nn.init as init
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import (BatchSampler, DataLoader, Dataset, RandomSampler,
                              Sampler, SequentialSampler)
from tqdm import tqdm, trange
from .network import log_sum_exp
from .metrics import call_multi_bleu_perl

logger = logging.getLogger(__name__)


def reconstruct(model, test_data_batch, vocab, strategy, fname):
    hyps = []
    refs = []
    with open(fname, "w") as fout:
        #for i in range(10):
            # batch_data = test_data_batch[i]

        for batch_data in test_data_batch:
            decoded_batch = model.reconstruct(batch_data, strategy)

            source = [[vocab.id2word(id_.item()) for id_ in sent] for sent in batch_data]
            for j in range(len(batch_data)):
                ref = " ".join(source[j])
                hyp = " ".join(decoded_batch[j])
                fout.write("SOURCE: {}\n".format(ref))
                fout.write("RECON: {}\n\n".format(hyp))

                refs += [ref[len("<s>"): -len("</s>")]]
                if strategy == "beam":
                    hyps += [hyp[len("<s>"): -len("</s>")]]
                else:
                    hyps += [hyp[: -len("</s>")]]

    fname_ref = fname + ".ref"
    fname_hyp = fname + ".hyp"
    with open(fname_ref, "w") as f:
        f.write("\n".join(refs))
    with open(fname_hyp, "w") as f:
        f.write("\n".join(hyps))
    call_multi_bleu_perl("scripts/multi-bleu.perl", fname_hyp, fname_ref, verbose=True)




def calc_iwnll(model_vae, eval_dataloader, args, ns=20):

    eval_loss = 0.0
    ############ Perplexity ############
    report_kl_loss = report_rec_loss = report_loss = 0
    report_num_words = report_num_sents = 0

    for batch in tqdm(eval_dataloader, desc="Evaluating PPL"):
        # pdb.set_trace()
        x0, x1, x_lengths = batch

        max_len_values, _ = x_lengths.max(0)
        x0 = x0[:,:max_len_values[0]]
        x1 = x1[:,:max_len_values[1]]

        x0 = x0.to(args.device)
        x1 = x1.to(args.device)
        x_lengths = x_lengths.to(args.device)

        # pdb.set_trace()
        # not predict start symbol
        report_num_words += x_lengths[:,1].sum().item()
        report_num_sents += args.eval_batch_size

        with torch.no_grad():
            loss, loss_rc, loss_kl = model_vae.loss_iw(x0, x1, nsamples=100, ns=5)

        loss_rc = loss_rc.sum()
        loss_kl = loss_kl.sum()
        loss = loss.sum()

        report_rec_loss += loss_rc.item()
        report_kl_loss += loss_kl.item()
        report_loss += loss.item()

        # pdb.set_trace()
        
    test_loss = report_loss / report_num_sents
    
    elbo = (report_kl_loss - report_rec_loss) / report_num_sents
    nll  = - report_rec_loss / report_num_sents
    kl   = report_kl_loss / report_num_sents
    ppl  = np.exp(-report_loss / report_num_words)

    return ppl, elbo, nll, kl



def calc_rec(model_vae, eval_dataloader, args, ns=1):

    eval_loss = 0.0
    ############ Perplexity ############
    report_kl_loss = report_rec_loss = report_loss = 0
    report_num_words = report_num_sents = 0

    i = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating PPL"):
        # pdb.set_trace()
        x0, x1, x_lengths = batch

        max_len_values, _ = x_lengths.max(0)
        x0 = x0[:,:max_len_values[0]]
        x1 = x1[:,:max_len_values[1]]

        x0 = x0.to(args.device)
        x1 = x1.to(args.device)
        x_lengths = x_lengths.to(args.device)

        # pdb.set_trace()
        # not predict start symbol
        report_num_words += x_lengths[:,1].sum().item()
        report_num_sents += args.eval_batch_size

        with torch.no_grad():
            loss, loss_rc, loss_kl = model_vae.loss_iw(x0, x1, nsamples=1, ns=1)

        loss_rc = loss_rc.sum()
        report_rec_loss += loss_rc.item()

        i += 1
        if i > 500:
            break


        # pdb.set_trace()

    nll_s  = - report_rec_loss / report_num_sents
    nll_w  = - report_rec_loss / report_num_words

    return nll_s, nll_w


def calc_mi(model_vae, test_data_batch, args):
    # calc_mi_v3
    

    mi = 0
    num_examples = 0

    mu_batch_list, logvar_batch_list = [], []
    neg_entropy = 0.
    for batch in tqdm(test_data_batch, desc="Evaluating MI, Stage 1"):

        x0, _, x_lengths = batch
        max_len_values, _ = x_lengths.max(0)
        x0 = x0[:,:max_len_values[0]]

        x0 = x0.to(args.device)

        with torch.no_grad():
            # encoding into bert features
            bert_fea = model_vae.encoder(x0)[1]

            # (batch_size, nz)
            mu, logvar = model_vae.encoder.linear(bert_fea).chunk(2, -1)

        x_batch, nz = mu.size()

        #print(x_batch, end=' ')

        num_examples += x_batch

        # E_{q(z|x)}log(q(z|x)) = -0.5*nz*log(2*\pi) - 0.5*(1+logvar).sum(-1)

        neg_entropy += (-0.5 * nz * math.log(2 * math.pi)- 0.5 * (1 + logvar).sum(-1)).sum().item()
        mu_batch_list += [mu.cpu()]
        logvar_batch_list += [logvar.cpu()]


    neg_entropy = neg_entropy / num_examples
    ##print()

    num_examples = 0
    log_qz = 0.
    for i in tqdm(range(len(mu_batch_list)), desc="Evaluating MI, Stage 2"):

        ###############
        # get z_samples
        ###############
        mu, logvar = mu_batch_list[i].cuda(), logvar_batch_list[i].cuda()
        
        # [z_batch, 1, nz]
        with torch.no_grad():
            z_samples = model_vae.reparameterize(mu, logvar, 1)

        z_samples = z_samples.view(-1, 1, nz)
        num_examples += z_samples.size(0)

        ###############
        # compute density
        ###############
        # [1, x_batch, nz]
        #mu, logvar = mu_batch_list[i].cuda(), logvar_batch_list[i].cuda()
        #indices = list(np.random.choice(np.arange(len(mu_batch_list)), 10)) + [i]
        indices = np.arange(len(mu_batch_list))
        mu = torch.cat([mu_batch_list[_] for _ in indices], dim=0).cuda()
        logvar = torch.cat([logvar_batch_list[_] for _ in indices], dim=0).cuda()
        x_batch, nz = mu.size()

        mu, logvar = mu.unsqueeze(0), logvar.unsqueeze(0)
        var = logvar.exp()

        # (z_batch, x_batch, nz)
        dev = z_samples - mu

        # (z_batch, x_batch)
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
            0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))

        # log q(z): aggregate posterior
        # [z_batch]
        log_qz += (log_sum_exp(log_density, dim=1) - math.log(x_batch)).sum(-1)

    log_qz /= num_examples
    mi = neg_entropy - log_qz

    return mi.item()





def calc_au(model_vae, eval_dataloader, args, delta=0.01):
    """compute the number of active units
    """
    cnt = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating AU, Stage 1"):

        x0, _, x_lengths = batch
        max_len_values, _ = x_lengths.max(0)
        x0 = x0[:,:max_len_values[0]]
        x0 = x0.to(args.device)

        with torch.no_grad():
            # encoding into bert features
            bert_fea = model_vae.encoder(x0)[1]

            # (batch_size, nz)
            mean, logvar = model_vae.encoder.linear(bert_fea).chunk(2, -1)

        if cnt == 0:
            means_sum = mean.sum(dim=0, keepdim=True)
        else:
            means_sum = means_sum + mean.sum(dim=0, keepdim=True)
        cnt += mean.size(0)

    # (1, nz)
    mean_mean = means_sum / cnt

    cnt = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating AU, Stage 2"):

        x0, _, _ = batch
        x0 = x0.to(args.device)

        with torch.no_grad():
            # encoding into bert features
            bert_fea = model_vae.encoder(x0)[1]

            # (batch_size, nz)
            mean, _ = model_vae.encoder.linear(bert_fea).chunk(2, -1)

        if cnt == 0:
            var_sum = ((mean - mean_mean) ** 2).sum(dim=0)
        else:
            var_sum = var_sum + ((mean - mean_mean) ** 2).sum(dim=0)
        cnt += mean.size(0)

    # (nz)
    au_var = var_sum / (cnt - 1)

    # pdb.set_trace()
    return (au_var >= delta).sum().item(), au_var


def sample_sentences(vae, vocab, device, num_sentences):
    global logging

    vae.eval()
    sampled_sents = []
    for i in range(num_sentences):
        z = vae.sample_from_prior(1)
        z = z.view(1,1,-1)
        start = vocab.word2id['<s>']
        # START = torch.tensor([[[start]]])
        START = torch.tensor([[start]])
        end = vocab.word2id['</s>']
        START = START.to(device)
        z = z.to(device)
        vae.eval()
        sentence = vae.decoder.sample_text(START, z, end, device)
        decoded_sentence = vocab.decode_sentence(sentence)
        sampled_sents.append(decoded_sentence)
    for i, sent in enumerate(sampled_sents):
        logging(i,":",' '.join(sent))


def visualize_latent(args, epoch, vae, device, test_data):
    nsamples = 1

    with open(os.path.join(args.exp_dir, f'synthetic_latent_{epoch}.txt'),'w') as f:
        test_data_batch, test_label_batch = test_data.create_data_batch_labels(batch_size=args.batch_size, device=device, batch_first=True)
        for i in range(len(test_data_batch)):
            batch_data = test_data_batch[i]
            batch_label = test_label_batch[i]
            batch_size, sent_len = batch_data.size()
            samples, _ = vae.encoder.encode(batch_data, nsamples)
            for i in range(batch_size):
                for j in range(nsamples):
                    sample = samples[i,j,:].cpu().detach().numpy().tolist()
                    f.write(batch_label[i] + '\t' + ' '.join([str(val) for val in sample]) + '\n')


