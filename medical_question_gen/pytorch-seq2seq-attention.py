# coding: utf-8
"""
NLP From Scratch: Translation with a Sequence to Sequence Network and Attention
*******************************************************************************
It would also be useful to know about Sequence to Sequence networks and
how they work:
-  `Learning Phrase Representations using RNN Encoder-Decoder for
   Statistical Machine Translation <https://arxiv.org/abs/1406.1078>`__
-  `Sequence to Sequence Learning with Neural
   Networks <https://arxiv.org/abs/1409.3215>`__
-  `Neural Machine Translation by Jointly Learning to Align and
   Translate <https://arxiv.org/abs/1409.0473>`__
-  `A Neural Conversational Model <https://arxiv.org/abs/1506.05869>`__
You will also find the previous tutorials on
:doc:`/intermediate/char_rnn_classification_tutorial`
and :doc:`/intermediate/char_rnn_generation_tutorial`
helpful as those concepts are very similar to the Encoder and Decoder
models, respectively.
And for more, read the papers that introduced these topics:
-  `Learning Phrase Representations using RNN Encoder-Decoder for
   Statistical Machine Translation <https://arxiv.org/abs/1406.1078>`__
-  `Sequence to Sequence Learning with Neural
   Networks <https://arxiv.org/abs/1409.3215>`__
-  `Neural Machine Translation by Jointly Learning to Align and
   Translate <https://arxiv.org/abs/1409.0473>`__
-  `A Neural Conversational Model <https://arxiv.org/abs/1506.05869>`__

**Changes**
- share embeding
- label smoothing

"""
from __future__ import unicode_literals, print_function, division
import time
import sys
from io import open
import unicodedata
import string
import re
import random
import logging
import math
import os
from datetime import datetime

import argparse
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler, SequentialSampler, RandomSampler, BatchSampler
from nlgeval import NLGEval

nlgeval = NLGEval()
from tqdm import tqdm, trange

from tokenizer import ZhTokenizer
from dataset import MedQaDataset, combine_text_answer
from utils import weight_init, showPlot, calculate_rouge,loss_calc
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_token = 0
SOS_token = 2
EOS_token = 3

# parameters

max_text_len = 512
max_question_len = 128

embed_dim = 300
enc_hid_dim = 300
dec_hid_dim = 300


######################################################################
# The Encoder
# -----------
#
# The encoder of a seq2seq network is a RNN that outputs some value for
# every word from the input sentence. For every input word the encoder
# outputs a vector and a hidden state, and uses the hidden state for the
# next input word.
#
# .. figure:: /_static/img/seq-seq-images/encoder-network.png
#    :alt:
#
#

class EncoderRNN(nn.Module):
    def __init__(self, embedding, vocab_size, enc_hid_dim, embed_dim):
        super(EncoderRNN, self).__init__()
        self.input_embedding = embedding
        self.token_type_embedding = nn.Embedding(2, embed_dim)
        self.gru = nn.GRU(embed_dim, enc_hid_dim,
                          batch_first=True, bidirectional=True)
        self.fc=nn.Linear(enc_hid_dim*2,dec_hid_dim)

    def forward(self, input_sen, token_type=None):
        # input_sen = [batch_size, text_max_len]
        input_embedded = self.input_embedding(input_sen)
        if token_type is not None:
            token_type_embedded = self.token_type_embedding(token_type)
            embedded = torch.add(input_embedded, token_type_embedded)
        else:
            embedded = input_embedded
        output, hidden = self.gru(embedded)
        # hidden[-2,:,:]和hidden[-1,:,:]分别代表前向和后向，通过tanh来激活
        hidden=torch.tanh(self.fc(torch.cat((hidden[-2,:,:],hidden[-1,:,:]),dim=1)))
        return output, hidden


######################################################################
# Attention Decoder
# ^^^^^^^^^^^^^^^^^
#
# If only the context vector is passed betweeen the encoder and decoder,
# that single vector carries the burden of encoding the entire sentence.
#
# Attention allows the decoder network to "focus" on a different part of
# the encoder's outputs for every step of the decoder's own outputs. First
# we calculate a set of *attention weights*. These will be multiplied by
# the encoder output vectors to create a weighted combination. The result
# (called ``attn_applied`` in the code) should contain information about
# that specific part of the input sequence, and thus help the decoder
# choose the right output words.
#
# .. figure:: https://i.imgur.com/1152PYf.png
#    :alt:
#
# Calculating the attention weights is done with another feed-forward
# layer ``attn``, using the decoder's input and hidden state as inputs.
# Because there are sentences of all sizes in the training data, to
# actually create and train this layer we have to choose a maximum
# sentence length (input length, for encoder outputs) that it can apply
# to. Sentences of the maximum length will use all the attention weights,
# while shorter sentences will only use the first few.
#
# .. figure:: /_static/img/seq-seq-images/attention-decoder-network.png
#    :alt:
#
#

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Parameter(torch.rand(dec_hid_dim))

    def forward(self, hidden, context, mask):

        # hidden = [batch size, dec hid dim]
        # context = [batch size, sen len, enc hid dim * 2]

        batch_size = context.shape[0]
        src_len = context.shape[1]

        # repeat encoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1).contiguous()

        context = context.contiguous()

        # hidden = [batch size, src sent len, dec hid dim]
        # context = [batch size, src sent len, enc hid dim * 2]

        sim = torch.tanh(self.attn(torch.cat((hidden, context), dim=2)))

        sim = sim.permute(0, 2, 1)

        # sim = [batch size, dec hid dim, sen len]

        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        # v = [batch size, 1, dec hid dim]

        attention = torch.bmm(v, sim).squeeze(1)

        # attention= [batch size, src len]
        attention = attention.masked_fill(mask == 0, float("-inf"))

        return F.softmax(attention, dim=1)


class AttnDecoderRNN(nn.Module):
    def __init__(self, embedding, vocab_size, emb_dim, enc_hid_dim, dec_hid_dim, attention, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.dropout_p = dropout_p

        self.embedding = embedding
        self.attention = attention
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU((enc_hid_dim * 2) + emb_dim,
                          dec_hid_dim, batch_first=True)
        self.out = nn.Linear((enc_hid_dim*2)+dec_hid_dim+emb_dim, vocab_size)

    def forward(self, inputs, hidden,  context, mask):
        # input = [batch size]
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [batch size, text_max_len, encode hid dim]
        embedded = self.dropout(self.embedding(inputs))
        attn_weights = self.attention(hidden, context, mask)
        attn_weights = attn_weights.unsqueeze(1)
        # attn_weights = [batch size, 1, src len]
        context = context.permute(0, 1, 2)
        # context = [batch size, src len, enc hid dim * 2]
        attn_applied = torch.bmm(attn_weights, context)
        # attn_applied = [batch size, 1, src len]
        attn_applied = attn_applied.permute(0, 1, 2)
        # attn_applied = [batch size, src len, enc hid dim * 2]
        gru_input = torch.cat((embedded, attn_applied), dim=2)

        output, hidden = self.gru(gru_input, hidden.unsqueeze(0))

        output = output.squeeze(1)
        attn_applied = attn_applied.squeeze(1)
        embedded = embedded.squeeze(1)

        output = self.out(torch.cat([output, attn_applied, embedded], dim=1))
        return output, hidden.squeeze(0), attn_weights.squeeze(1)

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


######################################################################
# The Seq2Seq Model
# =================
#
# A Recurrent Neural Network, or RNN, is a network that operates on a
# sequence and uses its own output as input for subsequent steps.
#
# A `Sequence to Sequence network <https://arxiv.org/abs/1409.3215>`__, or
# seq2seq network, or `Encoder Decoder
# network <https://arxiv.org/pdf/1406.1078v3.pdf>`__, is a model
# consisting of two RNNs called the encoder and decoder. The encoder reads
# an input sequence and outputs a single vector, and the decoder reads
# that vector to produce an output sequence.
#
# .. figure:: /_static/img/seq-seq-images/seq2seq.png
#    :alt:
#
# Unlike sequence prediction with a single RNN, where every input
# corresponds to an output, the seq2seq model frees us from sequence
# length and order, which makes it ideal for translation between two
# languages.
#
# Consider the sentence "Je ne suis pas le chat noir" → "I am not the
# black cat". Most of the words in the input sentence have a direct
# translation in the output sentence, but are in slightly different
# orders, e.g. "chat noir" and "black cat". Because of the "ne/pas"
# construction there is also one more word in the input sentence. It would
# be difficult to produce a correct translation directly from the sequence
# of input words.
#
# With a seq2seq model the encoder creates a single vector which, in the
# ideal case, encodes the "meaning" of the input sequence into a single
# vector — a single point in some N dimensional space of sentences.
#

class Seq2Seq(nn.Module):  # Bulid Seq2Seq Model
    def __init__(self, vocab_size, embed_dim, enc_hid_dim, dec_hid_dim, dropout_p, pad_token_id=0, eos_token_id=3):
        super(Seq2Seq, self).__init__()
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.embedding=nn.Embedding(vocab_size, embed_dim)
        self.attn = Attention(enc_hid_dim, dec_hid_dim)
        self.encoder = EncoderRNN(self.embedding, vocab_size, enc_hid_dim, embed_dim)
        self.decoder = AttnDecoderRNN(self.embedding, vocab_size, embed_dim, enc_hid_dim,
                     dec_hid_dim, self.attn, dropout_p)

    def create_mask(self, src):
        mask = (src != self.pad_token_id)
        return mask

    def forward(self, input_sen, token_type_ids, questions, teacher_forcing_ratio=0.75):
        if (questions[:,2]==self.eos_token_id).all():
            assert teacher_forcing_ratio == 0, "Must be zero during inference"
            inference = True
        else:
            inference = False
        max_len = questions.shape[1]
        batch_size = questions.shape[0]
        loss = 0
        # encode
        context, hidden = self.encoder(input_sen, token_type_ids)
        # hidden = (batch, hidden_size)

        # tensor to store
        attentions = torch.zeros(max_len, batch_size, input_sen.shape[1]).to(input_sen.device)
        outputs = torch.zeros(max_len, batch_size, vocab_size).to(input_sen.device)

        # decode
        mask = self.create_mask(input_sen)
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        # first input = <sos>
        decode_inputs = questions[:, 0].unsqueeze(1)
        generated = decode_inputs
        unfinished_sents = torch.ones(batch_size,dtype=torch.long).to(input_sen.device)
        for t in range(1, max_len):
            output, hidden, attention = self.decoder(
                decode_inputs, hidden, context, mask)
            outputs[t] = output
            attentions[t] = attention

            if use_teacher_forcing:
                decode_inputs = questions[:, t]
            else:
                top1 = output.argmax(1)
                decode_inputs = top1

            decode_inputs = decode_inputs.unsqueeze(1)
            generated = torch.cat((generated, decode_inputs), dim=1).long()
            # update unfinished_sents
            eos_in_sents = decode_inputs.squeeze(1) == self.eos_token_id
            # unfinished_sents is set to zero if eos in sentence
            unfinished_sents.mul_((~eos_in_sents).long())
            # pdb.set_trace()
            if inference and unfinished_sents.max() == 0:
                break

        return outputs, attentions, generated


######################################################################
# Training the Model
# ------------------
#
# To train we run the input sentence through the encoder, and keep track
# of every output and the latest hidden state. Then the decoder is given
# the ``<SOS>`` token as its first input, and the last hidden state of the
# encoder as its first hidden state.
#
# "Teacher forcing" is the concept of using the real target outputs as
# each next input, instead of using the decoder's guess as the next input.
# Using teacher forcing causes it to converge faster but `when the trained
# network is exploited, it may exhibit
# instability <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.378.4095&rep=rep1&type=pdf>`__.
#
# You can observe outputs of teacher-forced networks that read with
# coherent grammar but wander far from the correct translation -
# intuitively it has learned to represent the output grammar and can "pick
# up" the meaning once the teacher tells it the first few words, but it
# has not properly learned how to create the sentence from the translation
# in the first place.
#
# Because of the freedom PyTorch's autograd gives us, we can randomly
# choose to use teacher forcing or not with a simple if statement. Turn
# ``teacher_forcing_ratio`` up to use more of it.
#

teacher_forcing_ratio = 0.5
tokenizer = ZhTokenizer('vocab.txt', split_by_literals=True)
train_dataset = MedQaDataset(
    tokenizer, prefix='zhtokenizer', file_path='data/round1_train_0920.json')
valid_dataset = MedQaDataset(
    tokenizer, prefix='zhtokenizer', file_path='data/round1_valid_0920.json')
test_dataset = MedQaDataset(
    tokenizer, prefix='zhtokenizer', file_path='data/round1_test_0907.json')


######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.
#


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


######################################################################
# The whole training process looks like this:
#
# -  Start a timer
# -  Initialize optimizers and criterion
# -  Create set of training pairs
# -  Start empty losses array for plotting
#
# Then we call ``train`` many times and occasionally print the progress (%
# of examples, time so far, estimated time) and average loss.
#
def train(model, tokenizer, epochs, batch_size, save_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    model_save_dir = os.path.join(args.save_dir,'seq2seq_attn',datetime.now().strftime('%Y-%m-%d_%H%M'))
    tb_writer = SummaryWriter(model_save_dir)
    plot_losses = []
    save_every_total = 0  # Reset every save_every
    plot_loss_total = 0  # Reset every plot_every
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    
    model.apply(weight_init)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=batch_size)
    train_iterator = trange(int(epochs), desc="Epoch")
    best_rouge = 0.
    early_stopping_steps=0
    global_step = 1
    n_iters = len(train_dataloader)
    logger.info('train and eval')
    model.zero_grad()
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            text_tensor = batch[0].to(device)
            token_type_tensor = batch[1].to(device)
            question_tensor = batch[2].to(device)

            output, attention, result = model(
                text_tensor, token_type_tensor, question_tensor,teacher_forcing_ratio=0.5)
            # output = [batch size,trg sen len, output dim]
            # trg = [batch size,trg sen len]

            output = output[1:].view(-1, output.shape[-1])
            question_tensor = question_tensor[:, 1:, ]
            trg = question_tensor.reshape(-1)
            # output = [(trg sent len - 1) * batch size, output dim]
            # trg = [(trg sent len - 1) * batch size]

            loss = loss_calc(output, trg)
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()            
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)                       
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            plot_loss_total += loss.cpu().detach().item()
            optimizer.step()
            model.zero_grad()
            

            if global_step % save_every == 0:
                valid_loss, rouge_l = evaluate(
                    model, tokenizer, batch_size, max_question_len)
                tb_writer.add_scalar('valid_loss', valid_loss, global_step)
                tb_writer.add_scalar('valid_rouge_l', rouge_l, global_step)
                print('%s (%d %d%%) loss: %.4f rouge_l: %.4f' % (timeSince(start, step / n_iters),
                                                                 step, step / n_iters * 100, valid_loss, rouge_l))
                if best_rouge < rouge_l:
                    logger.info('save best weight')
                    best_rouge = rouge_l
                    
                    torch.save(model.state_dict(), os.path.join(model_save_dir,'pytorch_model.bin'))
                    early_stopping_steps = 0
                else:
                    early_stopping_steps += 1
                if args.early_stopping>0 and early_stopping_steps >= args.early_stopping:
                    break

            if global_step % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                tb_writer.add_scalar('train_loss', plot_loss_avg, global_step)
                plot_loss_total = 0
                showPlot(plot_losses)
            global_step+=1
        if args.early_stopping>0 and early_stopping_steps >= args.early_stopping:
            break
    tb_writer.close()
    return global_step, loss

######################################################################
# Evaluation
# ==========
#
# Evaluation is mostly the same as training, but there are no targets so
# we simply feed the decoder's predictions back to itself for each step.
# Every time it predicts a word we add it to the output string, and if it
# predicts the EOS token we stop there. We also store the decoder's
# attention outputs for display later.
#

def evaluate(model, tokenizer, batch_size, trg_sent_len):
    valid_sampler = RandomSampler(valid_dataset)
    valid_dataloader = DataLoader(
        valid_dataset, sampler=valid_sampler, batch_size=batch_size)
    epoch_iterator = tqdm(valid_dataloader, desc="Iteration")
    # criterion = nn.CrossEntropyLoss(ignore_index=0)
    references = []
    hypotheses = []
    epoch_loss = 0

    model.eval()
    for step, batch in enumerate(epoch_iterator):
        text_tensor = batch[0].to(device)
        token_type_tensor = batch[1].to(device)
        question_tensor = batch[2].to(device)   
        with torch.no_grad():
            output, attention, prediction = model(
                text_tensor, token_type_tensor, question_tensor,teacher_forcing_ratio=0)  # turn off teacher forcing
        rouge_l, r, h = calculate_rouge(
            prediction.cpu().tolist(), question_tensor.cpu().tolist(), tokenizer)
        references.extend(r)
        hypotheses.extend(h)
        # compute loss
        logit = output[1:].view(-1, output.shape[-1]).contiguous()
        # Find a way to avoid calling contiguous
        trg = question_tensor[:, 1:].reshape(-1).contiguous()

        # prediction = [(trg sent len - 1) * batch size, output dim]
        # trg = [(trg sent len - 1) * batch size]
        with torch.no_grad():
            loss = loss_calc(logit, trg)

        epoch_loss += loss
        if step % int(len(valid_dataloader) * 0.1) == 0:
            sample_t = tokenizer.decode(question_tensor[0].cpu().tolist(),True)
            sample_p = tokenizer.decode(prediction[0].cpu().tolist(),True)
            logger.info(f'Batch {step} loss: {loss.item()} ROUGE_L score: {rouge_l}\n'+
                        f'Target {sample_t}\n'+
                        f'Prediction {sample_p}\n\n')
        
    metrics_dict = nlgeval.compute_metrics(references, hypotheses)
    logger.info(metrics_dict)
    return epoch_loss / len(valid_dataloader), metrics_dict['ROUGE_L']


def predict(model, tokenizer, batch_size, trg_sent_len):
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size)
    epoch_iterator = tqdm(test_dataloader, desc="Iteration")
    predictions = []; answers = []
    epoch_loss = 0

    model.eval()
    for step, batch in enumerate(epoch_iterator):
        text_tensor = batch[0].to(device)
        token_type_tensor = batch[1].to(device)
        question_tensor = batch[2].to(device)   
        with torch.no_grad():
            output, attention, prediction = model(
                text_tensor, token_type_tensor, question_tensor,teacher_forcing_ratio=0)  # turn off teacher forcing
        answer_tensor = text_tensor[token_type_tensor]
        batch_size=text_tensor.shape[0]
        for i in range(batch_size):
            answer_tensor=torch.masked_select(text_tensor[i],token_type_tensor[i].bool())
            sample_a = tokenizer.decode(answer_tensor.cpu().tolist(),True)
            sample_p = tokenizer.decode(prediction[i].cpu().tolist(),True)
            answers.append(sample_a)
            predictions.append(sample_p)
    # export to tsv
    with open(os.path.join(args.checkpoint_dir,'predictions.csv'),'w',encoding='utf-8') as file:
        for a, p in zip(answers,predictions):
            file.write('{0}\t{1}\n'.format(a,p))



######################################################################
# We can evaluate interatively and print out the
# input, target, and output to make some subjective quality judgements:
#

def predictInterative(model, tokenizer, n=10):
    for i in range(n):
        text = input('text>')
        answer = input('answer>')
        if not text:
            break
        combined = combine_text_answer(text, answer, tokenizer, max_text_len)
        if not combined:
            print("input error!!")
            return
        text_tensor = torch.tensor([combined[0]], dtype=torch.long)
        token_type_tensor = torch.tensor([combined[1]], dtype=torch.long)
        question_tensor = torch.tensor([[SOS_token]], dtype=torch.long)
        prediction, attention, result = model(
            text_tensor, token_type_tensor, question_tensor)
        output_sentence = tokenizer.decode(prediction[0])
        input_sentence = tokenizer.decode(prediction[0])
        print('hypothersis:\n', output_sentence)
        print('')
        showAttention(input_sentence, output_sentence, attention)

######################################################################
# For a better viewing experience we will do the extra work of adding axes
# and labels:
#


def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       [EOS_token], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def parseArgs(args):
    """
    Parse the arguments from the given command line
    Args:
        args (list<str>): List of arguments to parse. If None, the default sys.argv will be parsed
    """

    parser = argparse.ArgumentParser()

    # Global options
    globalArgs = parser.add_argument_group('Global options')
    globalArgs.add_argument('--train',
                            action='store_true', default=True,
                            help='train model')
    globalArgs.add_argument('--predict',
                            action='store_true', default=False,
                            help='run predict')
    globalArgs.add_argument('--interactively',
                            action='store_true', default=False,
                            help='run predict')
    globalArgs.add_argument('--save_dir',
                            type=str, default='checkpoints',
                            help='model save dir')
    globalArgs.add_argument('--checkpoint_dir',
                            type=str, default=None,
                            help='checkpoint dir init from')
    # Dataset options
    datasetArgs = parser.add_argument_group('Dataset options')
    datasetArgs.add_argument(
        '--corpus', help='corpus on which extract the dataset.')

    # Training options
    trainingArgs = parser.add_argument_group('Training options')
    trainingArgs.add_argument(
        '--epochs', type=int, default=100, help='maximum number of epochs to run')
    trainingArgs.add_argument('--save_every', type=int, default=2000,
                              help='nb of mini-batch step before creating a model checkpoint')
    trainingArgs.add_argument('--batch_size', type=int,
                              default=2, help='mini-batch size')
    trainingArgs.add_argument(
        '--learning_rate', type=float, default=0.002, help='Learning rate')
    trainingArgs.add_argument(
        '--dropout_p', type=float, default=0.2, help='Dropout probabilities')
    trainingArgs.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    trainingArgs.add_argument(
        '--early_stopping', type=int, default=0, help='enable early stopping if value >0')
    trainingArgs.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    trainingArgs.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization lfevel selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    return parser.parse_args(args)


######################################################################
# Training and Evaluating
# =======================
#
# With all these helper functions in place (it looks like extra work, but
# it makes it easier to run multiple experiments) we can actually
# initialize a network and start training.
#
# Remember that the input sentences were heavily filtered. For this small
# dataset we can use relatively small networks of 256 hidden nodes and a
# single GRU layer. After about 40 minutes on a MacBook CPU we'll get some
# reasonable results.
#
# .. Note::
#    If you run this notebook you can train, interrupt the kernel,
#    evaluate, and continue training later. Comment out the lines where the
#    encoder and decoder are initialized and run ``trainIters`` again.
#
args = parseArgs(sys.argv[1:])
vocab_size = tokenizer.vocab_size()

model = Seq2Seq(vocab_size,embed_dim, enc_hid_dim, dec_hid_dim, args.dropout_p, PAD_token, EOS_token).to(device)

if args.train:
    global_step, loss = train(model, tokenizer, epochs=args.epochs, batch_size=args.batch_size, save_every=args.save_every, learning_rate=args.learning_rate)
    logger.info("training finished:\n global_step = %s, loss = %s", global_step, loss)
    avg_loss, rouge_l = evaluate(model, tokenizer, args.batch_size, max_question_len)
    logger.info("evaluate finished:\n avg rouge_l = %s, loss = %s", rouge_l, avg_loss)
if args.predict:
    checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'pytorch_model.bin'))
    model.load_state_dict(checkpoint)
    logger.info("Pre-trained Seq2seq-Attn is successfully loaded")
    if args.interactively:
        predictInterative(model, tokenizer)
    else:
        predict(model, tokenizer, args.batch_size, max_question_len)
