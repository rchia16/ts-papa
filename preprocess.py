import gc
from os import makedirs
from os.path import join, sep, exists
from sys import platform
import warnings
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle

# import matplotlib; matplotlib.use('Agg')

import argparse
import tqdm
from einops import rearrange


import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# from transformers import (
#     PatchTSTConfig, PatchTSTForPretraining, Trainer, TrainingArguments
# )
from sklearn.decomposition import FastICA, IncrementalPCA
from sklearn.preprocessing import LabelEncoder

from patchtst_backbone import PatchTST_backbone
from utils import segment_to_patches
from evaluations import Evaluation
from digitalsignalprocessing import (
    butter_lowpass_filter, movingaverage, movingmedian, window_filter
)
from config import (
    IMU_FS, BR_FS, SEAT_DATA_DIR, WINDOW_SIZE, WINDOW_SHIFT, USER,
    SBJ_PROCESSED_DIR, M_DIR
)
from utils import load_dataset, prepare_imu_pss, sync_with_last_val
from digitalsignalprocessing import reject_artefact, torch_fft_targets
import random

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

PSS_FS = BR_FS

MAX_LENGTH = PSS_FS*20

# device = ''
imu_issues = [17, 26, 30]
subjects = ['S'+str(i).zfill(2) for i in range(12,31) \
            if i not in imu_issues]

# m_dir = DATA_DIR.copy()

sbj_processed_dir = SBJ_PROCESSED_DIR
m_dir = M_DIR


def normalize_output_directory(output_directory: str) -> str:
    out = str(output_directory).strip().lower()
    if out == "mr":
        return "mr"
    if out in {"level", "levels"}:
        return "levels"
    raise ValueError(f"Unsupported output directory '{output_directory}'")

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", '--model', type=str,
                        default='lstm',
                        choices=['linreg', 'ard', 'xgboost', 'knn',
                                 'svr', 'cnn1d', 'fnn', 'lstm', 'ridge',
                                 'elastic'],
                       )
    parser.add_argument("--strategy", type=str, default='masked',
                        choices=['masked', 'seq2seq', 'gan', 'conv',
                                 'patchtst', 'patchtcn', 'patchattn'])
    parser.add_argument("-f", '--feature_method', type=str,
                        default='None',
                        choices=['tsfresh', 'minirocket', 'None']
                       )
    parser.add_argument('-o', '--overwrite', action='store_true',
                        help='Overwrite data or keep as is')
    parser.add_argument('-e', '--extend', action='store_true',
                        help='Load last checkpoint and extend training')
    parser.add_argument('--freeze', action='store_true',
                        help='Freeze encoder layers')
    parser.add_argument('--data_str', type=str,
                        default='imu_ica',
                        choices=['imu_filt', 'imu_ica']
                       )
    parser.add_argument('--condition', type=str, default='[M,R]',
                        choices=['[M,R]', '[!M]*', 'L*'])
    parser.add_argument('--output-directory', type=str, default='mr',
                        choices=['mr', 'level', 'levels'])
    parser.add_argument('--cls_embed', action='store_true',
                        help='Embed the condition token for patch '\
                        'reconstruction')
    parser.add_argument('--window_size', type=float,
                        default=20,
                        help='Window size for sliding window'\
                        ' procedure, set in seconds'
                       )
    parser.add_argument('--window_shift', type=float,
                        default=1,
                        help='Window shift for sliding window'\
                        ' procedure, set in seconds'
                       )
    parser.add_argument('--n_components', type=int,
                        default=1,
                        help='ICA components')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    return args

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

def set_requires_grad(model, dict_, requires_grad=True):
    for param in model.named_parameters():
        if param[0] in dict_:
            param[1].requires_grad = requires_grad

def load_loocv_dataset(sbj:str, sbj_dicts:list, data_str='imu_ica',
                       reject=False, output_dir='mr'):
    output_dir = normalize_output_directory(output_dir)
    test_list = [sbj_dict for sbj_dict in sbj_dicts if 
                 sbj_dict['subject'] == sbj]
    train_list = [sbj_dict for sbj_dict in sbj_dicts if 
                 sbj_dict['subject'] != sbj]

    def make_dataset(data_list, data_str):
        with open(join(sbj_processed_dir, output_dir, 'label_encoder.pkl'), 'rb') as f:
            le = pickle.load(f)
        x = np.concatenate(
            [data[data_str] for data in data_list], axis=0
        )
        pss = np.concatenate(
            [data['pss_filt'] for data in data_list], axis=0
        )
        br = np.concatenate(
            [data['br'] for data in data_list], axis=0
        )
        cond = np.concatenate(
            [data['conds'] for data in data_list], axis=0
        )
        cond = le.transform(cond)

        return x, pss, br, cond

    x_test, y_test, br_test, c_test = make_dataset(test_list, data_str)
    x_train, y_train, br_train, c_train = make_dataset(train_list, data_str)

    if reject:
        train_idxs = [
            i for i, data in enumerate(y_train) if not reject_artefact(data)
        ]
        test_idxs = [
            i for i, data in enumerate(y_test) if not reject_artefact(data)
        ]
        x_train, x_test = x_train[train_idxs], x_test[test_idxs]
        y_train, y_test = y_train[train_idxs], y_test[test_idxs]
        br_train, br_test = br_train[train_idxs], br_test[test_idxs]
        c_train, c_test = c_train[train_idxs], c_test[test_idxs]

    train_data = (x_train, y_train)
    test_data = (x_test, y_test)
    br_data = (br_train, br_test)
    cond_data = (c_train, c_test)

    return train_data, test_data, br_data, cond_data

# Load data to dataloader
class LoadDataset(Dataset):
    def __init__(self, x, y=None, cond=None, aug_ratio=0.3):
        self.len = len(x)
        if isinstance(x, np.ndarray):
            self.x = torch.from_numpy(x)
        else:
            self.x = x

        # make sure the Channels in second dim
        if self.x.shape.index(min(self.x.shape)) != 2:
            self.x = self.x.permute(0, 2, 1)

        if y is not None and isinstance(y, np.ndarray):
            self.y = torch.from_numpy(y).float()
        elif y is not None and not isinstance(y, np.ndarray):
            self.y = y.float()
        else:
            self.y = y

        if cond is not None and isinstance(cond, np.ndarray):
            self.cond = torch.from_numpy(cond).float()
        elif cond is not None and not isinstance(cond, np.ndarray):
            self.cond = cond.float()
        else:
            self.cond = cond

        if aug_ratio > 0:
            # Choose ratio random percentage
            aug_idxs = random.sample(range(len(x)), int(aug_ratio*len(x)))
            x_aug, y_aug = self.x[aug_idxs], self.y[aug_idxs]

            if cond is not None:
                c_aug = self.cond[aug_idxs]
                self.cond = torch.cat((self.cond, c_aug), dim=0)

            # channels first
            x_aug = x_aug.permute(0, 2, 1)
            # x_aug = dom_shuffle(x_aug, rate=3, dim=-1)
            x_aug = scaling(x_aug, sigma=1.1, device=x.device)
            x_aug = x_aug.permute(0, 2, 1)
            self.x = torch.cat((self.x, x_aug), dim=0)
            self.y = torch.cat((self.y, y_aug), dim=0)


    def __getitem__(self, index):
        if self.y is not None and self.cond is not None:
            return self.x[index], self.y[index], self.cond[index]
        elif self.y is not None:
            return self.x[index], self.y[index]
        else:
            item = {
                'past_values': self.x[index].float(),
                'past_observed_mask': torch.ones_like(self.x[index]).float(),
            }
            return item

    def __len__(self):
        return self.len

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in, out_channels=d_model,
            kernel_size=3, padding=padding, padding_mode='circular', 
            bias=False
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq
        ) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)

# Establish model to generate original IMU data
class EncoderRNN(nn.Module):
    def __init__(self, input_shape, hidden_size, dropout_p=0.1, stride=5,
                 patch_size=10):
        super(EncoderRNN, self).__init__()
        self.input_shape = input_shape
        self.window_size = input_shape[0]
        self.n_channels  = input_shape[1]
        self.hidden_size = hidden_size
        self.stride      = stride
        self.patch_size  = patch_size

        self.patch_num = (self.window_size - self.patch_size) // self.stride + 1
        self.patch_num += 1
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 

        # self.embedding = DataEmbedding(self.n_channels, self.hidden_size, dropout_p)
        self.embedding = DataEmbedding(self.n_channels * self.patch_size,
                                       self.hidden_size, dropout_p)
        # self.embedding = nn.Embedding(input_shape, hidden_size)
        self.input = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input, return_hidden=False, **kwargs):
        # embedded = self.dropout(self.embedding(input))
        input_x = rearrange(input, 'b l c -> b c l')
        input_x = self.padding_patch_layer(input_x)
        input_x = input_x.unfold(dimension=-1, size=self.patch_size, 
                                 step=self.stride)

        # patches and channels mixed (patches for each channel now in one dim)
        # second dim are the temporal patches
        # in a batch, across time, here are the patches for the channels
        input_x = rearrange(input_x, 'b c n p -> b n (p c)')
        embedded = self.embedding(input_x) # embedding already deals with dropout

        output, h0 = self.input(self.dropout(embedded))
        output, hidden = self.gru(self.dropout(output))
        if return_hidden:
            return output, hidden
        else:
            return output

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        # self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding = DataEmbedding(hidden_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru0 = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.gru1 = nn.GRU(hidden_size, hidden_size, batch_first=True)

        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty((batch_size, 1, self.hidden_size), 
                                    dtype=torch.float, device=device).fill_(0)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        decoder_output, decoder_hidden, attn_weights = self.forward_step(
            decoder_input, decoder_hidden, encoder_outputs
        )
        decoder_outputs.append(decoder_output.squeeze())
        attentions.append(attn_weights)

        """
        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                # Teacher forcing
                # FIXME Make this map a time locked section of the patch to a
                # several units of the target tensor
                decoder_input = target_tensor[:, i].unsqueeze(1)
                decoder_input = decoder_input.unsqueeze(1)
            else:
                # Without teacher forcing: use its own predictions as the next 
                # input
                _, topi = decoder_output.topk(1)
                # detach from history as input
                decoder_input = topi.squeeze(-1).detach()  

        """

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        # decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru0(input_gru, hidden)
        output, hidden = self.gru1(output, hidden)
        output = self.out(self.dropout(output))

        return output, hidden, attn_weights

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3,
                 stride=1,
                 downsample=None,
                 padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv1d(in_channels, out_channels,
                                  kernel_size=kernel_size, 
                                  stride=stride, padding=padding),
                        nn.BatchNorm1d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv1d(out_channels, out_channels,
                                  kernel_size=kernel_size,
                                  stride=stride, padding=padding),
                        nn.BatchNorm1d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ConvTransformerEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=128, n_heads=4, n_layers=4,
                 mask_prob=0.2, kernel_size=5, stride=2):
        super().__init__()

        self.mask_prob = mask_prob
        padding = kernel_size // 2

        # Convolutional feature extractor
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size, stride, 
                      padding=padding),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.conv2 = ResidualBlock(hidden_dim, hidden_dim,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        # self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, stride, padding=kernel_size // 2)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,   # [B, T, C]
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.positional_encoding = PositionalEncoding(hidden_dim)

    def forward(self, x):
        # x: [batch, channels, T_in]
        if x.shape[1] > x.shape[2]:
            x = x.permute(0, 2, 1)

        # # --- Input masking ---
        # if self.training and self.mask_prob > 0:
        #     mask = (torch.rand(x.shape[0], 1, x.shape[2], device=x.device) > self.mask_prob).float()
        #     x = x * mask

        # --- Conv feature extraction ---
        h = self.conv1(x)
        h = self.conv2(h)   # [B, C, T_enc]

        # Prepare for transformer: [B, T, C]
        h = h.transpose(1, 2)

        # --- Transformer encoding ---
        h = self.positional_encoding(h)
        h = self.transformer(h)     # [B, T_enc, hidden_dim]

        # Back to [B, C, T]
        h = h.transpose(1, 2)

        return h


# ---------- Decoder ----------
class ConvResampleDecoder(nn.Module):
    def __init__(self, hidden_dim=128, out_channels=1, upsample_factors=[2, 2, 2]):
        super().__init__()
        layers = []
        in_dim = hidden_dim
        for factor in upsample_factors:
            layers.append(
                nn.ConvTranspose1d(
                    in_dim, hidden_dim, kernel_size=factor*2, stride=factor, 
                    padding=factor//2
                )
            )
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.deconv_layers = nn.Sequential(*layers)
        self.final_conv = nn.Conv1d(hidden_dim, out_channels, kernel_size=3, 
                                    padding=1)

    def forward(self, h, T_out):
        y_hat = self.deconv_layers(h)
        y_hat = F.interpolate(y_hat, size=T_out, mode="linear", 
                              align_corners=True)
        return self.final_conv(y_hat)

class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim=128, out_channels=1, n_heads=4, n_layers=4):
        super().__init__()
        # self.fuse_conv = nn.Conv1d(hidden_dim + out_channels, hidden_dim, kernel_size=3, padding=1)
        self.fuse_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=n_layers)
        self.positional_encoding = PositionalEncoding(hidden_dim)

        self.final_conv = nn.Conv1d(hidden_dim, out_channels, kernel_size=3, padding=1)

    def forward(self, h, y_partial, T_out):
        # Upsample encoder representation to match target length
        h = F.interpolate(h, size=T_out, mode="linear", align_corners=True)

        # Fuse latent + partial target
        # fused = torch.cat([h, y_partial], dim=1)  # [B, hidden_dim+1, T_out]
        fused = F.relu(self.fuse_conv(h))     # [B, hidden_dim, T_out]

        fused = fused.transpose(1, 2)             # [B, T_out, hidden_dim]
        fused = self.positional_encoding(fused)
        fused = self.transformer(fused)

        fused = fused.transpose(1, 2)             # [B, hidden_dim, T_out]
        return self.final_conv(fused)             # [B, 1, T_out]

# ---------- Full Model ----------
class ConvTransformerMaskedAutoencoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        hidden_dim=128,
        out_channels=1,
        mask_prob=0.2,
        encoder_stride=10, # 2
        upsample_factors=[2, 2, 2],
        n_heads=4,
        n_layers=2, #4
    ):
        super().__init__()
        self.encoder = ConvTransformerEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            mask_prob=mask_prob,
            kernel_size=5,
            stride=encoder_stride,
        )
        self.decoder = TransformerDecoder(
            hidden_dim, out_channels, n_heads, n_layers)

    def forward(self, x, y_partial, mask, T_out, return_hidden=False):
        if x.shape[1] > x.shape[-1]:
            x = x.permute(0, 2, 1)

        h = self.encoder(x)
        y_pred_full = self.decoder(h, y_partial, T_out)
        y_hat = y_pred_full*(1-mask)
        if return_hidden:
            return y_hat, h
        else:
            return y_hat


class ConvAutoencoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        hidden_dim=128,
        out_channels=1,
        encoder_stride=10, # 2
        upsample_factors=[2, 2, 2],
        n_heads=4,
        n_layers=2, #4
    ):
        super().__init__()
        self.encoder = ConvTransformerEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=5,
            stride=encoder_stride,
        )
        self.decoder = ConvResampleDecoder(
            hidden_dim=hidden_dim, out_channels=out_channels,
            upsample_factors=upsample_factors)

    def forward(self, x, T_out, return_hidden=False):
        if x.shape[1] > x.shape[-1]:
            x = x.permute(0, 2, 1)

        h = self.encoder(x)
        y_hat = self.decoder(h, T_out)
        if return_hidden:
            return y_hat, h
        else:
            return y_hat

def create_output_mask(batch_size, length, mask_prob=0.3, device="cpu", segment_size=20):
    # Creates a binary mask for output signal (1 = keep, 0 = mask).
    mask = torch.ones(batch_size, 1, length, device=device)

    for b in range(batch_size):
        num_segments = int(length * mask_prob / segment_size)
        for _ in range(num_segments):
            start = torch.randint(0, length - segment_size, (1,))
            mask[b, :, start:start+segment_size] = 0.0
    return mask

def masked_loss(y_hat, y_true, mask):
    """
    Compute loss only over masked regions.
    """
    diff = (y_hat - y_true) ** 2
    masked_mse = (diff * (1 - mask)).sum() / ((1 - mask).sum() + 1e-8)
    return masked_mse


# Set up masked encoder training sequence
def mae_train_epoch(dataloader, model, optimizer, mask_prob=0.3):

    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data

        optimizer.zero_grad()
        
        if input_tensor.shape[1] > input_tensor.shape[-1]:
            # likely wrong shape [B, T, C]
            input_tensor = input_tensor.permute(0, 2, 1)

        input_tensor = input_tensor.float().to(device)
        target_tensor = target_tensor.float().to(device)

        T_out = target_tensor.shape[1]
        batch_size = len(input_tensor)

        # masking
        mask = create_output_mask(batch_size, T_out, mask_prob=mask_prob,
                                  device=device)
        if target_tensor.ndim < 3:
            y_partial = torch.unsqueeze(target_tensor, 1)
        else:
            y_partial = target_tensor.copy()

        y_partial *= mask

        outputs = model(input_tensor, y_partial, mask, T_out)

        if outputs.ndim > target_tensor.ndim:
            outputs = outputs.squeeze()

        loss = masked_loss(outputs, target_tensor, mask.squeeze())
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def mae_train(train_dataloader, model, n_epochs, learning_rate=1e-3,
              mask_prob=0.3, print_every=10):
    start = time.time()
    print_loss_total = 0  # Reset every print_every

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, n_epochs + 1):
        loss = mae_train_epoch(train_dataloader, model, optimizer,
                               mask_prob=mask_prob)
        print_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                         epoch, epoch / n_epochs * 100, 
                                         print_loss_avg))


def conv_train_epoch(dataloader, model, optimizer, criterion):

    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data

        optimizer.zero_grad()
        
        if input_tensor.shape[1] > input_tensor.shape[-1]:
            # likely wrong shape [B, T, C]
            input_tensor = input_tensor.permute(0, 2, 1)

        input_tensor = input_tensor.float().to(device)
        target_tensor = target_tensor.float().to(device)

        T_out = target_tensor.shape[1]
        batch_size = len(input_tensor)

        outputs = model(input_tensor, T_out)

        if outputs.ndim > target_tensor.ndim:
            outputs = outputs.squeeze()

        loss = criterion(outputs, target_tensor)
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def conv_train(train_dataloader, model, n_epochs, learning_rate=1e-3,
               print_every=10):
    start = time.time()
    print_loss_total = 0  # Reset every print_every

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.L1Loss(reduction='sum')

    for epoch in range(1, n_epochs + 1):
        loss = conv_train_epoch(train_dataloader, model, optimizer, criterion)
        print_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                         epoch, epoch / n_epochs * 100, 
                                         print_loss_avg))

def noam_lr_lambda(step, d_model, warmup_steps):
    step = max(step, 1)  # Ensure step is at least 1 to avoid division by zero
    return (d_model**-0.5) * min(step**-0.5, step * (warmup_steps**-1.5))

class ConcordanceCorrCoefLoss(nn.Module):
    """
    Concordance Correlation Coefficient (CCC) combined with L1 loss.
    Loss = (1 - CCC) + lambda * L1
    """
    def __init__(self, l1_weight:float=1.0, eps:float=1e-8):
        super().__init__()
        self.l1_weight = l1_weight
        self.eps = eps
        self.l1 = nn.L1Loss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Ensure same shape
        assert y_pred.shape == y_true.shape,\
                f"Shape mismatch: {y_pred.shape} vs {y_true.shape}"

        # Flatten only non-batch dims → shape: (B, -1)
        y_pred = y_pred.reshape(y_pred.size(0), -1)
        y_true = y_true.reshape(y_true.size(0), -1)

        # Per-batch means
        mean_true = torch.mean(y_true, dim=1, keepdim=True)
        mean_pred = torch.mean(y_pred, dim=1, keepdim=True)

        # Variances
        var_true = torch.var(y_true, dim=1, unbiased=False)
        var_pred = torch.var(y_pred, dim=1, unbiased=False)

        # Covariance
        cov = torch.mean((y_true - mean_true) * (y_pred - mean_pred), dim=1)

        # Concordance Correlation Coefficient per sample
        ccc = (2 * cov) / (var_true + var_pred + \
                           (mean_true.squeeze(1) - mean_pred.squeeze(1)) ** 2 \
                           + self.eps)

        # CCC loss (average across batch)
        ccc_loss = torch.mean(1 - ccc)

        # L1 loss across batch
        # NOTE
        # If you want correlation to dominate, keep l1_weight < 1.
        # If you want absolute error reduction to dominate, increase l1_weight.
        l1_loss = self.l1(y_pred, y_true)

        # Combined loss
        return ccc_loss + self.l1_weight * l1_loss

def patchtst_train_epoch(dataloader, model, optimizer, criterion,
                         mask_ratio=0.3, amp_alpha=1.0, phase_alpha=1.0, 
                         scheduler=None, scaler=None, device='cpu',
                         cls_embed=False, cls_criterion=None, cls_alpha=0.1,
                         **kwargs):

    total_loss = 0
    total_temporal_loss = 0
    total_amp_loss = 0
    total_phase_loss = 0

    for data in dataloader:
        if len(data) == 2:
            input_tensor, target_tensor = data
        else:
            input_tensor, target_tensor, cls_tensor = data

            cls_out = cls_tensor.view(-1, 1, 1)
            cls_out = torch.ones_like(input_tensor) * cls_out

            cls_patches, _ = segment_to_patches(cls_out, model.patch_len,
                                               axis=1)
            # (batch, patches, patch_size, nch)
            cls_patches = torch.Tensor(cls_patches).to(device)

            # xb:  bs, nch, patch_len, patch_num
            cls_patches = cls_patches.permute(0, 3, 2, 1)
            
            batch_size = input_tensor.size(0)
            # only need 1-value for embed token
            cls_patches = cls_patches[:, 0, 0, :].long()\
                    .view(batch_size, 1, 1, -1)
            kwargs.update({'cls_label': cls_patches})

        if target_tensor.ndim == 2:
            target_tensor = target_tensor.unsqueeze(-1)

        target_patches, _ = segment_to_patches(target_tensor, model.patch_len,
                                           axis=1)
        
        # (batch, patches, patch_size, nch)
        target_patches = torch.Tensor(target_patches).float().to(device)

        # Try reconstruct the FFT frequency and phase response of each patch
        # input:  bs, n_ch, patch_num, patch_len
        target_fft_patches = target_patches.permute(0, 3, 1, 2)
        # output:  bs, nch, patch_len, patch_num
        target_fft_amplitude, target_fft_phase = torch_fft_targets(
            target_fft_patches)

        # xb:  bs, nch, patch_len, patch_num
        target_patches = target_patches.permute(0, 3, 2, 1)

        # target_masked, target_kept, mask, ids_restore = model._patch_masking(
        #     target_patches, mask_ratio)
        optimizer.zero_grad()
        
        input_tensor = input_tensor.float().to(device)
        input_tensor = input_tensor.permute(0, 2, 1)

        target_tensor = target_tensor.float().to(device)
        target_tensor = target_tensor.permute(0, 2, 1)

        T_out = target_tensor.shape[2]
        batch_size = len(input_tensor)

        if scaler is not None:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
               recon = model(input_tensor, **kwargs)
               recon = recon.permute(0, 1, 3, 2)
        else:
            recon = model(input_tensor, **kwargs)
            recon = recon.permute(0, 1, 3, 2)

        # outputs = model(input_tensor, **kwargs)
        # recon, recon_amplitude, recon_phase, features = outputs
        # recon = recon.permute(0, 1, 3, 2)
        # recon_amplitude = recon_amplitude.permute(0, 1, 3, 2)
        # recon_phase = recon_phase.permute(0, 1, 3, 2)

        # print('temporal\t\trecon: {}\ttarget: {}'\
        #       .format(recon.shape, target_patches.shape))
        # print('amplitude\t\trecon: {}\ttarget: {}'\
        #       .format(recon_amplitude.shape, target_fft_amplitude.shape))
        # print('phase\t\trecon: {}\ttarget: {}'\
              # .format(recon_phase.shape, target_fft_phase.shape))

        temporal_loss = criterion(recon, target_patches)
        loss = temporal_loss

        # This made regression worse
        # if cls_embed:
        #     cls_loss = cls_criterion(logits.squeeze(),
        #                              cls_patches.float().squeeze())
        #     loss += cls_loss*cls_alpha

        # amplitude_loss = criterion(recon_amplitude, target_fft_amplitude)
        # phase_loss = criterion(recon_phase, target_fft_phase)

        # loss = temporal_loss \
        #         + amplitude_loss*amp_alpha \
        #         + phase_loss*phase_alpha

        # print(
        #     "loss: {}\ttemporal loss: {}\tamplitude loss: {}\tphase loss: {}"\
        #     .format(loss, temporal_loss, amplitude_loss, phase_loss)
        # )

        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)   # may be skipped if gradients invalid
            scaler.update()

        total_loss += loss.item()
        # total_temporal_loss += temporal_loss
        # total_amp_loss += amplitude_loss*amp_alpha
        # total_phase_loss += phase_loss*phase_alpha
    if scheduler is not None:
        scheduler.step()

    return total_loss / len(dataloader)
    # return (total_loss / len(dataloader), total_temporal_loss/len(dataloader),
    #         total_amp_loss/len(dataloader), total_phase_loss/len(dataloader))


def patchtst_train(train_dataloader, model, n_epochs, learning_rate=1.0,
                   print_every=10, device='cpu', n_classes=None, **kwargs):
    start = time.time()
    print_loss_total = 0  # Reset every print_every

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    # criterion = nn.L1Loss(reduction='sum')
    criterion = ConcordanceCorrCoefLoss()
    scaler = torch.amp.GradScaler(device)
    if n_classes-1 == 1:
        cls_criterion = nn.BCEWithLogitsLoss()
    else:
        cls_criterion = nn.CrossEntropyLoss()

    d_model = model.d_model
    warmup_steps = 4000
    lr_lambda = lambda step: noam_lr_lambda(step, d_model, warmup_steps)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    history = []

    for epoch in range(1, n_epochs + 1):
        loss = patchtst_train_epoch(train_dataloader, model, optimizer,
                                    criterion, scheduler=scheduler,
                                    scaler=scaler, device=device,
                                    cls_criterion=cls_criterion, **kwargs)
        # loss, temporal_loss, amplitude_loss, phase_loss = losses
        print_loss_total += loss

        history.append({'loss': loss,})
                        # 'temporal_loss': temporal_loss,
                        # 'amplitude_loss': amplitude_loss,
                        # 'phase_loss': phase_loss})

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                         epoch, epoch / n_epochs * 100, 
                                         print_loss_avg))
    return history

class PretrainedRegressor(nn.Module):
    def __init__(self, pretrained_model, hidden_size, dropout_p=0.2, 
                 activation='ReLU', scalar=24, strategy='seq2seq'):
        super(PretrainedRegressor, self).__init__()
        self.output_size = 1
        self.pretrained_model = pretrained_model
        self.hidden_size = hidden_size
        self.hidden_layer = nn.Linear(int(hidden_size*scalar), hidden_size)
        self.out = nn.Linear(hidden_size, self.output_size)
        self.dropout = nn.Dropout(dropout_p)
        self.strategy = strategy
        if activation == None:
            self.activation = activation
        else:
            self.activation = getattr(nn, activation)()

    def forward(self, input, T_out=None, channel_first:bool=False,
                **kwargs):
        hidden = None

        tmp = self.pretrained_model(input, **kwargs)

        if isinstance(tmp, tuple):
            output, hidden = tmp
        else:
            output = tmp

        output = nn.Flatten(start_dim=1)(output)
        output = self.hidden_layer(self.dropout(output))
        # output = torch.mean(output, axis=1)
        if self.activation is not None:
            output = self.activation(output)
        output = self.out(self.dropout(output))
        return output, hidden

# post proc labels
def regression_post_process(prds, pad_len=50):
    mov_mean_win = 8
    prds_postproc = movingaverage(
        butter_lowpass_filter(
            np.pad(
                prds, (pad_len, ), 'symmetric'
            ), 0.1, 2, order=5
        ), mov_mean_win
    )[pad_len:-pad_len]

    return prds_postproc

# Set up seq2seq training sequence
def seq2seq_train_epoch(dataloader, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, criterion):

    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data
        input_tensor = input_tensor.float().to(device)
        target_tensor = target_tensor.float().to(device)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor,
                                                  return_hidden=True)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, 
                                        target_tensor)
        # loss = criterion(
        #     decoder_outputs.view(-1, decoder_outputs.size(-1)),
        #     target_tensor.view(-1)
        # )
        loss = criterion(decoder_outputs, target_tensor)
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def seq2seq_train(train_dataloader, encoder, decoder, n_epochs, learning_rate=1e-3,
               print_every=10):
    start = time.time()
    print_loss_total = 0  # Reset every print_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.L1Loss(reduction='sum')

    for epoch in range(1, n_epochs + 1):
        loss = seq2seq_train_epoch(train_dataloader, encoder, decoder, 
                                   encoder_optimizer, decoder_optimizer, 
                                   criterion)
        print_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                         epoch, epoch / n_epochs * 100, 
                                         print_loss_avg))

# Pretrain
def do_pretraining(
    x_train, x_test, y_train, y_test,
    c_train, c_test,
    strategy='seq2seq',
    patch_size=10,
    stride=10,
    overwrite=False,
    learning_rate=1e-3,
    n_epochs=3000,
    batch_size=32,
    hidden_size=128,
    shuffle=True,
    drop_last=True,
    n_channels=6,
    output_size=MAX_LENGTH,
    n_plots=10,
    s_mdl_dir='./tmp',
    sbj='S12',
    print_every=10,
    mask_prob=0.3,
    device='cpu',
    extend=False,
    data_str='imu_filt',
    cls_embed=False,
    n_classes=None,
):
    plt_dir = join('plots', strategy)
    makedirs(plt_dir, exist_ok=True)

    lbl_fname  = 'labels.npy'
    pred_fname = 'preds.npy'
    plot_fname = join(plt_dir, f'{sbj}_{data_str}_{strategy}_plots.png')
    rng = np.random.default_rng()

    if cls_embed:
        train_data_to_load = (x_train, y_train, c_train)
        test_data_to_load = (x_test, y_test, c_test)
    else:
        train_data_to_load = (x_train, y_train)
        test_data_to_load = (x_test, y_test)


    train_dataloader = DataLoader(LoadDataset(*train_data_to_load),
                                  batch_size=batch_size, shuffle=shuffle,
                                  drop_last=drop_last, num_workers=0)

    load_from = join(s_mdl_dir, 'ckp_last.pt')
    if strategy == 'masked':
        # create encoder and decoder for masked autoencoding variant
        mae_model = ConvTransformerMaskedAutoencoder(in_channels=n_channels,
                                                     mask_prob=mask_prob)
        mae_model = mae_model.to(device)
        encoder = mae_model.encoder
        decoder = mae_model.decoder
        if overwrite or not exists(load_from):
            mae_train(train_dataloader, mae_model, n_epochs,
                      learning_rate=learning_rate, print_every=print_every)
            torch.save(encoder.state_dict(), load_from)
        else:
            print("loading from: ", load_from)
            chkpoint = torch.load(load_from, map_location=device,)
            encoder.load_state_dict(chkpoint)
    elif strategy == 'conv':
        # create encoder and decoder for masked autoencoding variant
        model = ConvAutoencoder(in_channels=n_channels)
        mae_model = model.to(device)
        encoder = model.encoder
        decoder = model.decoder
        if overwrite or not exists(load_from):
            conv_train(train_dataloader, model, n_epochs, 
                       learning_rate=learning_rate, print_every=print_every)
        elif extend:
            chkpoint = torch.load(load_from, map_location=device,)
            print("loading from ", load_from)
            encoder.load_state_dict(chkpoint['encoder_state_dict'])
            decoder.load_state_dict(chkpoint['decoder_state_dict'])
            conv_train(train_dataloader, model, n_epochs, 
                       learning_rate=learning_rate, print_every=print_every)
        else:
            print("loading from: ", load_from)
            chkpoint = torch.load(load_from, map_location=device,)
            encoder.load_state_dict(chkpoint['encoder_state_dict'])
            decoder.load_state_dict(chkpoint['decoder_state_dict'])
    elif 'patch' in strategy:
        # create encoder and decoder for patch autoencoding variant
        bs, context_length, n_ch = x_train.shape
        _, target_len = y_train.shape
        target_ch = 1

        model_instructions = {'reconstruct_target': True,
                              'cls_embed': cls_embed}
        # NOTE: trying revin as true
        backbone_kwargs = {'revin': True,
                           'cls_embed': cls_embed,
                           'pe': 'sincos',
                           'out_patch_size': out_patch_size,
                           'out_stride': out_stride}

        if strategy == 'patchtcn':
            backbone_kwargs.update({'tcn': True})
        elif strategy == 'patchattn':
            backbone_kwargs.update({'attn_tcn': True})

        if n_classes is not None:
            backbone_kwargs.update({'n_classes': n_classes})
            

        model = PatchTST_backbone(
            n_ch, context_length, target_len, patch_size, stride,
            **backbone_kwargs).to(device)
        encoder = model.backbone
        decoder = model.reconstruction_head
        if overwrite or not exists(load_from):
            history = patchtst_train(train_dataloader, model, n_epochs, 
                                     learning_rate=learning_rate,
                                     print_every=print_every,
                                     device=device,
                                     n_classes=n_classes,
                                     **model_instructions)
            hist = pd.DataFrame(history)
            hist.to_csv(join(s_mdl_dir, 'history.csv'), index=False)
        elif extend:
            chkpoint = torch.load(load_from, map_location=device,)
            print("loading from ", load_from)
            encoder.load_state_dict(chkpoint['encoder_state_dict'])
            decoder.load_state_dict(chkpoint['decoder_state_dict'])
            history = patchtst_train(train_dataloader, model, n_epochs, 
                                     learning_rate=learning_rate,
                                     print_every=print_every,
                                     device=device,
                                     n_classes=n_classes,
                                     **model_instructions)
            hist = pd.read_csv(join(s_mdl_dir, 'history.csv'))
            hist = pd.concat([hist, pd.DataFrame(history)])\
                    .reset_index(drop=True)
            hist.to_csv(join(s_mdl_dir, 'history.csv'), index=False)
        else:
            print("loading from: ", load_from)
            chkpoint = torch.load(load_from, map_location=device,)
            encoder.load_state_dict(chkpoint['encoder_state_dict'])
            decoder.load_state_dict(chkpoint['decoder_state_dict'])
    elif strategy == 'seq2seq':
        encoder = EncoderRNN(x_train.shape[1:], hidden_size).to(device)
        decoder = AttnDecoderRNN(hidden_size, output_size).to(device)
        if overwrite or not exists(load_from):
            seq2seq_train(train_dataloader, encoder, decoder, n_epochs, 
                  learning_rate=learning_rate)
        elif extend:
            print("loading from ", load_from)
            chkpoint = torch.load(load_from, map_location=device,)
            encoder.load_state_dict(chkpoint['encoder_state_dict'])
            decoder.load_state_dict(chkpoint['decoder_state_dict'])
            seq2seq_train(train_dataloader, encoder, decoder, n_epochs, 
                  learning_rate=learning_rate)
        else:
            print("loading from: ", load_from)
            chkpoint = torch.load(load_from, map_location=device,)
                                  # weights_only=True)
            pretrain_encoder_dict = chkpoint["encoder_state_dict"]
            pretrain_decoder_dict = chkpoint["decoder_state_dict"]

            encoder_dict = encoder.state_dict()
            decoder_dict = decoder.state_dict()

            encoder_dict.update(pretrain_encoder_dict)
            decoder_dict.update(pretrain_decoder_dict)

            encoder.load_state_dict(pretrain_encoder_dict)
            decoder.load_state_dict(pretrain_decoder_dict)

    if overwrite \
       or not exists(join(s_mdl_dir, f'ckp_last.pt')) \
       or not exists(plot_fname) \
       or extend:
        train_idxs = rng.choice(len(y_train), size=n_plots, replace=False)
        test_idxs  = rng.choice(len(y_test), size=n_plots, replace=False)

        if x_train.shape[1] > 1e3:
            train_dec_out = []
            for data in train_dataloader:
                if cls_embed:
                    xb, yb, cb = data
                else:
                    xb, yb = data

                if strategy == 'masked':
                    outputs = encoder(xb.to(device).float())

                    mask = create_output_mask(batch_size, yb.shape[-1],
                                              mask_prob=mask_prob,
                                              device='cpu')
                    y_partial = yb.unsqueeze(1)*mask
                    y_hat = decoder(outputs, y_partial.to(device),
                                    yb.shape[-1]).detach().cpu()
                    for i in range(batch_size):
                        plt.subplot(4, 4, i+1)
                        plt.plot(yb[i].squeeze())
                        plt.plot(y_hat[i].squeeze())

                    # y_pred_full = self.decoder(h, y_partial, T_out)

                    plt.show()
                    ipdb.set_trace()
                    train_enc_out = outputs.prediction_output
                    train_enc_hidden = outputs.hidden_states
                elif strategy == 'conv':
                    train_dec_out_ = model(xb.to(device).float(), yb.shape[-1])
                    train_dec_out.append(train_dec_out_.detach().cpu().numpy())
                elif 'patch' in strategy:
                    x_in = xb.to(device).float().permute(0, 2, 1)
                    recon = model(x_in, **model_instructions)
                    '''
                    outputs = model(x_in, **model_instructions)
                    recon, recon_amplitude, recon_phase, features = outputs

                    recon = recon.permute(0, 1, 3, 2)
                    recon_amplitude = recon_amplitude.permute(0, 1, 3, 2)
                    recon_phase = recon_phase.permute(0, 1, 3, 2)
                    '''

                    encoder = model.backbone.to(device)
                    decoder = model.reconstruction_head.to(device)

                    # TODO: this needs to be tested
                    train_dec_out.append(recon.reshape(yb.shape).detach().cpu().numpy())
                else:
                    train_enc_out, train_enc_hidden = encoder(xb.to(device).float())
                    train_dec_out_, _, _ = decoder(train_enc_out, train_enc_hidden)
                    train_dec_out.append(train_dec_out_.detach().cpu().numpy())
            train_dec_out = np.vstack(train_dec_out)
        else:
            train_enc_out, train_enc_hidden = encoder(
                torch.Tensor(x_train).to(device))
            train_dec_out, _, _ = decoder(train_enc_out, train_enc_hidden)
            train_dec_out = train_dec_out.detach().cpu().numpy()

        if strategy not in ['masked', 'conv'] and 'patch' not in strategy:
            test_enc_out, test_enc_hidden = encoder(torch.Tensor(x_test).to(device))
            test_dec_out, _, _ = decoder(test_enc_out, test_enc_hidden)
        elif strategy == 'conv':
            test_enc_out = encoder(torch.Tensor(x_test).to(device))
            test_dec_out = decoder(test_enc_out, yb.shape[-1])
        elif 'patch' in strategy:
            dataloader = DataLoader(
                LoadDataset(*test_data_to_load, aug_ratio=0.),
                batch_size=batch_size, shuffle=False,
                drop_last=drop_last, num_workers=0)

            test_dec_out = []
            for data in dataloader:
                if cls_embed:
                    x, y, _ = data
                else:
                    x, y = data

                x = torch.Tensor(x).to(device).float().permute(0, 2, 1)
                x = x.unfold(dimension=-1, size=model.patch_len,
                                   step=model.stride)
                x = x.permute(0,1,3,2)   # x_in: [bs x nvars x patch_len x patch_num]

                test_enc_out = encoder(x)
                test_dec_out.append(
                    decoder(test_enc_out).detach().cpu().numpy())
            test_dec_out = np.concatenate(test_dec_out, axis=0)
            bs, n_ch, n_patches, patch_len = test_dec_out.shape
            test_dec_out = np.reshape(test_dec_out,
                                      [bs, n_patches*patch_len])

        if not isinstance(test_dec_out, np.ndarray):
            test_dec_out = test_dec_out.detach().cpu().numpy()

        if train_dec_out.ndim > 2:
            train_dec_out = train_dec_out.squeeze()

        if test_dec_out.ndim > 2:
            test_dec_out = test_dec_out.squeeze()

        train_ex = y_train[train_idxs]
        test_ex  = y_test[test_idxs]
        fig, axs = plt.subplots(n_plots, 2, figsize=(7, 6))

        for i in range(n_plots):
            axs[i, 0].plot(train_ex[i])
            axs[i, 0].plot(train_dec_out[train_idxs][i])

            axs[i, 1].plot(test_ex[i])
            axs[i, 1].plot(test_dec_out[test_idxs][i])

        axs[0, 0].set_title("training set")
        axs[0, 1].set_title("testing set")

        fig.savefig(plot_fname)
        # Save encoder and decoder weights
        chkpoint = {
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict()
        }
        torch.save(chkpoint, join(s_mdl_dir, f'ckp_last.pt'))


        train_evals, train_cond_evals = get_evals(
            y_train, train_dec_out, c_train)
        test_evals, test_cond_evals = get_evals(
            y_test, test_dec_out, c_test)

        with open(join(s_mdl_dir, 'train_eval.pkl'), 'wb') as f:
            pickle.dump(train_evals, f)
        with open(join(s_mdl_dir, 'train_cond_eval.pkl'), 'wb') as f:
            pickle.dump(train_cond_evals, f)

        with open(join(s_mdl_dir, 'test_eval.pkl'), 'wb') as f:
            pickle.dump(test_evals, f)
        with open(join(s_mdl_dir, 'test_cond_eval.pkl'), 'wb') as f:
            pickle.dump(test_cond_evals, f)

        # Save test label and prediction outputs
        np.save(join(s_mdl_dir, 'labels.npy'), y_test)
        np.save(join(s_mdl_dir, f'pred.npy'), test_dec_out)

    return encoder, decoder

def get_evals(lbl, pred, conds):
    evals = Evaluation(lbl, pred).get_evals()

    cond_list = np.unique(conds)
    cond_evals = {}
    for c_str in cond_list:
        mask = conds == c_str
        lbl_c, pred_c = lbl[mask], pred[mask]
        cond_evals[c_str] = Evaluation(lbl_c, pred_c).get_evals()

    return evals, cond_evals

# Fine tune
def do_fine_tuning(regressor,
                   x, y,
                   validation_split=0.25,
                   overwrite=False,
                   learning_rate=1e-3,
                   n_epochs=30,
                   batch_size=32,
                   hidden_size=128,
                   shuffle=True,
                   drop_last=False,
                   n_channels=6,
                   output_size=1,
                   n_plots=10,
                   s_mdl_dir='./tmp',
                   print_every=10,
                   strategy='seq2seq',
                   patch_size=5,
                   stride=5,
                   device='cpu',
                   **kwargs):

    optimizer = optim.Adam(regressor.parameters(), lr=learning_rate)
    criterion = nn.L1Loss()
    x_train, x_val, y_train, y_val = train_test_split(
        x.copy(), y.copy(), test_size=validation_split)

    x_val = torch.Tensor(x_val).to(device)
    y_val = torch.Tensor(y_val).to(device)

    dataloader = DataLoader(LoadDataset(x_train, y_train, aug_ratio=0.),
                            batch_size=batch_size, shuffle=shuffle,
                            drop_last=drop_last, num_workers=0)
    val_dataloader = DataLoader(LoadDataset(x_val, y_val, aug_ratio=0.),
                                batch_size=batch_size, shuffle=shuffle,
                                drop_last=drop_last, num_workers=0)

    training_history, val_history = [], []
    print_loss_total = 0  # Reset every print_every
    val_loss_total = 0  # Reset every print_every

    start = time.time()
    for epoch in range(1, n_epochs + 1):
        regressor.train()
        total_loss = 0
        val_loss = 0
        for data in dataloader:
            input_tensor, target_tensor = data
            input_tensor = input_tensor.float().to(device)
            target_tensor = target_tensor.float().to(device)

            optimizer.zero_grad()

            if 'patch' in strategy:
                input_tensor = input_tensor.permute(0, 2, 1)
                input_tensor = input_tensor.unfold(dimension=-1,
                                     size=patch_size,
                                     step=stride)
                input_tensor = input_tensor.permute(0,1,3,2) 

            predictions, hidden = regressor(input_tensor)
            loss = criterion(predictions.squeeze(), target_tensor)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print_loss_total += total_loss/len(dataloader)

        regressor.eval()
        for val in val_dataloader:
            x_val_data, y_val_data = val

            if 'patch' in strategy:
                x_val_data = x_val_data.permute(0, 2, 1)
                x_val_data = x_val_data.unfold(dimension=-1,
                                     size=patch_size,
                                     step=stride)
                x_val_data = x_val_data.permute(0,1,3,2) 

            predictions, _ = regressor(x_val_data)
            val_loss += criterion(predictions.squeeze(), y_val_data)\
                    .detach().cpu().tolist()

        val_loss_total += val_loss/len(val_dataloader)
        
        training_history.append(total_loss)
        val_history.append(val_loss)

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            val_loss_avg = val_loss_total / print_every
            print_loss_total = 0
            val_loss_total = 0
            print('%s (%d %d%%) %.4f | %.4f' % (
                timeSince(start, epoch / n_epochs),
                epoch, epoch / n_epochs * 100, 
                print_loss_avg, val_loss_avg))

    history = {'loss': training_history, 'val_loss': val_history}
    return history, regressor

def run_ica(data, n_components=6, fs=IMU_FS, **kwargs):
    ica = FastICA(n_components, whiten='arbitrary-variance',
                  random_state=seed,  **kwargs)
    data_ica = ica.fit_transform(data)
    win_data = window_filter(data_ica, 2*fs, window='triang')
    return win_data

def fit_subject_ica(window_list, n_components=6, **kwargs):
    ica = FastICA(n_components, whiten='arbitrary-variance',
                  random_state=seed, **kwargs)
    subject_data = np.concatenate(window_list, axis=0)
    ica.fit(subject_data)
    return ica

def apply_ica_model(window_list, ica, fs=IMU_FS):
    return [
        window_filter(ica.transform(win_data), 2*fs, window='triang')
        for win_data in window_list
    ]

def get_subject_level_ica_models(sbj_dir, subject, acc_filt_list, gyr_filt_list,
                                 fs=IMU_FS, n_components=2, output_dir='mr',
                                 **kwargs):
    output_dir = normalize_output_directory(output_dir)
    if output_dir == 'mr':
        acc_ica = fit_subject_ica(acc_filt_list, n_components=n_components,
                                  **kwargs)
        gyr_ica = fit_subject_ica(gyr_filt_list, n_components=n_components,
                                  **kwargs)
        return acc_ica, gyr_ica

    mr_fname = join(sbj_dir, subject+'.pkl')
    if not exists(mr_fname):
        raise FileNotFoundError(
            f"Missing subject-level MR ICA fit for {subject}: {mr_fname}"
        )

    with open(mr_fname, 'rb') as f:
        mr_processed = pickle.load(f)

    acc_ica = mr_processed.get('acc_ica_model')
    gyr_ica = mr_processed.get('gyr_ica_model')
    if acc_ica is None or gyr_ica is None:
        raise KeyError(
            f"Subject-level MR ICA models not found in {mr_fname}"
        )

    return acc_ica, gyr_ica

def signal_processing(sbj_processed_dir:str, sbj_dicts:list, fs=IMU_FS,
                      window_size=20, window_shift=1, n_components=2, 
                      plot=False, overwrite=False, output_dir='mr'):
    output_dir = normalize_output_directory(output_dir)
    sbj_processed_list = []
    for sbj_dict in sbj_dicts:
        subject = sbj_dict['subject']
        sbj_dir = join(sbj_processed_dir, output_dir)
        sbj_fname = join(sbj_dir, subject+'.pkl')

        makedirs(sbj_dir, exist_ok=True)

        if exists(sbj_fname) and not overwrite:
            with open(sbj_fname, 'rb') as f:
                sbj_processed = pickle.load(f)
        else:
            sbj_processed = {}
            imu_df = sbj_dict['imu']
            pss_df = sbj_dict['pss']
            br_df = sbj_dict['br']

            # split to windows
            # imu_signal_processing and pressure sig proc
            data = prepare_imu_pss(imu_df, pss_df, window_size=window_size, 
                                   window_shift=window_shift)
            if data is None:
                continue

            imu_filt, pss_filt, pss_freqs, pss_time, conds_wins = data

            # FastICA for each window
            imu_filt_list = [data[..., 1:] for data in imu_filt]
            acc_filt_list = [data[..., :3] for data in imu_filt_list]
            gyr_filt_list = [data[..., 3:] for data in imu_filt_list]

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                acc_ica_model, gyr_ica_model = get_subject_level_ica_models(
                    join(sbj_processed_dir, 'mr'),
                    subject,
                    acc_filt_list,
                    gyr_filt_list,
                    fs=fs,
                    n_components=n_components,
                    output_dir=output_dir,
                    whiten_solver='eigh',
                    max_iter=1000,
                )
                acc_ica_list = apply_ica_model(acc_filt_list, acc_ica_model,
                                               fs=fs)
                gyr_ica_list = apply_ica_model(gyr_filt_list, gyr_ica_model,
                                               fs=fs)
                imu_ica_list = [
                    np.concatenate((acc, gyr), axis=1) for acc, gyr in zip(
                        acc_ica_list, gyr_ica_list)
                ]

            # Plotting
            if plot:
                imu_nsamples = imu_filt.shape[1]
                pss_nsamples = pss_filt.shape[-1]

                idxs = np.random.randint(0, len(imu_ica_list), (6,))
                for i, idx in enumerate(idxs):
                    plt.subplot(3, 2, i+1)
                    plt.plot(np.linspace(0, 1, imu_nsamples), imu_ica_list[idx])
                    ax = plt.gca(); twin = ax.twinx()
                    twin.plot(np.linspace(0, 1, pss_nsamples), pss_filt[idx],
                              color='tab:red')
                plt.show()

            imu_time = imu_filt[..., 0]
            br_time = br_df['sec'].values

            br_idxs = [
                sync_with_last_val(time, br_time) for time in imu_time
            ]
            br = br_df['BR'].values[br_idxs]

            sbj_processed['subject'] = subject
            sbj_processed['imu_filt'] = imu_filt[..., 1:]
            sbj_processed['imu_ica'] = np.array(imu_ica_list)
            sbj_processed['acc_ica_model'] = acc_ica_model
            sbj_processed['gyr_ica_model'] = gyr_ica_model
            sbj_processed['ica_fit_conditions'] = (
                np.array(['M', 'R']) if output_dir == 'levels'
                else np.unique(conds_wins)
            )
            sbj_processed['ica_scope'] = 'subject'
            sbj_processed['pss_filt'] = pss_filt
            sbj_processed['pss_freqs'] = pss_freqs
            sbj_processed['br'] = br
            sbj_processed['conds'] = conds_wins
            sbj_processed['imu_time'] = imu_time
            sbj_processed['pss_time'] = pss_time
            sbj_processed['br_time'] = br_time

            with open(sbj_fname, 'wb') as f:
                pickle.dump(sbj_processed, f)

        sbj_processed_list.append(sbj_processed)
    return sbj_processed_list

def main(args):
    window_size  = args.window_size
    window_shift = args.window_shift
    overwrite    = args.overwrite
    strategy     = args.strategy
    freeze       = args.freeze
    n_components = args.n_components
    extend       = args.extend
    data_str     = args.data_str
    n_classes    = args.n_classes

    sbj_processed_list = args.sbj_processed_list
    model_parent_directory = args.model_parent_directory

    if torch.cuda.is_available():
        device = 'cuda:'+str(args.device)
    else:
        device = 'cpu'

    print("training on ", device)

    if strategy == 'gan': raise NotImplementedError

    channel_first = True if 'patch' in strategy else False


    if window_size > 10:
        batch_size = 8
    else:
        batch_size = 32

    if strategy == 'patchtst':
        batch_size = 56
    elif strategy == 'patchtcn':
        batch_size = 56

    shuffle = True
    drop_last = False
    n_channels = n_components
    hidden_size = 128
    output_size = MAX_LENGTH

    if args.debug:
        n_epochs = 10
    else:
        n_epochs = 1000

    # learning_rate = 3e-4 # 3e-4
    learning_rate = 3e-4
    if 'patch' in strategy:
        learning_rate = 1.0
    n_plots = 10

    out_patch_size = 36
    out_stride = out_patch_size

    patch_size = int((120/18)*36)
    stride = patch_size

    if args.cls_embed:
        strategy += '_cls'

    params = {
        'strategy'      : strategy,
        'shuffle'       : shuffle,
        'drop_last'     : drop_last,
        'overwrite'     : overwrite,
        'n_channels'    : n_channels,
        'hidden_size'   : hidden_size,
        'output_size'   : MAX_LENGTH,
        'n_epochs'      : n_epochs,
        'learning_rate' : learning_rate,
        'n_plots'       : 10,
        'patch_size'    : patch_size,
        'stride'        : stride,
        's_mdl_dir'     : '',
        'print_every'   : 10,
        'device'        : device,
        'extend'        : extend,
        'data_str'      : data_str,
        'cls_embed'     : args.cls_embed,
        'n_classes'     : n_classes,
    }

    ft_params = params.copy()
    ft_params.update({'validation_split': 0.25,
                      'learning_rate': 1e-4, 
                      'n_epochs': 50})
    if args.debug:
        ft_params['n_epochs'] = 5

    if hasattr(args, "sbj"):
        if isinstance(args.sbj, str):
            subject_list = [args.sbj]
        else:
            subject_list = args.sbj
    else:
        subject_list = subjects

    # for each subject
    for sbj in subject_list:

        # get which files to use for training and testing from the
        # preprocessing scratch directory
        s_mdl_dir = join(model_parent_directory, sbj, strategy)
        makedirs(s_mdl_dir, exist_ok=True)

        print("saving model outputs here: ", s_mdl_dir)

        params.update({'s_mdl_dir': s_mdl_dir, 'sbj': sbj})
        ft_params.update({'s_mdl_dir': s_mdl_dir})

        train_data, test_data, br_data, conds = load_loocv_dataset(
            sbj, sbj_processed_list,
            data_str=data_str,
            output_dir=args.output_directory)
        x_train, y_train = train_data
        x_test, y_test = test_data
        br_train, br_test = br_data
        c_train, c_test = conds

        pretrain_enc, pretrain_dec = do_pretraining(
            x_train, x_test, y_train, y_test, c_train, c_test, **params)
        
        if strategy != 'conv':
            if window_size == 1:
                scalar = 24
            elif window_size == 20:
                scalar = 480
            else:
                raise NotImplementedError
        
        if strategy == 'masked': scalar = int(scalar*1.25)
        
        if strategy == 'conv': scalar = 240

        if 'patch' in strategy:
            if data_str == 'imu_ica':
                scalar = int(scalar)
            else:
                scalar = 1440

        if strategy == 'patchattn':
            if data_str == 'imu_ica':
                scalar = 132
            else:
                scalar = 396
        elif strategy == 'patchtst':
            if data_str == 'imu_ica':
                scalar = int(scalar)
            else:
                scalar = 396
        elif strategy == 'patchtcn':
            if data_str == 'imu_ica':
                scalar = int(scalar)
            else:
                scalar = 396

        regressor = PretrainedRegressor(
            pretrain_enc, hidden_size, scalar=scalar, strategy=strategy
        ).to(device)

        if freeze:
            mdl_ckpt_str = join(s_mdl_dir, f'ckp_freeze.pt')
            br_lbl_fname = join(s_mdl_dir, 'freeze_labels.npy')
            br_prd_fname = join(s_mdl_dir, 'freeze_pred.npy')

            pretrained_dict = pretrain_enc.state_dict()
            # Freeze everything except last layer.
            set_requires_grad(pretrain_enc, pretrained_dict, 
                              requires_grad=False)
        else:
            mdl_ckpt_str = join(s_mdl_dir, f'ckp_finetune.pt')
            br_lbl_fname = join(s_mdl_dir, 'finetune_labels.npy')
            br_prd_fname = join(s_mdl_dir, 'finetune_pred.npy')

        if overwrite or not exists(br_prd_fname) or not exists(br_lbl_fname) \
           or args.extend:
            # Save weights
            history, regressor = do_fine_tuning(regressor, x_train, br_train, 
                                                **ft_params)
            regressor.eval()

            test_dataloader = DataLoader(LoadDataset(x_test, y_test,
                                                     aug_ratio=0.),
                                         batch_size=batch_size, shuffle=shuffle,
                                         drop_last=drop_last, num_workers=0)
            predictions = []
            for x, y in test_dataloader:
                if 'patch' in strategy:
                    x = torch.Tensor(x).float().permute(0, 2, 1)
                    x = x.unfold(dimension=-1, size=patch_size, step=stride)
                    x = x.permute(0,1,3,2) 
                else:
                    x = torch.Tensor(x)

                y_hat, _ = regressor(x.to(device))
                predictions.append(y_hat.detach().cpu().numpy())
            predictions = np.concatenate(predictions, axis=0)

            evals = Evaluation(br_test, predictions.squeeze())
            print(evals.get_evals())

            chkpoint = {
                'state_dict': regressor.state_dict(),
            }
            torch.save(chkpoint, mdl_ckpt_str)

            # Save test label and prediction outputs
            np.save(br_lbl_fname, br_test)
            np.save(br_prd_fname, predictions)
        else:
            chkpoint = torch.load(mdl_ckpt_str, map_location=device,)
                                  # weights_only=True)
            ckpt_dict = chkpoint["state_dict"]
            model_dict = regressor.state_dict()
            model_dict.update(ckpt_dict)
            regressor.load_state_dict(model_dict)
            regressor.eval()

            br_lbls = np.load(br_lbl_fname).flatten()
            br_prds = np.load(br_prd_fname).flatten()

            br_prds_postproc = regression_post_process(br_prds)

            evals = Evaluation(br_lbls, br_prds)
            evals_postproc = Evaluation(br_lbls, br_prds_postproc)

            m_mask = c_test=='M'
            r_mask = c_test=='R'

            m_evals = Evaluation(br_lbls[m_mask], br_prds_postproc[m_mask])
            r_evals = Evaluation(br_lbls[r_mask], br_prds_postproc[r_mask])

            print("{}:\n{}\n{}".format(sbj,evals.get_evals(),
                                       evals_postproc.get_evals()))

            print("{}:\nR:\n{}\nM:\n{}".format(sbj,m_evals.get_evals(),
                                               r_evals.get_evals()))

            # plt.figure()
            # plt.plot(br_lbls); plt.plot(br_prds); plt.plot(br_prds_postproc)
            # plt.show()

        plt.close()
        gc.collect()
        torch.cuda.empty_cache()


def train_subjects_on_gpu(rank, world_size, subjects, args):
    # set the right device
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)

    # slice subjects for this GPU
    local_subjects = subjects[rank::world_size]

    for sbj in local_subjects:
        print(f"[GPU {rank}] Training subject {sbj}")
        # pass device info into params
        args.device = rank
        args.sbj = sbj
        main(args)   # will internally loop over only this sbj

def get_results(args):
    data_str = args.data_str
    condition = args.condition
    strategy = args.strategy
    freeze = args.freeze
    n_components = args.n_components

    window_size = args.window_size
    window_shift = args.window_shift

    sbj_results = {}
    sbj_dicts = load_dataset(subjects, condition=condition, 
                             data_dir=SEAT_DATA_DIR)
    sbj_processed_list = signal_processing(sbj_processed_dir, sbj_dicts,
                                           window_size=window_size,
                                           window_shift=window_shift,
                                           n_components=n_components,
                                           output_dir=args.output_directory)

    # Get labels and predictions from args for each subject
    for s_iter, sbj in enumerate(tqdm.tqdm(subjects)):
        s_mdl_dir = join(model_parent_directory, sbj, strategy)

        if freeze:
            br_lbl_fname = join(s_mdl_dir, 'freeze_labels.npy')
            br_prd_fname = join(s_mdl_dir, 'freeze_pred.npy')
        else:
            br_lbl_fname = join(s_mdl_dir, 'finetune_labels.npy')
            br_prd_fname = join(s_mdl_dir, 'finetune_pred.npy')

        lbls = np.load(br_lbl_fname).flatten()
        prds = np.load(br_prd_fname).flatten()

        prds_post= regression_post_process(prds)

        evals = Evaluation(lbls, prds)
        evals_postproc = Evaluation(lbls, prds_post)

        _, _, br_data, conds = load_loocv_dataset(
            sbj, sbj_processed_list,
            data_str=data_str,
            output_dir=args.output_directory)
        br_train, br_test = br_data
        train_conds, test_conds = conds

        # Evaluate for M and R
        evals = Evaluation(lbls, prds_post)
        sbj_results[sbj] = evals.get_evals()
        sbj_results[sbj].update({'mask': test_conds})

    sbj_df = pd.DataFrame(sbj_results)
    return sbj_df

if __name__ == '__main__':
    gc.enable()
    args = arg_parser()

    if args.debug:
        subjects = subjects[:3]

    MAX_LENGTH = int(PSS_FS*args.window_size)

    # args in seconds
    # if args.window_size == WINDOW_SIZE and args.window_shift == WINDOW_SHIFT:
    #     model_parent_directory = sep.join(
    #         model_parent_directory.split(sep)[:-1]+['loocv_matched']
    #     )

    model_parent_directory = ''
    m_dir = join(m_dir, args.data_str)

    if m_dir == SEAT_DATA_DIR:
        model_parent_directory = sep.join(m_dir.split(sep)[:-1]+['loocv'])
    else:
        model_parent_directory = join(m_dir, 'loocv')


    args.model_parent_directory = model_parent_directory

    output_dir = normalize_output_directory(args.output_directory)

    print("Saving data to ", sbj_processed_dir)
    sbj_dicts = load_dataset(subjects, condition=args.condition, 
                             data_dir=SEAT_DATA_DIR, debug=args.debug)

    sbj_processed_list = signal_processing(sbj_processed_dir, sbj_dicts,
                                           fs=IMU_FS,
                                           window_size=args.window_size,
                                           window_shift=args.window_shift,
                                           n_components=args.n_components,
                                           overwrite=args.overwrite,
                                           output_dir=output_dir)
    c_list = []
    for sbj_df in sbj_processed_list:
        c_list.append(sbj_df['conds'])
    
    le = LabelEncoder()
    le.fit(np.concatenate(c_list, axis=0))
    le_fname = join(sbj_processed_dir, output_dir, 'label_encoder.pkl')
    makedirs(join(sbj_processed_dir, output_dir), exist_ok=True)
    with open(le_fname, 'wb') as f:
        pickle.dump(le, f)

    args.sbj_processed_list = sbj_processed_list
    args.n_classes = len(le.classes_)
