"""DP-TVAE Synthesizer module.

This module provides a differentially private version of the TVAE synthesizer
using a manual, hook-based implementation of DP-SGD for efficient per-sample
gradient computation.
"""

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import cross_entropy
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import random_state
from ctgan.synthesizers.tvae import TVAE, Decoder, Encoder, _loss_function
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import opacus

from dpsdg.data.dp_data_transformer import DPDataTransformer

class DPTVAE(TVAE):
    """A TVAE synthesizer with a manual, efficient DP-SGD implementation."""

    def __init__(
        self,
        epsilon=1.0,
        delta=1e-5,
        max_grad_norm=1.0,
        use_gradient_penalty=True,
        use_opacus_noise_mul=False,
        **kwargs,
    ):
        """Create a DP-TVAE synthesizer."""
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.use_opacus_noise_mul = use_opacus_noise_mul

    @random_state
    def fit_transformer(self, full_data, discrete_columns):
        self.transformer = DPDataTransformer()
        self.transformer.fit(full_data, discrete_columns)

    @random_state
    def sample(self, num_rows):
        return super().sample(num_rows)

    @random_state
    def fit(self, train_data, discrete_columns=()):
        """Fit the TVAE Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
        """
        random_seeds = torch.empty(2, dtype=int).random_()
        loader_gen = torch.Generator()
        loader_gen.manual_seed(random_seeds[0].item())
        noise_gen = torch.Generator(device=self._device)
        noise_gen.manual_seed(random_seeds[1].item())

        self.transformer = DataTransformer()
        self.transformer.fit(train_data, discrete_columns)
        train_data = self.transformer.transform(train_data)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(train_data.astype('float32'))),
            batch_size=self.batch_size, shuffle=True, drop_last=False,
            generator=loader_gen
        )
        loader = opacus.data_loader.DPDataLoader.from_data_loader(loader)

        data_dim = self.transformer.output_dimensions
        encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self._device)
        self.decoder = Decoder(self.embedding_dim, self.decompress_dims, data_dim).to(self._device)
        all_parameters = list(encoder.parameters()) + list(self.decoder.parameters())
        optimizerAE = Adam(all_parameters, weight_decay=self.l2scale)

        noise_multiplier = 0.0
        if self.epsilon > 0:
            noise_multiplier = opacus.accountants.utils.get_noise_multiplier(
                target_epsilon=self.epsilon,
                target_delta=self.delta,
                sample_rate=1/len(loader),
                epochs=self.epochs,
            )
            print(f"{noise_multiplier=}")

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Batch', 'Loss'])
        iterator = tqdm(range(self.epochs), disable=(not self.verbose))
        if self.verbose:
            iterator_description = 'Loss: {loss:.3f}'
            iterator.set_description(iterator_description.format(loss=0))

        for i in iterator:
            loss_values = []
            batch = []

            for id_, data in enumerate(loader):
                batch_size = data[0].shape[0]
                real = data[0].to(self._device)
                mu, std, logvar = encoder(real)
                eps = torch.randn_like(std)
                emb = eps * std + mu
                rec, sigmas = self.decoder(emb)

                accum_grads = [torch.zeros_like(p) for p in all_parameters]
                total_loss = 0

                #_loss_function
                #loss_sigmas = torch.log(sigmas).sum()
                loss_cols = []
                loss_sigmas = 0
                st = 0
                for column_info in self.transformer.output_info_list:
                    for span_info in column_info:
                        ed = st + span_info.dim
                        if span_info.activation_fn != 'softmax':
                            std = sigmas[st]
                            eq = real[:, st] - torch.tanh(rec[:, st])
                            loss_cols.append((eq**2 / 2 / (std**2)))
                            loss_sigmas += torch.log(std)
                        else:
                            loss_cols.append(
                                cross_entropy(
                                    rec[:, st:ed],
                                    real[:, st:ed].argmax(dim=-1),
                                    reduction='none'
                                )
                            )
                        st = ed
                assert st == rec.size()[1]

                loss_1 = (loss_sigmas + torch.stack(loss_cols).sum(dim=0)) * self.loss_factor
                loss_2 = torch.sum((1 + logvar - mu**2 - logvar.exp()) / -2, dim=1)

                for j in range(batch_size):
                    optimizerAE.zero_grad()
                    loss = loss_1[j] + loss_2[j]
                    loss.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(all_parameters, self.max_grad_norm)
                    for k, p in enumerate(all_parameters):
                        accum_grads[k] += p.grad
                    total_loss += loss.item()

                for j, p in enumerate(all_parameters):
                    noise = torch.normal(
                        0,
                        noise_multiplier * self.max_grad_norm,
                        size=p.grad.shape,
                        device=self._device,
                    )
                    p.grad = (accum_grads[j] + noise) / batch_size

                optimizerAE.step()
                self.decoder.sigma.data.clamp_(0.01, 1.0)

                batch.append(id_)
                loss_values.append(total_loss / batch_size)

            epoch_loss_df = pd.DataFrame({
                'Epoch': [i] * len(batch),
                'Batch': batch,
                'Loss': loss_values,
            })
            if not self.loss_values.empty:
                self.loss_values = pd.concat([self.loss_values, epoch_loss_df]).reset_index(
                    drop=True
                )
            else:
                self.loss_values = epoch_loss_df

            if self.verbose:
                iterator.set_description(
                    iterator_description.format(loss=loss_values[-1])
                )
