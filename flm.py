from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch as t
from torch.nn import MSELoss


class ComplexEmbedding(t.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.real = t.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.imag = t.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

    def forward(self, input):
        return self.real(input) + 1j * self.imag(input)


class FreqEmbedding(t.nn.Module):
    def __init__(self, vocab_size: int, n_channels: int, n_freqs: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_channels = n_channels
        self.n_freqs = n_freqs
        self.freq_embeddings = t.nn.ModuleList([ComplexEmbedding(num_embeddings=vocab_size,
                                                                 embedding_dim=n_channels)
                                                for f in range(n_freqs)])

    def forward(self, indices: t.Tensor) -> t.Tensor:
        embeddings = t.zeros(list(indices.shape) + [self.n_freqs, self.n_channels]) + 0 * 1j
        for i_freq in range(self.n_freqs):
            embeddings[..., i_freq, :] = self.freq_embeddings[i_freq](indices)
        return embeddings

class FLayer(t.nn.Module):
    def __init__(self, n_freqs:int, n_channels:int):
        super().__init__()
        self.n_freqs= n_freqs
        self.n_channels = n_channels
        self.weights = t.nn.Parameter(t.randn((n_freqs,n_channels))+0j,requires_grad=True)
    def forward(self,embeddings:t.Tensor)->t.Tensor:
        return embeddings * self.weights + embeddings

def get_magnitude(x: t.Tensor) -> t.Tensor:
    return t.abs(x)


def get_phase(x: t.Tensor) -> t.Tensor:
    return t.angle(x)


def get_signal(freq_bins: t.Tensor,
               embeddings: t.Tensor,
               signal_length: int = 10000) -> Tuple[t.Tensor, t.Tensor]:
    minibatch_size, seq_length, n_freqs, n_channels = embeddings.shape
    magnitudes = get_magnitude(embeddings)
    phases = get_phase(embeddings)
    xs = t.linspace(0, int(freq_bins[-1] * 2), signal_length)
    signals = t.zeros((minibatch_size, n_channels, len(xs)))
    for i_entry in range(minibatch_size):
        for i_token in range(seq_length):
            for i_freq in range(n_freqs):
                for i_channel in range(n_channels):
                    signals[i_entry, i_channel] += magnitudes[i_entry, i_token, i_freq, i_channel] \
                                                   * t.sin(2 * np.pi * freq_bins[i_freq] * xs
                                                           + phases[i_entry, i_token, i_freq, i_channel])

    return xs, signals


def plot_signals(signals: t.Tensor, xs: t.Tensor):
    assert len(signals.shape) == 2, "pass single entry, not a minibatch!"
    for i_channel, channel_signal in enumerate(signals):
        plt.plot(xs.detach().numpy(), channel_signal.detach().numpy(), #label=f"{i_channel}"
                 )


if __name__ == "__main__":

    vocab_size = 100
    n_channels = 4
    n_freqs = 32
    freq_band = (0.1, 1)
    indices = t.arange(0, 10).view(1, -1).int()

    freq_bins = t.linspace(*freq_band, n_freqs)
    complex_embedding = FreqEmbedding(vocab_size=vocab_size,
                                      n_channels=n_channels,
                                      n_freqs=n_freqs)

    n_layers = 5
    layers = t.nn.Sequential(
        *[FLayer(n_freqs=n_freqs,n_channels=n_channels) for n in range(n_layers)]
    )
    opt = t.optim.Adam(params=list(layers.parameters()))
    opt.zero_grad()
    mse = MSELoss()

    counter= 0
    plt.ion()
    while True:
        embeddings = complex_embedding(indices)
        output = layers(embeddings)
        loss = (mse(input=embeddings.real,target=output.real) + mse(input=embeddings.imag,target=output.imag) ) / 2
        loss.backward()
        opt.step()
        opt.zero_grad()
        print(f"{counter} {loss.detach().item()}")
        counter += 1
        if counter % 100==0:
            xs, signals = get_signal(freq_bins=freq_bins,embeddings=embeddings)
            _, signals_out = get_signal(freq_bins=freq_bins,embeddings=output)
            plt.clf()
            plt.subplot(1,2,1)
            plt.title("original signal")
            plot_signals(signals[0],xs=xs)
            plt.subplot(1,2,2)
            plt.title("output signal")
            plot_signals(signals_out[0],xs=xs)
            plt.pause(0.1)

