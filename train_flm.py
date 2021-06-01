
import youtokentome as yttm
import torch as t
from torch.nn import MSELoss
import numpy as np
from torch.utils.data import DataLoader

from data import TextDataSet, apply_bpe
from flm import FreqEmbedding, FLayer

vocab_size = 5000
bpe_model_path = "bpe.txt"
train_data_path = "train_data.txt"


apply_bpe(train_data_path=train_data_path,
          bpe_model_path=bpe_model_path,
          encoded_path="encoded.txt",
          vocab_size=vocab_size)

ds = TextDataSet(path_data="encoded.txt",
                 window_length=50)
dl = DataLoader(dataset=ds, batch_size=4)


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


counter = 0

while True:
    for minibatch in dl:
        minibatch = minibatch.permute(1,0,2)
        indices_input, indices_target = minibatch
        embeddings_input = complex_embedding(indices_input)
        embeddings_target = complex_embedding(indices_target)
        output = layers(embeddings_input)
        loss = (mse(input=output.real, target=embeddings_target.real)
                + mse(input=output.imag, target=embeddings_target.imag)) / 2
        loss.backward()
        opt.step()
        opt.zero_grad()
        print(f"{counter} {loss.detach().item()}")

    counter += 1






