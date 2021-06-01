import torch as t
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

class TextDataSet(t.utils.data.Dataset):
    def __init__(self, path_data: str, window_length: int):
        self.path_data = path_data
        self.window_length = window_length
        self.tokens = []
        with open(path_data, "r") as f:
            for line in f.readlines():
                for token in line.split():
                    self.tokens.append(int(token))

        self.tokens = t.from_numpy(np.array(self.tokens))

    def __len__(self):
        return (len(self.tokens) - self.window_length) + 1

    def __getitem__(self, idx):
        return t.stack((self.tokens[idx:idx + self.window_length], self.tokens[idx+1:(idx + self.window_length)+1]))


import youtokentome as yttm


def apply_bpe(train_data_path: str,
              bpe_model_path: str,
              encoded_path: str,
              vocab_size: int) -> yttm.BPE:
    # Training model
    yttm.BPE.train(data=train_data_path, vocab_size=vocab_size, model=bpe_model_path)

    # Loading model
    bpe = yttm.BPE(model=bpe_model_path)
    print("counting lines")
    n_lines = 0
    with open(train_data_path, "r") as f:
        for line in f:
            n_lines += 1
    with open(train_data_path, "r") as f:
        with open(encoded_path, "a") as f_write:
            for line in tqdm(f, total=n_lines):
                encoded = bpe.encode(sentences=[line],
                                     bos=True,
                                     eos=True)[0]
                f_write.write(" ".join(map(str, encoded)) + "\n")
    return bpe


if __name__ == "__main__":
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
    print(ds[0].shape)
    print(ds[1].shape)
    for minibatch in dl:
        print(f"{minibatch.shape=}")
        print(f"{len(minibatch)=}")
        for entry in minibatch:
            print(f"{len(entry)=}")
        break
