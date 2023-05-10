'''Synthetic datasets to test PDE in-context learning ability.'''

import os
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader
from typing import Dict
import numpy as np
from tqdm import tqdm
from collections import Counter
import h5py

from src.dataloaders.base import SequenceDataset
from src.dataloaders.FHN import get_data as make_FHN


# def generate_start_seq(vocab: Vocab, input_seq_len: int, rng: np.random.Generator):
#     """Generate token sequence up to and including the copy_prefix token."""
#     vocab_seq = rng.choice(
#         vocab.vocab,
#         input_seq_len,
#         replace=True,
#         # Do not generate any special tokens
#         p=[1/(len(vocab)-len(vocab.special_tokens)) if p not in vocab.special_tokens else 0 for p in vocab.vocab])
#     vocab_seq = np.append(vocab_seq, vocab.copy_prefix)
#     return vocab_seq.tolist()

# def generate_induction_head(
#     vocab: Vocab,
#     input_seq_len: int,
#     copy_prefix: str,
#     induction_len: int,
#     num_triggers: int,
#     rng: np.random.Generator,
#     valid_chars: list = None,
# ):
#     """Generate sequence where the copy prefix is inserted into the input
#     and then the character after the copy prefix is copied at the end.
#     """
#     if valid_chars is not None:
#         raise NotImplementedError("Valid chars not implemented for induction heads.")
#     vocab_seq = generate_start_seq(vocab, input_seq_len, rng)
#     if rng.uniform() < 0.5:
#         num_triggers = 1
#     pos = sorted(rng.integers(
#         input_seq_len - (1 + induction_len), size=num_triggers
#     ))
#     pos_filtered = []
#     for i, p in enumerate(pos):
#         if i == 0:
#             pos_filtered.append(p)
#         elif p - pos_filtered[-1] > induction_len:
#             pos_filtered.append(p)
#     to_copy = [
#         vocab_seq[pos_filtered[0]+1+i]
#         for i in range(induction_len)
#     ]
#     for pos in pos_filtered:
#         vocab_seq[pos] = copy_prefix
#         for i in range(induction_len):
#             vocab_seq[pos+1+i] = to_copy[i]
#     # if valid_chars is not None and to_copy not in valid_chars:
#     #     vocab_seq[pos+1] = rng.choice(valid_chars)
#     #     to_copy = vocab_seq[pos+1]
#     vocab_seq = vocab_seq + to_copy
#     return " ".join(vocab_seq)

# def generate_assoc_recall(
#     vocab: Vocab,
#     input_seq_len: int,
#     num_keys: int,
#     rng: np.random.Generator,
#     allow_dot: bool = True,
#     valid_chars: list = None,
# ):
#     """Generate sequence where the input has a sequence of key value pairs
#     and the copy prefix at the end, and then a key value pair is inserted
#     after the copy prefix."""
#     non_special_vocab_size = len(vocab.non_special_vocab)
#     keys = vocab.non_special_vocab[:non_special_vocab_size // 2]
#     values = vocab.non_special_vocab[non_special_vocab_size // 2:]
#     keys_multi = [ [key] for key in keys ]
#     for i in range(num_keys-1):
#         keys_multi = [ key + [key2] for key in keys_multi for key2 in keys ]
#     kv_map = {
#         tuple(k): rng.choice(values) for k in keys_multi
#     }

#     key_present = {}
#     vocab_seq = []
#     for _ in range(input_seq_len // (num_keys + 1)):
#         k = tuple(rng.choice(list(kv_map.keys())))
#         v = kv_map[k]
#         vocab_seq += list(k) + [v]
#         key_present[k] = True
#         # vocab_seq.append(v)

    
#     k = tuple(rng.choice(list(kv_map.keys())))
#     if not allow_dot:
#         while k not in key_present:
#             k = tuple(rng.choice(list(key_present.keys())))
#     to_copy = [vocab.copy_prefix] + list(k) + [ kv_map[k] if k in key_present else vocab.noop ]
#     vocab_seq = vocab_seq + to_copy
#     return " ".join(vocab_seq)

# def generate_fhn(
#     input_seq_len: int
# ): # (1, 1000, 1), (1, 1000, 1)
#     return make_FHN(1)

class PDEDataModule(SequenceDataset):
    _name_ = "icl_pde"

    def __init__(
        self,
        num_examples: int,
        num_test_examples: int,
        num_initial_conditions: int,
        pde: str,
        seed: int = 0,
        batch_size: int = 32,
        # split_train_test: bool = False,
        allow_dot: bool = False,
        # vocab_size: int,
        # input_seq_len: int,
        # copy_method: str,
        # number_duplicates_per_epoch: int = 0,
        # max_copy_len: int = 10,
        # test_seq_len: int = None,
        # num_keys: int = 1, # number of keys for associative recall,
        data_dir: str = None,
        file_name: str = None,
        *args, **kwargs
    ):
        self.num_examples = num_examples
        self.num_test_examples = num_test_examples
        self.pde = pde
        self.num_initial_conditions = num_initial_conditions
        # self.input_seq_len = input_seq_len
        # self.vocab_size = vocab_size
        # self.copy_method = copy_method
        # assert copy_method in ["induction_head", "assoc_recall", "FHN"]
        # self.number_duplicates_per_epoch = number_duplicates_per_epoch
        self.seed = seed
        self.batch_size = batch_size
        # self.split_train_test = split_train_test # let the same copy chars appear in train/test
        # self.induction_len = induction_len
        # self.induction_num_triggers = induction_num_triggers
        self.allow_dot = allow_dot
        # self.max_copy_len = max_copy_len
        self.data_dir = data_dir
        self.file_name = file_name

        self.rng = np.random.default_rng()
        
        # if test_seq_len is not None:
        #     self.test_seq_len = test_seq_len
        # else:
        #     self.test_seq_len = input_seq_len
        # self.num_keys = num_keys

        # self.num_extra_seq_len = 2

        # if self.number_duplicates_per_epoch > 0:
        #     self.duplicate_ex = self.generate_example()
        #     self.duplicate_index = max(int(self.num_examples / self.number_duplicates_per_epoch), 1)
        # else:
        #     self.duplicate_ex = None
        #     self.duplicate_index = -1

        # self.total_seq_len = self.input_seq_len + self.num_extra_seq_len

    # def generate_induction_head(self, seqlen=None, valid_chars=None):
    #     return generate_induction_head(self.vocab, seqlen if seqlen is not None else self.input_seq_len, self.special_vocabs["copy_prefix"], self.induction_len, self.induction_num_triggers, self.rng, valid_chars=valid_chars)

    # def generate_assoc_recall(self, seqlen=None, valid_chars=None):
    #     return generate_assoc_recall(self.vocab, seqlen if seqlen is not None else self.input_seq_len, self.num_keys, self.rng, allow_dot = self.allow_dot, valid_chars=valid_chars)

    # def generate_fhn(self, seqlen=None, valid_chars=None):
    #     return generate_fhn(seqlen if seqlen is not None else self.input_seq_len)

    # def generate_example(self, seqlen=None, valid_chars=None):
    #     vocab_seq = self.copy_f(seqlen=seqlen, valid_chars=valid_chars)
    #     if self.copy_method not in ["FHN"]:
    #         return self.tokenizer.tokenize(vocab_seq, return_tensor=True)
    #     else:
    #         return vocab_seq

    def get_icl(self, num_seqs, num_examples, data_x, data_y):
        icl_seqs = []
        for _ in range(num_seqs):
            indices = self.rng.choice(data_x.shape[0], num_examples)
            examples = [torch.concat([data_x[i], data_y[i]]) for i in indices]
            icl_seq = torch.concat(examples) # (L * 2 * num_examples)
            # print(icl_seq.shape)
            icl_seqs.append(icl_seq.unsqueeze(1))
        icl_seqs = torch.stack(icl_seqs, dim=0) # num_seqs x (L * 2 * num_examples) x 1
        # print(icl_seqs.shape)
        return icl_seqs

    def get_icl2(self, num_seqs, num_examples, data_x, data_y):
        icl_seqs = []
        for _ in range(num_seqs):
            indices = self.rng.choice(data_x.shape[0], num_examples)
            examples = []
            for i in range(len(indices)):
                examples.append(data_x[indices[i]])
                examples.append(data_y[indices[i]])
            icl_seq = torch.stack(examples, dim=0) # (2 * num_examples) x L
            icl_seqs.append(icl_seq)
        icl_seqs = torch.stack(icl_seqs, dim=0) # num_seqs x (2 * num_examples) x L
        return icl_seqs

    # Choose 
    def get_icl_t(self, num_seqs, num_examples, data, start=0.75, end=1):
        data_x = data[:, 0, :] # dataset_size x L
        icl_seqs = []
        for _ in range(num_seqs):
            t = np.random.randint(int(data.shape[1]*start), int(data.shape[1]*end))
            data_y_temp = data[:, t, :]
            indices = self.rng.choice(data_x.shape[0], num_examples)
            examples = [torch.concat([data_x[i], data_y_temp[i]]) for i in indices]
            icl_seq = torch.concat(examples) # (L * 2 * num_examples)
            # print(icl_seq.shape)
            icl_seqs.append(icl_seq.unsqueeze(1))
        icl_seqs = torch.stack(icl_seqs, dim=0) # num_seqs x (L * 2 * num_examples) x 1
        # print(icl_seqs.shape)
        return icl_seqs

    def get_icl_t2(self, num_seqs, num_examples, data, start=0.75, end=1):
        data_x = data[:, 0, :] # dataset_size x L
        icl_seqs = []
        pde_t = []
        for _ in range(num_seqs):
            t = np.random.randint(int(data.shape[1]*start), int(data.shape[1]*end))
            data_y_temp = data[:, t, :]
            indices = self.rng.choice(data_x.shape[0], num_examples)
            examples = []
            for i in range(len(indices)):
                examples.append(data_x[indices[i]])
                examples.append(data_y_temp[indices[i]])
            icl_seq = torch.stack(examples, dim=0) # (2*num_examples) x L
            icl_seqs.append(icl_seq)
            pde_t.append(t)
        icl_seqs = torch.stack(icl_seqs, dim=0) # num_seqs x (2*num_examples) x L
        pde_t = torch.tensor(pde_t) # num_seqs
        return icl_seqs, pde_t

    def get_icl_trange(self, num_seqs, num_examples, data, start=0.75, end=1, num_steps=8):
        icl_seqs = []
        pde_t = []
        for _ in range(num_seqs//num_steps):
            indices = self.rng.choice(data.shape[0], num_examples)
            ts = np.linspace(start, end, num_steps)
            for i in range(num_steps):
                t = int(ts[i]*data.shape[1])-1
                examples = []
                for j in range(len(indices)):
                    examples.append(data[indices[j], 0, :])
                    examples.append(data[indices[j], t, :])
                icl_seq = torch.stack(examples, dim=0) # (2*num_examples) x L
                icl_seqs.append(icl_seq)
                pde_t.append(t)
        icl_seqs = torch.stack(icl_seqs, dim=0) # num_seqs x (2*num_examples) x L
        pde_t = torch.tensor(pde_t) # num_seqs//num_steps * num_steps
        return icl_seqs, pde_t

    def get_icl_viscosity(self, num_seqs, num_examples):
        icl_seqs = []
        pde_viscosity = []
        files = os.listdir(self.data_dir)
        files = [f for f in files if os.path.isfile(os.path.join(self.data_dir, f))]

        for _ in range(num_seqs):
            random_file = np.random.choice(files)
            viscosity = float(random_file.split("Nu")[1].split(".npy")[0])
            df = h5py.File(os.path.join(self.data_dir, random_file), "r")
            df_tensor = df["tensor"]
            #indices = self
        # TODO finish

    def setup(self, stage=None):
        train_tensor = test_tensor = None
        # print(f"data dir: {self.data_dir}")
        # print(f"PDE: {self.pde}")
        if self.pde in ["1d_burgers", "1d_burgers_seq", "1d_burgers_icl_t", "1d_burgers_seq2", "1d_burgers_icl_t2", "1d_burgers_icl_trange"]:
            df = h5py.File(os.path.join(self.data_dir, "1D/Burgers/Train/1D_Burgers_Sols_Nu0.1.hdf5"), "r")
            df_tensor = df["tensor"]
            # try:
            train_tensor = torch.tensor(df_tensor[:self.num_examples], dtype=torch.float32)
            print(f"Train tensor shape: {train_tensor.shape}")
            test_tensor = torch.tensor(df_tensor[self.num_examples:self.num_examples+self.num_test_examples], dtype=torch.float32)
            train_tensor_x = train_tensor[:, 0, :] # num_examples x L
            train_tensor_y = train_tensor[:, -1, :] # num_examples x L
            test_tensor_x = test_tensor[:, 0, :] # num_examples x L
            test_tensor_y = test_tensor[:, -1, :] # num_examples x L
            if self.pde in ["1d_burgers_seq"]:
                icl_train_examples = self.get_icl(self.num_examples, self.num_initial_conditions, train_tensor_x, train_tensor_y) # num_examples x L x 1
                print(f"ICL train examples shape: {icl_train_examples.shape}")
                icl_test_examples = self.get_icl(self.num_test_examples, self.num_initial_conditions, test_tensor_x, test_tensor_y) # num_test_examples x L x 1
            if self.pde in ["1d_burgers_icl_t"]:
                icl_train_examples = self.get_icl_t(self.num_examples, self.num_initial_conditions, train_tensor, start=0.5, end=0.75) # num_examples x L x 1
                print(f"ICL train examples shape: {icl_train_examples.shape}")
                icl_test_examples = self.get_icl_t(self.num_test_examples, self.num_initial_conditions, test_tensor, start=0.75, end=1) # num_test_examples x L x 1
            if self.pde in ["1d_burgers_seq2"]:
                icl_train_examples = self.get_icl2(self.num_examples, self.num_initial_conditions, train_tensor_x, train_tensor_y) # num_examples x (2*num_initial_conditions) x L
                print(f"ICL train examples shape: {icl_train_examples.shape}")
                icl_test_examples = self.get_icl2(self.num_test_examples, self.num_initial_conditions, test_tensor_x, test_tensor_y)
            if self.pde in ["1d_burgers_icl_t2"]:
                icl_train_examples, pde_t_train = self.get_icl_t2(self.num_examples, self.num_initial_conditions, train_tensor, start=0.25, end=1) # num_examples x L x 1
                print(f"ICL train examples shape: {icl_train_examples.shape}")
                icl_test_examples, pde_t_test = self.get_icl_t2(self.num_test_examples, self.num_initial_conditions, test_tensor, start=0.25, end=1) # num_test_examples x L x 1
            if self.pde in ["1d_burgers_icl_trange"]:
                icl_train_examples, pde_t_train = self.get_icl_trange(self.num_examples, self.num_initial_conditions, train_tensor, start=0.25, end=1) # num_examples x L x 1
                print(f"ICL train examples shape: {icl_train_examples.shape}")
                icl_test_examples, pde_t_test = self.get_icl_trange(self.num_test_examples, self.num_initial_conditions, test_tensor, start=0.25, end=1)
            # except:
            #     pass

        # if self.data_dir is not None:
        #     try: 
        #         train_tensor = torch.load(os.path.join(self.data_dir, 
        #             f"train_{self.copy_method}_{self.num_examples}_{self.vocab_size}_{self.input_seq_len}.pt"))
        #         test_tensor = torch.load(os.path.join(self.data_dir, 
        #             f"test_{self.copy_method}_{self.num_examples}_{self.vocab_size}_{self.input_seq_len}.pt"))
        #     except:
        #         pass
                
        # if train_tensor is None or test_tensor is None:     
        #     if hasattr(self, 'dataset'):
        #         return
        #     self.rng = np.random.default_rng(self.seed)

        #     if self.copy_method not in ["FHN"]:
        #         # Make vocab
        #         if self.split_train_test:
        #             all_vocab = self.vocab.non_special_vocab
        #             train_vocab = set(self.rng.choice(all_vocab, size=len(all_vocab) // 2, replace=False))
        #             test_vocab = set(all_vocab) - train_vocab
        #             train_vocab = list(train_vocab)
        #             test_vocab = list(test_vocab)
        #         else:
        #             train_vocab = None
        #             test_vocab = None

        #         all_examples = []
        #         for i, (example_count, valid_vocab) in enumerate(zip([self.num_examples, self.num_test_examples], [train_vocab, test_vocab])):
        #             examples = torch.stack([self.generate_example(
        #                 seqlen=self.input_seq_len if i == 0 else self.test_seq_len,
        #                 valid_chars=valid_vocab
        #             )['input_ids'] for _ in tqdm(range(example_count))])
        #             examples = torch.unique(examples, dim=0, sorted=False).tolist()
                    
        #             while len(examples) < example_count:
        #                 new_example = self.generate_example(
        #                     seqlen=self.input_seq_len if i == 0 else self.test_seq_len,
        #                     valid_chars=valid_vocab
        #                 )['input_ids'].tolist()
        #                 if new_example not in examples:
        #                     examples.append(new_example)

        #             self.rng.shuffle(examples)
        #             all_examples.append(torch.LongTensor(examples))

        #         # all_examples = torch.concat(all_examples)
        #         train_tensor = torch.stack([torch.stack([example[:-1], example[1:]]) for example in all_examples[0]])
        #         test_tensor = torch.stack([torch.stack([example[:-1], example[1:]]) for example in all_examples[1]])
        #         if self.copy_method not in ["FHN"]:
        #             test_tensor[:, 1, :-1 * (self.num_extra_seq_len - 1)] = -100
        #         if self.copy_method in ["assoc_recall"]:
        #             test_tensor[:, 1, :-1] = -100
        #         if self.copy_method in ["majority", "fom1"]:
        #             train_tensor[:, 1, :-1 * (self.num_extra_seq_len - 1)] = -100
                
        #         if self.data_dir is not None:
        #             torch.save(train_tensor, os.path.join(self.data_dir, 
        #                 f"train_{self.copy_method}_{self.num_examples}_{self.vocab_size}_{self.input_seq_len}.pt")
        #             )
        #             torch.save(test_tensor, os.path.join(self.data_dir, 
        #                 f"test_{self.copy_method}_{self.num_examples}_{self.vocab_size}_{self.input_seq_len}.pt")
        #             )  

        # Basic FNO setup
        if self.pde in ["1d_burgers"]:
            self.dataset = {
                "train": TensorDataset(train_tensor_x.unsqueeze(2), train_tensor_y.unsqueeze(2)), # B x problem_dim x 1
                "test": TensorDataset(test_tensor_x.unsqueeze(2), test_tensor_y.unsqueeze(2))
            }
        # ICL
        if self.pde in ["1d_burgers_seq", "1d_burgers_icl_t"]:
            self.dataset = {
                "train": TensorDataset(
                    icl_train_examples[:, :(2*self.num_initial_conditions-1)*icl_train_examples.shape[1]//(2*self.num_initial_conditions), :],
                    icl_train_examples[:, icl_train_examples.shape[1]//(2*self.num_initial_conditions):, :]
                    # icl_train_examples[:, -1, :]
                    # icl_train_examples[:, :-1024, :],
                    # icl_train_examples[:, 1024:, :]
                ),
                "test": TensorDataset(
                    icl_test_examples[:, :(2*self.num_initial_conditions-1)*icl_train_examples.shape[1]//(2*self.num_initial_conditions), :],
                    icl_test_examples[:, icl_train_examples.shape[1]//(2*self.num_initial_conditions):, :]
                    # icl_test_examples[:, -1, :]
                )
            }
        if self.pde in ["1d_burgers_seq2", "1d_burgers_icl_t2", "1d_burgers_icl_trange"]:
            self.dataset = {
                "train": TensorDataset(
                    icl_train_examples[:, :-1, :],
                    # icl_train_examples[:, 1:, :]
                    icl_train_examples[:, 1:, :],
                    pde_t_train,
                ),
                "test": TensorDataset(
                    icl_test_examples[:, :-1, :],
                    # icl_test_examples[:, 1:, :]
                    icl_test_examples[:, 1:, :],
                    pde_t_test,
                )
            }

        # if self.copy_method not in ["FHN"]:
        #     self.dataset = {
        #         'train': TensorDataset(train_tensor[:, 0, :], train_tensor[:, 1, :]),
        #         'test': TensorDataset(test_tensor[:, 0, :], test_tensor[:, 1, :])
        #     }
        # else:
        #     train_FHN_x, train_FHN_y = make_FHN(self.num_examples, self.input_seq_len)
        #     test_FHN_x, test_FHN_y = make_FHN(self.num_test_examples, self.input_seq_len)
        #     self.dataset = {
        #         'train': TensorDataset(torch.tensor(train_FHN_x, dtype=torch.float32), torch.tensor(train_FHN_y, dtype=torch.float32)),
        #         'test': TensorDataset(torch.tensor(test_FHN_x, dtype=torch.float32), torch.tensor(test_FHN_y, dtype=torch.float32))
        #     }

    def train_dataloader(self, *args, **kwargs):
        return self._data_loader(self.dataset['train'], shuffle=True)

    def val_dataloader(self, *args, **kwargs):
        return self._data_loader(self.dataset['test'], shuffle=False)

    def test_dataloader(self, *args, **kwargs):
        return self._data_loader(self.dataset['test'], shuffle=False)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=10,
            shuffle=shuffle,
            persistent_workers=True
        )