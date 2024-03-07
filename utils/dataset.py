import torch
import torch.utils.data as data_utils
from torch.utils.data.distributed import DistributedSampler
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

tqdm.pandas()

def train_val_test_split(user_history):
    train_history = {}
    val_history = {}
    test_history = {}
    for key, history in tqdm(user_history.items()):
        train_history[key] = history[:-2]
        val_history[key] = history[-2:-1]
        test_history[key] = history[-1:]
    return train_history, val_history, test_history


def get_train_loader(dataset, args):
    if args.is_parallel:
        dataloader = data_utils.DataLoader(dataset, batch_size=args.train_batch_size, sampler=DistributedSampler(dataset))
    else:
        dataloader = data_utils.DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, pin_memory=True)
    return dataloader

def get_val_loader(dataset, args):
    if args.is_parallel:
        dataloader = data_utils.DataLoader(dataset, batch_size=args.val_batch_size, sampler=DistributedSampler(dataset))
    else:
        dataloader = data_utils.DataLoader(dataset, batch_size=args.val_batch_size, shuffle=False, pin_memory=True)
    return dataloader

def get_test_loader(dataset, args):
    if args.is_parallel:
        dataloader = data_utils.DataLoader(dataset, batch_size=args.test_batch_size, sampler=DistributedSampler(dataset))
    else:
        dataloader = data_utils.DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False, pin_memory=True)
    return dataloader


class Build_full_EvalDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len, mask_token, num_item):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.mask_token = mask_token
        self.num_item = num_item + 1

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user][:-1]
        answer = self.u2answer[user]
        answer = answer[-1:][0]


        labels = [0] * self.num_item
        labels[answer] = 1
        seq = seq + [self.mask_token]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [self.mask_token] * padding_len + seq

        return torch.LongTensor(seq), torch.LongTensor(labels)


class reset_df(object):

    def __init__(self):
        print("=" * 10, "Initialize Reset DataFrame Object", "=" * 10)
        self.item_enc = LabelEncoder()
        self.user_enc = LabelEncoder()

    def fit_transform(self, df):
        print("=" * 10, "Resetting user ids and item ids in DataFrame", "=" * 10)
        df['item_id'] = self.item_enc.fit_transform(df['item_id']) + 1
        df['user_id'] = self.user_enc.fit_transform(df['user_id']) + 1
        return df

    def inverse_transform(self, df):
        df['item_id'] = self.item_enc.inverse_transform(df['item_id']) - 1
        df['user_id'] = self.user_enc.inverse_transform(df['user_id']) - 1
        return df


class BuildTrainDataset(data_utils.Dataset):
    def __init__(self, u2seq, max_len, mask_token, num_items):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_token = mask_token
        self.num_items = num_items

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self._getseq(user)

        tokens = seq[:-1]
        labels = seq[1:]
        #
        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]

        x_len = len(tokens)
        y_len = len(labels)

        x_mask_len = self.max_len - x_len
        y_mask_len = self.max_len - y_len


        tokens = [self.mask_token] * x_mask_len + tokens
        labels = [self.mask_token] * y_mask_len + labels

        return torch.LongTensor(tokens), torch.LongTensor(labels)

    def _getseq(self, user):
        return self.u2seq[user]

class BertTrainDataset(data_utils.Dataset):
    def __init__(self, u2seq, max_len, mask_prob, mask_token, num_items, rng):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.num_items = num_items
        self.rng = rng

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self._getseq(user)

        tokens = []
        labels = []
        for s in seq:
            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob

                if prob < 0.8:
                    tokens.append(self.mask_token)
                elif prob < 0.9:
                    tokens.append(self.rng.randint(1, self.num_items))
                else:
                    tokens.append(s)

                labels.append(s)
            else:
                tokens.append(s)
                labels.append(0)

        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]

        mask_len = self.max_len - len(tokens)

        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels

        return torch.LongTensor(tokens), torch.LongTensor(labels)

    def _getseq(self, user):
        return self.u2seq[user]

