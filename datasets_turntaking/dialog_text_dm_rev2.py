from argparse import ArgumentParser
from os.path import expanduser, join, exists
from os import listdir, cpu_count
import re
import shutil
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_from_disk, load_dataset
import pytorch_lightning as pl


CACHE_PATH = join(expanduser("~"), ".cache/datasets_turntaking/conversational")


class ConversationalDM2(pl.LightningDataModule):
    
    def __init__(
        self,
        tokenizer,
        datasets='ami',
        datasets_subset = 'headset-single',
        savepath=None,
        batch_size=2,
        max_length=1024,
        num_workers=1,
        pin_memory=True,
        overwrite=False,
        include_dialog=False,
        load_from_cache_file=True,
        num_proc=None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.pin_memory = pin_memory
        self.num_workers = num_workers

        # `datasets` parameters
        self.load_from_cache_file = load_from_cache_file
        self.num_proc = num_proc
        self.include_dialog = include_dialog

        # Datasets
        self.datasets = datasets
        self.datasets_subset = datasets_subset

        if savepath is None:
            savepath = CACHE_PATH
        self.savepath = join(savepath, self.tokenizer.tokenizer.name_or_path)
        self.overwrite = overwrite

    def get_split_path(self, split):
        return join(self.savepath, split)

    def filter_empty_turns(self, examples):
        """
        return only dialogs with no empty turns
        """
        for utterance in examples["dialog"]:
            if utterance == "" or not re.search(r"\w", utterance):  # utt is empty
                return False
        return True

    def encode(self, examples):
        """omit `attention_mask`"""
        #t = self.tokenizer(examples)
        return self.tokenizer(examples)

    def prepare_data(self):
        datasets = load_dataset(self.datasets, self.datasets_subset)

        for split in ["train", "validation", "test"]:
            split_path = self.get_split_path(split)

            if (
                self.overwrite
                or not self.load_from_cache_file
                or not exists(split_path)
                or len(listdir(split_path)) == 0
            ):

                # Remove if it exists in order to overwrite
                if self.overwrite and exists(split_path):
                    shutil.rmtree(split_path)

                dataset = datasets[split]
                if split == 'train':
                    dataset = dataset.select([i for i in range(114)])
                else:
                    dataset = dataset.select([i for i in range(12)])
                dataset = dataset.map(
                    self.encode,
                    load_from_cache_file=self.load_from_cache_file,
                    num_proc=self.num_proc,
                )
                dataset_list = []
                for i in range(len(dataset)):
                  for j in range(len(dataset[i]['word_ids'])):
                    data_dict = {'input_ids': dataset[i]['word_ids'][j], 
                                 'speaker_ids': dataset[i]['speaker_ids'][j]}
                    dataset_list.append(data_dict)
                if split == 'train':
                    self.train_dset = dataset_list
                if split == 'validation':
                    self.val_dset = dataset_list
                if split == 'test':
                    self.test_dset = dataset_list


    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dset = load_from_disk(self.get_split_path("train"))
            self.val_dset = load_from_disk(self.get_split_path("validation"))

        if stage == "test":
            self.test_dset = load_from_disk(self.get_split_path("test"))

    def collate_fn(self, batch):        
        batch_dict = [np.load(path) for path in batch]
        input_word = [torch.tensor(b["input_ids"]) for b in batch_dict] # list of tensor(1024)
        input_speaker = [torch.tensor(b["speaker_ids"]) for b in batch_dict] # list of tensor(1024)
        input_closeup1 = [torch.tensor(b['closeup1']) for b in batch_dict] # list of tensor(1024 * H * W * 3)
        input_closeup2 = [torch.tensor(b['closeup2']) for b in batch_dict] # list of tensor(1024 * H * W * 3)
        input_closeup3 = [torch.tensor(b['closeup3']) for b in batch_dict] # list of tensor(1024 * H * W * 3)
        input_closeup4 = [torch.tensor(b['closeup4']) for b in batch_dict] # list of tensor(1024 * H * W * 3)
        input_corner = [torch.tensor(b['corner']) for b in batch_dict]

        # before padding everything, create original attention_mask without padding
        attention_mask_list = [torch.ones_like(word) for word in input_word]
        
        # in case all tensor in the batch is shorter than 1024, padding the first entity 
        if len(input_word[0]) != self.max_length:
            input_word[0] = torch.nn.functional.pad(input_word[0], (0, self.max_length-len(input_word[0])), 'constant', self.tokenizer.tokenizer.pad_token_id)
            input_closeup1[0] = torch.nn.functional.pad(input_closeup1[0].permute([1,2,3,0]), (0, self.max_length-len(input_word[0])), 'constant', 0).permute([3,0,1,2])
            input_closeup2[0] = torch.nn.functional.pad(input_closeup2[0].permute([1,2,3,0]), (0, self.max_length-len(input_word[0])), 'constant', 0).permute([3,0,1,2])
            input_closeup3[0] = torch.nn.functional.pad(input_closeup3[0].permute([1,2,3,0]), (0, self.max_length-len(input_word[0])), 'constant', 0).permute([3,0,1,2])
            input_closeup4[0] = torch.nn.functional.pad(input_closeup4[0].permute([1,2,3,0]), (0, self.max_length-len(input_word[0])), 'constant', 0).permute([3,0,1,2])
            input_corner[0] = torch.nn.functional.pad(input_corner[0].permute([1,2,3,0]), (0, self.max_length-len(input_word[0])), 'constant', 0).permute([3,0,1,2])
            input_overhead[0] = torch.nn.functional.pad(input_overhead[0].permute([1,2,3,0]), (0, self.max_length-len(input_word[0])), 'constant', 0).permute([3,0,1,2])

        # pad_sequence to input_word
        input_word_pad = pad_sequence(input_word, batch_first = True, padding_value=self.tokenizer.tokenizer.pad_token_id)
        input_closeup1_pad = pad_sequence(input_closeup1, batch_first = True, padding_value=0)
        input_closeup2_pad = pad_sequence(input_closeup2, batch_first = True, padding_value=0)
        input_closeup3_pad = pad_sequence(input_closeup3, batch_first = True, padding_value=0)
        input_closeup4_pad = pad_sequence(input_closeup4, batch_first = True, padding_value=0)
        input_corner_pad = pad_sequence(input_corner, batch_first = True, padding_value=0)
        input_overhead_pad = pad_sequence(input_overhead, batch_first = True, padding_value=0)
        

        # since padding_mode = 'replicate' didn't work, let's do it manually...
        # create a tensor to store the result
        input_speaker_pad = torch.zeros_like(input_word_pad)
        for i in range(len(input_speaker)):
          input_speaker_element = torch.nn.functional.pad(input_speaker[i], (0, self.max_length-len(input_speaker[i])), 'constant', input_speaker[i][-1].item())
          input_speaker_pad[i] = input_speaker_element

        # create a tensor to store the result
        attention_mask = torch.zeros_like(input_word_pad)
        for i in range(len(attention_mask_list)):
          attention_mask_element = torch.nn.functional.pad(attention_mask_list[i], (0, self.max_length-len(attention_mask_list[i])), 'constant', 0)
          attention_mask[i] = attention_mask_element
        
        del input_word, input_speaker, input_closeup1, input_closeup2, input_closeup3, input_closeup4, input_corner, input_overhead
        gc.collect()

        return {'input_ids': input_word_pad, 'speaker_ids': input_speaker_pad, 'attention_mask': attention_mask,
                'closeup1': input_closeup1_pad, 'closeup2': input_closeup2_pad, 'closeup3': input_closeup3_pad, 'closeup4': input_closeup4_pad,
                'corner': input_corner_pad, 'overhead': input_overhead_pad}    
    def train_dataloader(self):
        return DataLoader(
            self.train_dset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dset,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dset,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        n_cpus = cpu_count()
        parser.add_argument(
            "--datasets", type=str, nargs="+", default="ami"
        )
        parser.add_argument(
            "--max_length",
            default=1024,
            type=int,
            help="maximum length of sequences (applied in `collate_fn`)",
        )
        # arguments for `datasets` library
        parser.add_argument("--batch_size", default=4, type=int)
        parser.add_argument("--num_workers", default=n_cpus, type=int)
        parser.add_argument("--pin_memory", default=True, type=bool)
        parser.add_argument("--load_from_cache_file", default=True, type=bool)
        parser.add_argument("--num_proc", default=n_cpus, type=int)
        return parser





