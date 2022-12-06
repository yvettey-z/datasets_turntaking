from argparse import ArgumentParser
import os
from os.path import expanduser, join, exists
from os import listdir, cpu_count
import re
import shutil
import librosa
import math
import gc
from typing import Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from datasets import load_from_disk, load_dataset
from datasets import Sequence
import pytorch_lightning as pl
import numpy as np
import cv2 as cv
import ffmpeg
import logging



CACHE_PATH = join(expanduser("~"), ".cache/datasets_turntaking/conversational")
class ConversationalDM2(pl.LightningDataModule):
    
    def __init__(
        self,
        tokenizer,
        datasets='ami',
        datasets_subset = 'headset-single',
        savepath='/ocean/projects/cis220078p/stomita/project/dataset',
        tensorpath = '/ocean/projects/cis220078p/stomita/project/dataset_tensor/',
        videodirpath = '/ocean/projects/cis220078p/yasano/amicorpus/',
        batch_size=16,
        max_length=1024,
        num_workers=5,
        pin_memory=True,
        overwrite=False,
        include_dialog=False,
        load_from_cache_file=True,
        num_proc=5,
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

        self.savepath = savepath
        self.tensor_path = tensorpath
        self.videodir_path = videodirpath
        self.overwrite = overwrite

    def get_split_path(self, split):
        pass
        # return join(self.savepath, split)

    def filter_empty_turns(self, examples):
        """
        return only dialogs with no empty turns
        """
        for utterance in examples["dialog"]:
            if utterance == "" or not re.search(r"\w", utterance):  # utt is empty
                return False
        return True

    def get_video(self, word_id, video_type): 
        path_video = self.videodir_path + word_id + '/video/'
        counter = 0
        # sometimes videocapture failed to get video, but somehow loading video multiple times works
        # so here video will be loaded until loaded correctly
        while True:
            video = cv.VideoCapture(path_video + word_id + '.' + video_type + '.avi', cv.CAP_ANY)
            if video.isOpened():
                break
            if counter > 100:
                print(f'loading video {path_video + word_id + "." + video_type + ".avi"}')
                raise RuntimeError(f'loading video {path_video + word_id + "." + video_type + ".avi"}, unable to load video')
            counter += 1
        return video


    def get_frame(self, video, word_start_times, word_end_times, fps=25, frame_definiton='end'):
        """
        inputs: 
          video (cv.VideoCapture) : a video to get a frame 
          word_start_times (float): the starting time of a word
          word_end_times   (float): the ending time of a word
          fps (int): fps of video. In AMI corpus, it is 25.
          frame_definiton (str, 'end' or 'middle'): if 'end', it will return the last frame. 
                                                    if 'middle', it will return the frame in the middle. 
        """
        # convert times to ms scale
        time_start = float(1000*word_start_times)
        time_end = float(1000*word_end_times)
        
        # if we get middle time 
        if frame_definiton == 'middle':
          time_end = (time_start+time_end)/2

        # set time to the ending time
        video.set(cv.CAP_PROP_POS_MSEC, time_end-fps)
        available, frame = video.read()
        if available:
          return frame
        else:
          # return zero array with shape of frame in AMI
          return np.zeros([288, 352, 3])


    def encode(self, examples):
        return self.tokenizer(examples)

    def prepare_data(self, skip_preprocessing = True):
        if not skip_preprocessing:
            # in PSC you need to activate ffmpeg module
            os.system('module load ffmpeg')
            
            if not os.path.exists(self.savepath):
                datasets = load_dataset(self.datasets, self.datasets_subset)
                for split in ["train", "validation", "test"]:
                    dataset = datasets[split]
                    if split == 'train':
                        dataset = dataset.select([i for i in range(114)])
                    elif split == 'validation':
                        dataset = dataset.select([i for i in range(12)])
                    elif split == 'test':
                        dataset = dataset.select([i for i in range(11)])
                    dataset = dataset.map(
                        self.encode,
                        num_proc=self.num_proc,
                    )
                    dataset.save_to_disk(os.path.join(self.savepath, split))

            for split in ["train", "validation", "test"]:
                tensor_path = self.tensor_path + split
                dataset = load_from_disk(os.path.join(self.savepath, split))
                for i in range(len(dataset)):
                    for j in range(len(dataset[i]['word'])):
                        save_dict_name = split + str(i) + 'part' + str(j) + '.npz'
                        data_path = os.path.join(tensor_path, save_dict_name)
                        if not os.path.exists(data_path):
                            dataset = load_from_disk(os.path.join(self.savepath, split))
                            words = dataset[i]['word'][j]
                            speakers = dataset[i]['speaker_ids'][j]
                            starting_time_list = list(dataset[i]['word_start_times'][j])
                            ending_time_list = list(dataset[i]['word_start_times'][j])
                            word_ids_list = list(dataset[i]['word_ids'][j])
                            frame_closeup1_list = []
                            frame_closeup2_list = []
                            frame_closeup3_list = []
                            frame_closeup4_list = []
                            frame_corner_list = []
                                
                            prev_word_ids = word_ids_list[0]
                            video = self.get_video(prev_word_ids, 'Closeup1')
                            for k in range(len(word_ids_list)):
                                if prev_word_ids != word_ids_list[k]:
                                    prev_word_ids =word_ids_list[k]
                                    video = self.get_video(prev_word_ids, 'Closeup1')
                                frame_closeup1 = self.get_frame(video, starting_time_list[k], ending_time_list[k], frame_definiton='end') 
                                frame_closeup1_list.append(frame_closeup1)
                                del frame_closeup1
                            del video
                            gc.collect()
                            print('finish Closeup1')

                            prev_word_ids = word_ids_list[0]
                            video = self.get_video(prev_word_ids, 'Closeup2')
                            for k in range(len(word_ids_list)):
                                if k % 500 == 0:
                                    print(k)
                                if prev_word_ids != word_ids_list[k]:
                                    prev_word_ids = word_ids_list[k]
                                    video = self.get_video(prev_word_ids, 'Closeup2')
                                frame_closeup2 = self.get_frame(video, starting_time_list[k], ending_time_list[k], frame_definiton='end') 
                                frame_closeup2_list.append(frame_closeup2)
                                del frame_closeup2
                            del video
                            gc.collect()
                            print('finish Closeup2')
                                
                            prev_word_ids = word_ids_list[0]
                            video = self.get_video(prev_word_ids, 'Closeup3')
                            for k in range(len(word_ids_list)):
                                if prev_word_ids != word_ids_list[k]:
                                    prev_word_ids = word_ids_list[k]
                                    video = self.get_video(prev_word_ids, 'Closeup3')
                                frame_closeup3 = self.get_frame(video, starting_time_list[k], ending_time_list[k], frame_definiton='end') 
                                frame_closeup3_list.append(frame_closeup3)
                                del frame_closeup3
                            del video
                            gc.collect()
                            print('finish Closeup3')
                                
                            prev_word_ids = word_ids_list[0]
                            video = self.get_video(prev_word_ids, 'Closeup4')
                            for k in range(len(word_ids_list)):
                                if prev_word_ids != word_ids_list[k]:
                                    prev_word_ids = word_ids_list[k]
                                    video = self.get_video(prev_word_ids, 'Closeup4')
                                frame_closeup4 = self.get_frame(video, starting_time_list[k], ending_time_list[k], frame_definiton='end') 
                                frame_closeup4_list.append(frame_closeup4)
                                del frame_closeup4
                            del video
                            gc.collect()
                            print('finish Closeup4')
                                
                            prev_word_ids = word_ids_list[0]
                            if prev_word_ids[:2] == 'ES':
                                video = self.get_video(prev_word_ids, 'Corner')
                            elif prev_word_ids[:2] == 'IS':
                                video = self.get_video(prev_word_ids, 'C')
                            elif prev_word_ids[:2] == 'TS':
                                if (prev_word_ids == 'TS3008b') | (prev_word_ids == 'TS3008c') | (prev_word_ids == 'TS3008d'):
                                    video = self.get_video(prev_word_ids, 'Overview3')
                                else:
                                    video = self.get_video(prev_word_ids, 'Overview2')

                            for k in range(len(word_ids_list)):
                                if prev_word_ids != word_ids_list[k]:
                                    prev_word_ids = word_ids_list[k]
                                    if prev_word_ids[:2] == 'ES':
                                        video = self.get_video(prev_word_ids, 'Corner')
                                    elif prev_word_ids[:2] == 'IS':
                                        video = self.get_video(prev_word_ids, 'C')
                                    elif prev_word_ids[:2] == 'TS':
                                        if (prev_word_ids == 'TS3008b') | (prev_word_ids == 'TS3008c') | (prev_word_ids == 'TS3008d'):
                                            video = self.get_video(prev_word_ids, 'Overview3')
                                        else:
                                            video = self.get_video(prev_word_ids, 'Overview2')
                                frame_corner = self.get_frame(video, starting_time_list[k], ending_time_list[k], frame_definiton='end') 
                                frame_corner_list.append(frame_corner)
                                del frame_corner
                            del video
                            gc.collect()

                            input_ids = np.array(words)
                            speaker_ids = np.array(speakers)                        
                            closeup1 = np.array(frame_closeup1_list)
                            closeup2 = np.array(frame_closeup2_list)
                            closeup3 = np.array(frame_closeup3_list)
                            closeup4 = np.array(frame_closeup4_list)
                            corner = np.array(frame_corner_list)
                                
                            del words, speakers, frame_closeup1_list, frame_closeup2_list, frame_closeup3_list, frame_closeup4_list, frame_corner_list 
                            gc.collect()
                                                        
                            np.savez_compressed(data_path, input_ids = input_ids, 
                                                speaker_ids = speaker_ids, closeup1 = closeup1,
                                                closeup2 = closeup2, closeup3 = closeup3, 
                                                closeup4 = closeup4, corner = corner)
                            
                            del input_ids, speaker_ids, closeup1, closeup2, closeup3, closeup4, corner
                            gc.collect()
                
        train_dir_path = os.path.join(self.tensor_path, 'train')
        # so far we moved validation data to different folder
        #val_dir_path = os.path.join(self.tensor_path, 'validation')
        val_dir_path = '/ocean/projects/cis220078p/yasano/amicorpus/validation'
        test_dir_path = os.path.join(self.tensor_path, 'test')
        
        dataset_list_train = self.get_dset_paths(train_dir_path)
        dataset_list_val = self.get_dset_paths(val_dir_path)
        dataset_list_test = self.get_dset_paths(test_dir_path)
        
        self.train_dset = dataset_list_train
        self.val_dset = dataset_list_val
        self.test_dset = dataset_list_test

    def get_dset_paths(self, directory):
        """
        Args:
            directory (str): name of the directory
        Outputs:
            file_path_list (list): list of absolute path in the directory
        """
        file_path_list = []
        for dirpath,_,filenames in os.walk(directory):
            for f in filenames:
                file_path_list.append(os.path.join(dirpath, f))
        return file_path_list
                

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        pass
        """
        if stage == "fit" or stage is None:
            self.train_dset = load_from_disk(self.get_split_path("train"))
            self.val_dset = load_from_disk(self.get_split_path("validation"))

        if stage == "test":
            self.test_dset = load_from_disk(self.get_split_path("test"))
        """

    def collate_fn(self, batch):
        # 'batch' will be iterable of path name
        # load data here
        batch_dict = [np.load(path) for path in batch]
        
        # create batch of tensor 
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

        # pad_sequence to input words and frames
        input_word_pad = pad_sequence(input_word, batch_first = True, padding_value=self.tokenizer.tokenizer.pad_token_id)
        input_closeup1_pad = pad_sequence(input_closeup1, batch_first = True, padding_value=0)
        input_closeup2_pad = pad_sequence(input_closeup2, batch_first = True, padding_value=0)
        input_closeup3_pad = pad_sequence(input_closeup3, batch_first = True, padding_value=0)
        input_closeup4_pad = pad_sequence(input_closeup4, batch_first = True, padding_value=0)
        input_corner_pad = pad_sequence(input_corner, batch_first = True, padding_value=0)
        
        # for speaker_id, pad the last speaker id
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
        
        del input_word, input_speaker, input_closeup1, input_closeup2, input_closeup3, input_closeup4, input_corner
        gc.collect()

        return {'input_ids': input_word_pad, 'speaker_ids': input_speaker_pad, 'attention_mask': attention_mask,
                'closeup1': input_closeup1_pad, 'closeup2': input_closeup2_pad, 'closeup3': input_closeup3_pad, 'closeup4': input_closeup4_pad,
                'corner': input_corner_pad}
    
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
        parser.add_argument("--savepath", default=None, type=str)
        parser.add_argument("--overwrite", default=False, type=bool)
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




