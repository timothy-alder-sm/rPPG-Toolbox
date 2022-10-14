"""The dataloader for SCAMPS datasets.

Details for the SCAMPS Dataset see https://github.com/danmcduff/scampsdataset
If you use this dataset, please cite the following publication:
McDuff, Daniel and Wander, Miah and Liu, Xin and Hill, Brian L and Hernandez, Javier and Lester, Jonathan and Baltrusaitis, Tadas
SCAMPS: Synthetics for Camera Measurement of Physiological Signals
in: Conference on Neural Information Processing Systems' 2022
"""
import glob
import json
import os
import re
from multiprocessing import Pool, Process, Value, Array, Manager

import cv2
import mat73
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm
from utils.utils import sample


class SCAMPSLoader(BaseLoader):
    """The data loader for the SCAMPS Processed dataset."""

    def __init__(self, name, data_path, config_data):
        """Initializes an SCAMPS Processed dataloader.
            Args:
                data_path(string): path of a folder which stores raw video and ground truth biosignal in mat files.
                Each mat file contains a video sequence of resolution of 72x72 and various ground trugh signal.
                e.g., dXsub -> raw/normalized data; d_ppg -> pulse signal, d_br -> resp signal
                -----------------
                     ProcessedData/
                     |   |-- P000001.mat/
                     |   |-- P000002.mat/
                     |   |-- P000003.mat/
                     ...
                -----------------
                name(str): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data)

    def get_data(self, data_path):
        """Returns data directories under the path(For COHFACE dataset)."""
        data_dirs = glob.glob(data_path + os.sep + "*.mat")
        if not data_dirs:
            raise ValueError(self.name + " dataset get data error!")
        dirs = list()
        for data_dir in data_dirs:
            subject = os.path.split(data_dir)[-1]
            dirs.append({"index": subject, "path": data_dir})
        return dirs

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i):
        """   invoked by preprocess_dataset for multi_process.   """

        matfile_path = data_dirs[i]['path']

        frames = self.read_video(matfile_path)
        frames = (np.round(frames * 255)).astype(np.uint8)
        bvps = self.read_wave(matfile_path)
        frames_clips, bvps_clips = self.preprocess(
            frames, bvps, config_preprocess)
        count, input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips,
                                                                          data_dirs[i]['index'])

    def preprocess_dataset(self, data_dirs, config_preprocess, begin, end):
        """Preprocesses the raw data."""
        file_num = len(data_dirs)
        print("file_num:", file_num)
        choose_range = range(0, file_num)
        if begin != 0 or end != 1:
            choose_range = range(int(begin * file_num), int(end * file_num))
            print(choose_range)
        pbar = tqdm(list(choose_range))
        # multi_process
        p_list = []
        running_num = 0
        for i in choose_range:
            process_flag = True
            while process_flag:  # ensure that every i creates a process
                if running_num < 16:  # in case of too many processes
                    p = Process(target=self.preprocess_dataset_subprocess, args=(data_dirs, config_preprocess, i))
                    p.start()
                    p_list.append(p)
                    running_num += 1
                    process_flag = False
                for p_ in p_list:
                    if not p_.is_alive():
                        p_list.remove(p_)
                        p_.join()
                        running_num -= 1
                        pbar.update(1)
        # join all processes
        for p_ in p_list:
            p_.join()
            pbar.update(1)
        pbar.close()

        # load all data and corresponding labels (sorted for consistency)
        self.load()

    def preprocess_dataset_backup(self, data_dirs, config_preprocess):
        """Preprocesses the raw data."""
        file_num = len(data_dirs)
        pbar = tqdm(list(range(file_num)))
        for i in pbar:
            matfile_path = data_dirs[i]['path']
            pbar.set_description("Processing %s" % matfile_path)
            frames = self.read_video(matfile_path)
            bvps = self.read_wave(matfile_path)
            frames_clips, bvps_clips = self.preprocess(
                frames, bvps, config_preprocess)
            self.len += self.save(frames_clips, bvps_clips,
                                  data_dirs[i]['index'])

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T,H,W,3) """
        mat = mat73.loadmat(video_file)
        frames = mat['Xsub']  # load raw frames
        return np.asarray(frames)

    @staticmethod
    def read_wave(wave_file):
        """Reads a bvp signal file."""
        mat = mat73.loadmat(wave_file)
        ppg = mat['d_ppg']  # load raw frames
        return np.asarray(ppg)