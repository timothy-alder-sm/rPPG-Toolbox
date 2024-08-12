import pyeod
import glob
import os
from tqdm import tqdm
import numpy as np
import cv2
import io
import imageio
from dataset.data_loader.BaseLoader import BaseLoader
import pandas as pd


# Preprocess steps:

# DeepPhys: 
# Defines gold-standard physiological PPG signal under **4.1** as video with m'(t) 
# (see section for definition of m'(t)) as inside two standard deviations. Only 
# use gold-standard for training.
#
# Uses piecewise cubic Hermite interpolation to interpolate PPG signal to video 
# frame data.

class EODLoader:
    def __init__(self, BaseLoader, name, data_path, config_data):
        super.__init__(name, data_path, config_data)

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For MR-NIRP dataset)."""
        data_dirs = glob.glob(data_path + os.sep + "Subject*" + os.sep + "subject*")

        if not data_dirs:
            raise ValueError("dataset data paths empty!")
        dirs = [{"index": os.path.basename(data_dir), "path": data_dir} for data_dir in data_dirs]
        return dirs


    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values."""
        if begin == 0 and end == 1:  # return the full directory if begin == 0 and end == 1
            return data_dirs

        file_num = len(data_dirs)
        choose_range = range(int(begin * file_num), int(end * file_num))
        data_dirs_new = []

        for i in choose_range:
            data_dirs_new.append(data_dirs[i])

        return data_dirs_new
    

    def load_preprocessed_data(self):
        """ Loads the preprocessed data listed in the file list.
        """
        file_list_path = self.file_list_path  # get list of files in
        file_list_df = pd.read_csv(file_list_path)
        base_inputs = file_list_df['input_files'].tolist()
        filtered_inputs = []
        print(self.filtering.SELECT_TASKS)
        print(self.filtering.TASK_LIST)
        print(base_inputs)

        for input in base_inputs:
            input_name = input.split(os.sep)[-1].split('.')[0].rsplit('_', 1)[0]
            subject_name = input_name.rsplit('_')[0]
            task = input_name.rsplit('_', 1)[0].split('_', 1)[1]
            subject_task = input_name.rsplit('_', 1)[0]

            if self.filtering.SELECT_TASKS:
                if input_name not in self.filtering.TASK_LIST and subject_name not in self.filtering.TASK_LIST and task not in self.filtering.TASK_LIST and subject_task not in self.filtering.TASK_LIST:
                    continue
                
            if self.filtering.USE_EXCLUSION_LIST:
                if input_name in self.filtering.EXCLUSION_LIST or subject_name in self.filtering.EXCLUSION_LIST or task in self.filtering.EXCLUSION_LIST or subject_task in self.filtering.EXCLUSION_LIST:
                    continue

            print(input_name)
            filtered_inputs.append(input)

        if not filtered_inputs:
            raise ValueError(self.dataset_name + ' dataset loading data error!')
        
        filtered_inputs = sorted(filtered_inputs)  # sort input file name list
        labels = [input_file.replace("input", "label") for input_file in filtered_inputs]
        self.inputs = filtered_inputs
        self.labels = labels
        self.preprocessed_data_len = len(filtered_inputs)

    @staticmethod
    def read_video_unzipped(video_file):
        frames = list()
        all_pgm = sorted(glob.glob(os.path.join(video_file, "Frame*.pgm")))
        for pgm_path in all_pgm:
            try:
                frame = cv2.imread(pgm_path, cv2.IMREAD_UNCHANGED)          # read 10bit raw image (in uint16 format)
                frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2RGB)         # Demosaice rggb to RGB Image
            except:
                print("Error in reading frame:", pgm_path)
                continue
            
            frame = (frame >> 8).astype(np.uint8)                       # convert from uint16 to uint8

            frames.append(frame)
            
        return np.asarray(frames, dtype=np.uint8)
    
    @staticmethod
    def read_wave_unzipped(wave_file):
        """Reads a bvp signal file."""
        raw_data = loadmat(wave_file + os.sep + "pulseOx.mat")
        
        timestamps = (raw_data['pulseOxTime'][0] - raw_data['pulseOxTime'][0][0])
        ppg = raw_data['pulseOxRecord'][0]
        
        return ppg, timestamps