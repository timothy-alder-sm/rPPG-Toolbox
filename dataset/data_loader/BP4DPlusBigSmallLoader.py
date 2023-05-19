"""The dataloader for BP4D+ Big Small datasets. This dataloader was adapted from the following git repository 
based on rPPG Toolbox: https://github.com/girishvn/BigSmall

Details on the BigSmall model can be found here: https://girishvn.github.io/BigSmall/
Details for the BP4D+ Dataset see https://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html

If you use this dataset, please cite the following publications:

Xing Zhang, Lijun Yin, Jeff Cohn, Shaun Canavan, Michael Reale, Andy Horowitz, Peng Liu, and Jeff Girard
“BP4D-Spontaneous: A high resolution spontaneous 3D dynamic facial expression database”
Image and Vision Computing, 32 (2014), pp. 692-706  (special issue of the Best of FG13)

AND

Xing Zhang, Lijun Yin, Jeff Cohn, Shaun Canavan, Michael Reale, Andy Horowitz, and Peng Liu
“A high resolution spontaneous 3D dynamic facial expression database”
The 10th IEEE International Conference on Automatic Face and Gesture Recognition (FG13),  April, 2013. 

If you use the BigSmall model or preprocessing please cite the following publication:

Girish Narayanswamy, Yujia Liu, Yuzhe Yang, Chengqian Ma, Xin Liu, Daniel McDuff, and Shwetak Patel
"BigSmall: Efficient Multi-Task Learning for Disparate Spatial and Temporal Physiological Measurements"
arXiv:2303.11573 (https://arxiv.org/abs/2303.11573)

"""

import glob
import zipfile
import os
import re

import cv2
from skimage.util import img_as_float
import numpy as np
import pandas as pd
import pickle 

from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm

from dataset.data_loader.BaseLoader import BaseLoader


class BP4DPlusBigSmallLoader(BaseLoader):
    """The data loader for the BP4D+ dataset."""

    def __init__(self, dataset_name, raw_data_path, config_data):
        """Initializes an BP4D+ dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                -----------------
                    RawData/
                    |   |-- 2D+3D/
                    |       |-- F001.zip/
                    |       |-- F002.zip
                    |       |...
                    |   |-- 2DFeatures/
                    |       |-- F001_T1.mat
                    |       |-- F001_T2.mat
                    |       |...
                    |   |-- 3DFeatures/
                    |       |-- F001_T1.mat
                    |       |-- F001_T2.mat
                    |       |...
                    |   |-- AUCoding/
                    |       |-- AU_INT/
                    |            |-- AU06/
                    |               |-- F001_T1_AU06.csv
                    |               |...
                    |           |...
                    |       |-- AU_OCC/
                    |           |-- F00_T1.csv 
                    |           |...
                    |   |-- IRFeatures/
                    |       |-- F001_T1.txt
                    |       |...
                    |   |-- Physiology/
                    |       |-- F001/
                    |           |-- T1/
                    |               |-- BP_mmHg.txt
                    |               |-- microsiemens.txt
                    |               |--LA Mean BP_mmHg.txt
                    |               |--LA Systolic BP_mmHg.txt
                    |               |-- BP Dia_mmHg.txt
                    |               |-- Pulse Rate_BPM.txt
                    |               |-- Resp_Volts.txt
                    |               |-- Respiration Rate_BPM.txt
                    |       |...
                    |   |-- Thermal/
                    |       |-- F001/
                    |           |-- T1.mv
                    |           |...
                    |       |...
                    |   |-- BP4D+UserGuide_v0.2.pdf
                -----------------
                name(str): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """

        self.inputs = list()
        self.labels = list()
        self.dataset_name = dataset_name
        self.raw_data_path = raw_data_path
        self.cached_path = config_data.CACHED_PATH
        self.file_list_path = config_data.FILE_LIST_PATH
        self.preprocessed_data_len = 0
        self.data_format = config_data.DATA_FORMAT
        self.do_preprocess = config_data.DO_PREPROCESS
        

        assert (config_data.BEGIN < config_data.END)
        assert (config_data.BEGIN > 0 or config_data.BEGIN == 0)
        assert (config_data.END < 1 or config_data.END == 1)

        if config_data.DO_PREPROCESS:
            self.preprocess_dataset(config_data)
        else:
            if not os.path.exists(self.cached_path):
                raise ValueError(self.dataset_name,
                                 'Please set DO_PREPROCESS to True. Preprocessed directory does not exist!')
            if not os.path.exists(self.file_list_path):
                print('File list does not exist... generating now...')
                self.build_file_list_retroactive(self.raw_data_dirs, config_data.BEGIN, config_data.END)
                print('File list generated.', end='\n\n')

            self.load_preprocessed_data()

        print('Cached Data Path', self.cached_path, end='\n\n')
        print('File List Path', self.file_list_path)
        print(f" {self.dataset_name} Preprocessed Dataset Length: {self.preprocessed_data_len}", end='\n\n')


    def preprocess_dataset(self, config_data):
        print('Starting Preprocessing...')

        # GET DATASET INFORMATION (PATHS AND OTHER META DATA REGARDING ALL VIDEO TRIALS)
        data_dirs = self.get_raw_data(config_data)

        # REMOVE ALREADY PREPROCESSED SUBJECTS
        data_dirs = self.adjust_data_dirs(data_dirs, config_data)

        # CREATE CACHED DATA PATH
        cached_path = config_data.CACHED_PATH
        if not os.path.exists(cached_path):
            os.makedirs(cached_path, exist_ok=True)

        # READ RAW DATA, PREPROCESS, AND SAVE PROCESSED DATA FILES
        file_list_dict = self.multi_process_manager(data_dirs, config_data)

        #TODO build file lists
        print("DONE Preprocessing!")


    def get_raw_data(self, config_data):
        """Returns data directories under the path(For PURE dataset)."""

        data_path = config_data.RAW_DATA_PATH # get raw data path

        # GET ALL SUBJECT TRIALS IN DATASET
        f_subj_trials = glob.glob(os.path.join(data_path, "Physiology", "F*", "T*"))
        m_subj_trials = glob.glob(os.path.join(data_path, "Physiology", "M*", "T*"))
        subj_trials = f_subj_trials + m_subj_trials

        # SPLIT PATH UP INTO INFORMATION (SUBJECT, TRIAL, ETC.)
        data_dirs = list()
        for trial_path in subj_trials:
            trial_data = trial_path.split(os.sep)
            index = trial_data[-2] + trial_data[-1] # should be of format: F008T8
            trial = trial_data[-1] # trial number 
            subj_sex = index[0] # subject biological sex
            subject = int(index[1:4]) # subject number (by sex)

            # If processesing AU Subset only process trials T1, T6, T7, T8 (only ones that have AU labels)
            if not trial in ['T1', 'T6', 'T7', 'T8']:
                continue
            
            # append information to data dirs list
            data_dirs.append({"index": index, "path": data_path, "subject": subject, "trial": trial, "sex": subj_sex})

        # RETURN DATA DIRS 
        return data_dirs



    def adjust_data_dirs(self, data_dirs, config_preprocess):
        file_list = glob.glob(os.path.join(config_preprocess.CACHED_PATH, '*label*.npy'))
        trial_list = [f.replace(config_preprocess.CACHED_PATH, '').split('_')[0].replace(os.sep, '') for f in file_list]
        trial_list = list(set(trial_list)) # get a list of completed video trials

        for d in data_dirs:
            idx = d['index']

            if idx in trial_list: # if trial has already been processed
                data_dirs.remove(d)

        return data_dirs
    



    def preprocess_dataset_subprocess(self, data_dirs, config_data, i, file_list_dict):
        """ invoked by preprocess_dataset for multi_process """

        data_dir_info = data_dirs[i] # get data raw data file path 
        saved_filename = data_dirs[i]['index'] # get subject and trial in format of  FXXXTXX

        # CONSTRUCT DATA DICTIONARY FOR VIDEO TRIAL
        data_dict = self.construct_data_dict(data_dir_info, config_data) # construct a dictionary of ALL labels and video frames (of equal length)
        data_dict = self.generate_psuedo_labels(data_dict) # adds POS psuedo BVP labels to dataset
        
        # SEPERATE DATA INTO VIDEO FRAMES AND LABELS ARRAY
        frames = self.read_video(data_dict) # read in the video frames
        labels = self.read_labels(data_dict) # read in video labels 
        if frames.shape[0] != labels.shape[0]: # check if data and labels are the same length
            raise ValueError(' Preprocessing dataset subprocess: frame and label time axis not the same')
        
        # PREPROCESS VIDEO FRAMES AND LABELS (eg. DIFF-NORM, RAW_STD)
        big_clips, small_clips, labels_clips = preprocess(frames, labels, config_data)

        # SAVE PREPROCESSED FILE CHUNKS
        count, input_name_list, label_name_list = save_multi_process(big_clips, small_clips, labels_clips, saved_filename, config_data)

        file_list_dict[i] = input_name_list



    def construct_data_dict(self, data_dir_info, config_preprocess):

        # GET TRIAL NUMBER 
        trial = data_dir_info['trial']

        # BUILD DICTIONARY TO STORE FRAMES AND LABELS
        data_dict = dict()

        # READ IN RAW VIDEO FRAMES
        data_dict = self.read_raw_vid_frames(data_dir_info, config_preprocess, data_dict)

        # READ IN RAW PHYSIOLOGICAL SIGNAL LABELS 
        data_dict = self.read_raw_phys_labels(data_dir_info, data_dict)

        # READ IN ACTION UNIT (AU) LABELS (if trial in [1, 6, 7, 8]: trials w/ AU labels)
        if trial in ['T1', 'T6', 'T7', 'T8']:
            data_dict, start_np_idx, end_np_idx = self.read_au_labels(data_dir_info, config_preprocess, data_dict)

            # CROP DATAFRAME W/ AU START END
            if config_preprocess['AU_SUBSET']: # if using only the AU dataset subset
                data_dict = self.crop_au_subset_data(data_dict, start_np_idx, end_np_idx)

        # FRAMES AND LABELS SHOULD BE OF THE SAME LENGTH
        for k in data_dict.keys():
            if not data_dict[k].shape[0] == data_dict['X'].shape[0]:
                print('Shape Mismatch', k, data_dict[k].shape[0])
                raise ValueError('Shape Mismatch')

        return data_dict
    


    def downsample_frame(self, frame, dim_h=144, dim_w=144):

        if dim_h == dim_w: # square crop
            vidLxL = cv2.resize(img_as_float(frame[int((frame.shape[0]-frame.shape[1])):,:,:]), (dim_h,dim_w), interpolation=cv2.INTER_AREA)
        else:
            vidLxL = cv2.resize(img_as_float(frame), (dim_h,dim_w), interpolation=cv2.INTER_AREA)

        return cv2.cvtColor(vidLxL.astype('float32'), cv2.COLOR_BGR2RGB)



    def read_raw_vid_frames(self, data_dir_info, config_preprocess, data_dict):

        data_path = data_dir_info['path']
        subject_trial = data_dir_info['index'][0:4]
        trial = data_dir_info['trial']

        # GRAB EACH FRAME FROM ZIP FILE
        imgzip = open(os.path.join(data_path, '2D+3D', subject_trial+'.zip'))
        zipfile_path = os.path.join(data_path, '2D+3D', subject_trial+'.zip')

        cnt = 0

        with zipfile.ZipFile(zipfile_path, "r") as zippedImgs:
            for ele in zippedImgs.namelist():
                ext = os.path.splitext(ele)[-1]
                ele_task = str(ele).split('/')[1]
                if ext == '.jpg' and ele_task == trial:
                    data = zippedImgs.read(ele)
                    vid_frame = cv2.imdecode(np.fromstring(data, np.uint8), cv2.IMREAD_COLOR)

                    dim_h = config_preprocess['BIG_H']
                    dim_w = config_preprocess['BIG_W']
                    vid_LxL = self.downsample_frame(vid_frame, dim_h=dim_h, dim_w=dim_w) # downsample frames (otherwise processing time becomes WAY TOO LONG)

                    # clip image values to range (1/255, 1)
                    vid_LxL[vid_LxL > 1] = 1
                    vid_LxL[vid_LxL < 1./255] = 1./255
                    vid_LxL = np.expand_dims(vid_LxL, axis=0)
                    if cnt == 0:
                        Xsub = vid_LxL
                    else:
                        Xsub = np.concatenate((Xsub, vid_LxL), axis=0)
                    cnt += 1
        
        if cnt == 0:
            return
        
        data_dict['X'] = Xsub
        return data_dict


    def read_raw_phys_labels(self, data_dir_info, data_dict):

        data_path = data_dir_info['path']
        subject = data_dir_info['index'][0:4] # of format F008
        trial = data_dir_info['trial'] # of format T05
        base_path = os.path.join(data_path, "Physiology", subject, trial)

        len_Xsub = data_dict['X'].shape[0] # TODO WHAT IS THE CORRECT FRAME INDEX?????????

        # READ IN PHYSIOLOGICAL LABELS TXT FILE DATA
        try:
            bp_wave = pd.read_csv(os.path.join(base_path, "BP_mmHg.txt")).to_numpy().flatten()
            HR_bpm = pd.read_csv(os.path.join(base_path, "Pulse Rate_BPM.txt")).to_numpy().flatten()
            resp_wave = pd.read_csv(os.path.join(base_path, "Resp_Volts.txt")).to_numpy().flatten()
            resp_bpm = pd.read_csv(os.path.join(base_path, "Respiration Rate_BPM.txt")).to_numpy().flatten()
            mean_BP = pd.read_csv(os.path.join(base_path, "LA Mean BP_mmHg.txt")).to_numpy().flatten()
            sys_BP = pd.read_csv(os.path.join(base_path, "LA Systolic BP_mmHg.txt")).to_numpy().flatten()
            dia_BP = pd.read_csv(os.path.join(base_path, "BP Dia_mmHg.txt")).to_numpy().flatten()
            eda = pd.read_csv(os.path.join(base_path, "EDA_microsiemens.txt")).to_numpy().flatten()
        except FileNotFoundError:
            print('Label File Not Found At Basepath', base_path)
            return

        # RESIZE SIGNALS TO LENGTH OF X (FRAMES) AND CONVERT TO NPY ARRAY
        bp_wave = np.interp(np.linspace(0, len(bp_wave), len_Xsub), np.arange(0, len(bp_wave)), bp_wave)
        HR_bpm = np.interp(np.linspace(0, len(HR_bpm), len_Xsub), np.arange(0, len(HR_bpm)), HR_bpm)
        resp_wave = np.interp(np.linspace(0, len(resp_wave), len_Xsub), np.arange(0, len(resp_wave)), resp_wave)
        resp_bpm = np.interp(np.linspace(0, len(resp_bpm), len_Xsub), np.arange(0, len(resp_bpm)), resp_bpm)
        mean_BP = np.interp(np.linspace(0, len(mean_BP), len_Xsub), np.arange(0, len(mean_BP)), mean_BP)
        sys_BP = np.interp(np.linspace(0, len(sys_BP), len_Xsub), np.arange(0, len(sys_BP)), sys_BP)
        dia_BP = np.interp(np.linspace(0, len(dia_BP), len_Xsub), np.arange(0, len(dia_BP)), dia_BP)
        eda = np.interp(np.linspace(0, len(eda), len_Xsub), np.arange(0, len(eda)), eda)

        data_dict['bp_wave'] = bp_wave
        data_dict['HR_bpm'] = HR_bpm
        data_dict['mean_bp'] = mean_BP
        data_dict['systolic_bp'] = sys_BP
        data_dict['diastolic_bp'] = dia_BP
        data_dict['resp_wave'] = resp_wave
        data_dict['resp_bpm'] = resp_bpm
        data_dict['eda'] = eda
        return data_dict  



    def read_au_labels(self, data_dir_info, config_preprocess, data_dict):

        # DATA PATH INFO    
        subj_idx = data_dir_info['index']
        base_path = config_preprocess['RAW_DATA_PATH']
        AU_OCC_url = os.path.join(base_path, 'AUCoding', "AU_OCC", subj_idx[0:4] + '_' + subj_idx[4:] + '.csv')

        # DATA CHUNK LENGTH
        frame_shape = data_dict['X'].shape[0]

        # READ IN AU CSV FILE
        AUs = pd.read_csv(AU_OCC_url, header = 0).to_numpy()

        # NOTE: START AND END FRAMES ARE 1-INDEXED
        start_frame = AUs[0,0] # first frame w/ AU encoding
        end_frame = AUs[AUs.shape[0] - 1, 0] # last frame w/ AU encoding

        # ENCODED AUs
        AU_num = [1, 2, 4, 5, 6, 7, 9, 10, 11, 
                12, 13, 14, 15, 16, 17, 18, 19, 20,
                22, 23, 24, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
        AU_int_num = [6, 10, 12, 14, 17] # AU w/ intensity encoding (0-5)
    
        # ITERATE THROUGH ENCODED AUs
        for au_idx, au in enumerate(AU_num): # Iterate through list of AUs

            # Define AU str name
            if au < 10:
                AU_key = 'AU' + '0' + str(au)
            else:
                AU_key = 'AU' + str(au)

            # GET SPECIFIC ACTION UNIT DATA
            aucoding = AUs[:, au_idx + 1] # indx + 1 as first row/column is index

            if start_frame > 1: # indx + 1 as first row/column is 1-indexed
                # pad the previous frame with -1
                aucoding = np.pad(aucoding, (start_frame - 1, 0), 'constant', constant_values = (-1, -1))
            if end_frame < frame_shape:
                # pad the following frame with -1 as well
                aucoding = np.pad(aucoding, (0, frame_shape - end_frame), 'constant', constant_values = (-1, -1))

            # Save out info to dict
            data_dict[AU_key] = aucoding

            # READ IN INTENSITY (INT) ENCODED AUs
            if au in AU_int_num:
                AU_INT_url = os.path.join(base_path, 'AUCoding', 'AU_INT', AU_key, subj_idx[0:4] + '_' + subj_idx[4:] + '_' + AU_key + '.csv')
                AUs_int = pd.read_csv(AU_INT_url, header = None).to_numpy() # read in each csv file
                assert (AUs_int.shape[0] == AUs.shape[0]) # ensure int encoding same length as binary encoding
                aucoding_int = AUs_int[:, 1]
                if start_frame > 1:
                    # pad the previous frame with -1
                    aucoding_int = np.pad(aucoding_int, (start_frame - 1, 0), 'constant', constant_values = (-1, -1))
                if end_frame < frame_shape:
                    # pad the following frame with -1
                    aucoding_int = np.pad(aucoding_int, (0, frame_shape - end_frame), 'constant', constant_values = (-1, -1))

                # Save out info to dict
                AU_int_key = AU_key + 'int'
                data_dict[AU_int_key] = aucoding_int

        # return start crop index if using AU subset data
        start_np_idx = start_frame - 1 
        end_np_idx = end_frame - 1
        return data_dict, start_np_idx, end_np_idx
        


    def crop_au_subset_data(data_dict, start, end):

        keys = data_dict.keys()

        # Iterate through video frames ad labels and crop based off start and end frame
        for k in keys:
            data_dict[k] = data_dict[k][start:end+1] # start and end frames are inclusive 

        return data_dict
