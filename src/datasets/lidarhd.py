import os
import os.path as osp
import laspy
from laspy.file import File 
import sys
import torch
import shutil
import glob
import logging
from plyfile import PlyData
from src.datasets import BaseDataset
from src.data import Data
from src.datasets.lidarhd_config import *
from torch_geometric.data import Dataset, download_url, extract_tar


DIR = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)


# Occasional Dataloader issues with LidarHD on some machines. Hack to
# solve this:
# https://stackoverflow.com/questions/73125231/pytorch-dataloaders-bad-file-descriptor-and-eof-for-workers0
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


__all__ = ['LidarHD']


########################################################################
#                                 Utils                                #
########################################################################

import os.path as osp 
import numpy as np 
import laspy  
from torch_geometric.data import Dataset, download_url,Data    
from laspy.file import File 

COLORS_NORMALIZATION_MAX_VALUE = 255.0 * 256.0 
RETURN_NUMBER_NORMALIZATION_MAX_VALUE = 7.0 

def lidar_hd_pre_transform(points): 
    """Turn pdal points into torch-geometric Data object. 
    Builds a composite (average) color channel on the fly.     Calculate NDVI on the fly. 
    Args: 
        las_filepath (str): path to the LAS file. 
    Returns: 
        Data: the point cloud formatted for later deep learning training. 
    """ 

    # Positions and base features 
    pos = np.asarray([points["X"], points["Y"], points["Z"]], dtype=np.float32).transpose() 

    # normalization 
    occluded_points = points["return_number"] > 1 

    points["return_number"] = (points["return_number"]) / (RETURN_NUMBER_NORMALIZATION_MAX_VALUE) 
    points["number_of_returns"] = (points["number_of_returns"]) / ( 
        RETURN_NUMBER_NORMALIZATION_MAX_VALUE 
    ) 

    for color in ["red", "green", "blue", "nir"]: 
        assert points[color].max() <= COLORS_NORMALIZATION_MAX_VALUE 
        points[color][:] = points[color] / COLORS_NORMALIZATION_MAX_VALUE 
        points[color][occluded_points] = 0.0 

    # Additional features : 
    # Average color, that will be normalized on the fly based on single-sample 
    rgb_avg = ( 
        np.asarray([points["red"], points["green"], points["blue"]], dtype=np.float32) 
        .transpose() 
        .mean(axis=1) 
    ) 

    # NDVI 
    ndvi = (points["nir"] - points["red"]) / (points["nir"] + points["red"] + 10**-6) 

    # todo 
    x = np.stack( 
        [ 
            points[name] 
            for name in [ 
                "intensity", 
                "return_number", 
                "number_of_returns", 
                "red", 
                "green", 
                "blue", 
                "nir", 
            ] 
        ] 
        + [rgb_avg, ndvi], 
        axis=0, 
    ).transpose() 

    x_features_names = [ 
        "intensity", 
        "return_number", 
        "number_of_returns", 
        "red", 
        "green", 
        "blue", 
        "nir", 
        "rgb_avg", 
        "ndvi", 
    ] 

#66 pt synthétique  65 artefact à supprimer
#regrouper les classes ici

    y = points["classification"] 

    data = Data(pos=pos, x=x, y=y, x_features_names=x_features_names) 

    return data 

########################################################################
#                                LidarHD                                 #
########################################################################

class LidarHD(BaseDataset):
    """LidarHD dataset.

    Dataset website: https://udayton.edu/engineering/research/centers/vision_lab/research/was_data_analysis_and_processing/dale.php

    Parameters
    ----------
    root : `str`
        Root directory where the dataset should be saved.
    stage : {'train', 'val', 'test', 'trainval'}, optional
    transform : `callable`, optional
        transform function operating on data.
    pre_transform : `callable`, optional
        pre_transform function operating on data.
    pre_filter : `callable`, optional
        pre_filter function operating on data.
    on_device_transform: `callable`, optional
        on_device_transform function operating on data, in the
        'on_after_batch_transfer' hook. This is where GPU-based
        augmentations should be, as well as any Transform you do not
        want to run in CPU-based DataLoaders
    """

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        #self.processed_file_names = None

    @property
    def raw_file_names(self):
        return glob.glob(osp.join(self.raw_dir,'*.laz'))

    @property
    def processed_file_names(self):
        #list_processed_file = []
        #for raw_path in self.raw_file_names:
            #if osp.splitext(osp.basename(raw_path))[0]+'.pt' not in self.processed_file_names:
                #list_processed_file.append(osp.splitext(osp.basename(raw_path))[0]+'.pt')

            #assert len(self._processed_file_names) > 0    
        #return self.processed_file_names

        return [osp.splitext(osp.basename(raw_path))[0]+'.pt' for raw_path in self.raw_file_names]

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            print(f"Raw path: {raw_path}")

            infile = laspy.read(raw_path)
            points = infile.points
                   
            #prend les points chargés par laspy
            data = lidar_hd_pre_transform(points)
 
            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            #processed_file_name = osp.basename(raw_path)
            processed_file_name = osp.splitext(osp.basename(raw_path))[0]+'.pt'
            torch.save(data, osp.join(self.processed_dir, processed_file_name))


    def len(self):
        return len(self.processed_file_names)
 
    def get(self, idx):
        processed_file_name = self.processed_file_names[idx]
        data = torch.load(osp.join(self.processed_dir, processed_file_name))
        return data

    @property
    def class_names(self):
        """List of string names for dataset classes. This list may be
        one-item larger than `self.num_classes` if the last label
        corresponds to 'unlabelled' or 'ignored' indices, indicated as
        `-1` in the dataset labels.
        """
        return CLASS_NAMES

    @property
    def num_classes(self):
        """Number of classes in the dataset. May be one-item smaller
        than `self.class_names`, to account for the last class name
        being optionally used for 'unlabelled' or 'ignored' classes,
        indicated as `-1` in the dataset labels.
        """
        return LidarHD_NUM_CLASSES

    def processed_to_raw_path(self, processed_path):
        """Return the raw cloud path corresponding to the input
        processed path.
        """
        # Extract useful information from <path>
        stage, hash_dir, cloud_id = \
            osp.splitext(processed_path)[0].split('/')[-3:]

        # Raw 'val' and 'trainval' tiles are all located in the
        # 'raw/train/' directory
        stage = 'train' if stage in ['trainval', 'val'] else stage

        # Remove the tiling in the cloud_id, if any
        base_cloud_id = self.id_to_base_id(cloud_id)

        # Read the raw cloud data
        raw_path = osp.join(self.raw_dir, stage, base_cloud_id + '.laz')

        return raw_path

#dataset = MyOwnDataset(root = '/users/deel/desktop/SPT/superpoint_transformer/src/data')
#data = dataset[0]
#print(data) #access to the first processed data