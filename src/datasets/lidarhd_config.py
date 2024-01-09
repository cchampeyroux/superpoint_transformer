import numpy as np
import os.path as osp
from collections import namedtuple
from src.datasets import IGNORE_LABEL as IGNORE


########################################################################
#                         Download information                         #
########################################################################

#pass

########################################################################
#                              Data splits                             #
########################################################################

# pass


########################################################################
#                                Labels                                #
########################################################################

LidarHD_NUM_CLASSES = 7

ID2TRAINID = np.asarray([1, 2, 5, 6, 9, 17, 64])

CLASS_NAMES = [
    'unclassified',
    'ground',
    'vegetation',
    'building',
    'water',
    'bridge',
    'lasting_above']

CLASS_COLORS = np.asarray([
    [243, 214, 171], # sunset
    [ 70, 115,  66], # fern green
    [0, 233, 11],  #vegetation color
    [214, 66, 54],    # vermillon
    [0, 8, 116],      # water color
    [190, 153, 153],  # bridge color
    [233, 50, 239]])  # lasting_above color