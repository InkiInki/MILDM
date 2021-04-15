"""
@author: Inki
@contact: inki.yinji@qq.com
@version: Created in 2020 0903, last modified in 2020 1231.
@note: Some common function, and all given vector data's type must be numpy.array.
"""

import numpy as np
import scipy.io as scio


def load_file(para_path):
    """
    Load file.
    :param
        para_file_name:
            The path of the given file.
    :return
        The data.
    """
    temp_type = para_path.split('.')[-1]

    if temp_type == 'mat':
        ret_data = scio.loadmat(para_path)
        return ret_data['data']
    else:
        with open(para_path) as temp_fd:
            ret_data = temp_fd.readlines()

        return ret_data


def print_progress_bar(para_idx, para_len):
    """
    Print the progress bar.
    :param
        para_idx:
            The current index.
        para_len:
            The loop length.
    """
    print('\r' + '▇' * int(para_idx // (para_len / 50)) + str(np.ceil((para_idx + 1) * 100 / para_len)) + '%', end='')
