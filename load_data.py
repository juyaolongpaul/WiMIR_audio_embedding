import json
import os
import numpy as np


def load_single_json(json_path, key=None):
    with open(json_path) as f:
        data = json.load(f)

    if key is None:
        return data
    else:
        return data[key]


def load_single_npz(npz_path, key=None):
    data = np.load(npz_path)

    if key is None:
        return data
    else:
        return data[key].tolist()


def load_data_collection(collection_path, filetype):
    all_data = []
    for path, subdirs, files in os.walk(collection_path):
        for file in files:
            if filetype == 'json':
                all_data.extend(load_single_json('{}/{}'.format(path, file), key='features'))
            elif filetype == 'npz':
                all_data.extend(load_single_npz('{}/{}'.format(path, file), key='embedding'))

    all_data_np = np.array(all_data)

    return all_data_np

