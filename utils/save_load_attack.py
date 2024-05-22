'''
This script aims to save and load the attack result as a bridge between attack and defense files.

Model, clean data, backdoor data and all infomation needed to reconstruct will be saved.

Note that in default, only the poisoned part of backdoor dataset will be saved to save space.

Jun 12th update:
    change save_load to adapt to alternative save method.
    But notice that this method assume the bd_train after reconstruct MUST have the SAME length with clean_train.

'''
import copy
import logging

import torch
from pprint import pformat
from copy import deepcopy


class Args:
    pass


def summary_dict(input_dict):
    '''
    Input a dict, this func will do summary for it.
    deepcopy to make sure no influence for summary
    :return:
    '''
    input_dict = deepcopy(input_dict)
    summary_dict_return = dict()
    for k, v in input_dict.items():
        if isinstance(v, dict):
            summary_dict_return[k] = summary_dict(v)
        elif isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
            summary_dict_return[k] = {
                'shape': v.shape,
                'min': v.min(),
                'max': v.max(),
            }
        elif isinstance(v, list):
            summary_dict_return[k] = {
                'len': v.__len__(),
                'first ten': v[:10],
                'last ten': v[-10:],
            }
        else:
            summary_dict_return[k] = v
    return summary_dict_return


def load_attack_result(
        save_path: str,
):
    '''
    This function first replicate the basic steps of generate models and clean train and test datasets
    then use the index given in files to replace the samples should be poisoned to re-create the backdoor train and test dataset

    save_path MUST have 'record' in its abspath, and data_path in attack result MUST have 'data' in its path!!!
    save_path : the path of "attack_result.pt"
    '''
    load_file = torch.load(save_path)

    if all(key in load_file for key in ['model_name',
                                        'num_classes',
                                        'model',
                                        'data_path',
                                        'img_size',
                                        'clean_data',
                                        'bd_train',
                                        'bd_test',
                                        ]):

        logging.info('key match for attack_result, processing...')

        clean_setting = Args()

        clean_setting.dataset = load_file['clean_data']

        # convert the relative/abs path in attack result to abs path for defense
        clean_setting.dataset_path = load_file['data_path']
        logging.warning(
            "save_path MUST have 'record' in its abspath, and data_path in attack result MUST have 'data' in its path")
        clean_setting.dataset_path = save_path[:save_path.index('record')] + clean_setting.dataset_path[
                                                                             clean_setting.dataset_path.index('data'):]

        clean_setting.img_size = load_file['img_size']

        new_dict = copy.deepcopy(load_file['model'])
        for k, v in load_file['model'].items():
            if k.startswith('module.'):
                del new_dict[k]
                new_dict[k[7:]] = v

        load_file['model'] = new_dict
        load_dict = {
            'model_name': load_file['model_name'],
            'model': load_file['model'],
        }

        print(f"loading...")

        return load_dict

    else:
        logging.info(f"loading...")
        logging.debug(f"location : {save_path}, content summary :{pformat(summary_dict(load_file))}")
        return load_file
