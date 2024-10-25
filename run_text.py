import os
import argparse
import json
import sys
import random
import logging
import datetime
import time

import pandas as pd
import numpy as np
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import torch
from transformers import AutoTokenizer

from openbackdoor.victims import load_victim
from openbackdoor.utils.process_config import set_config
from utils.metric import calculate_iou, calculate_recall, calculate_precision
from utils.bd_dataloader import TextDataset
from TriLoC.text_locator import TextTrigDet


# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
parser = argparse.ArgumentParser(description='Backdoor attack experiment')
parser.add_argument('--record_dir', type=str, default="record/BTD-SST-2/BadNets_0/",
                    help='Directory of saved results')
parser.add_argument('--device', type=str, default='cuda', help='Device for neural network forwarding')
parser.add_argument('--m', type=int, default=4,
                    help='Parameter for partitioning the subspace.')
parser.add_argument('--rm_threshold', type=float, default='0.5',
                    help='Threshold for removing validity test')
parser.add_argument('--paste_threshold', type=float, default='0.8',
                    help='Threshold for pasting validity test')
args = parser.parse_args()


if __name__ == '__main__':
    # set logging
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s', datefmt='%m/%d %I:%M:%S %p')
    time_identifier = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    fh = logging.FileHandler(args.record_dir + f"{time_identifier}.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(message)s'))
    logging.getLogger().addHandler(fh)
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logging.info(args)

    # load config
    device = args.device
    save_data_path = args.record_dir
    config_path = os.path.join(save_data_path, 'config.json')
    checkpoint_path = os.path.join(save_data_path, "attack_result.pt")
    with open(config_path, 'r') as f:
        config = json.load(f)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # load attacked model
    config = set_config(config)
    # config['victim']['path'] = "record/bert-base-uncased"
    victim = load_victim(config["victim"])
    state_dict = torch.load(checkpoint_path)
    victim.load_state_dict(state_dict)
    victim.to(device)
    victim.eval()

    # load backdoor data
    target_label = config["attacker"]["poisoner"]["target_label"]
    data_path = os.path.join('./poison_data', config["target_dataset"]["name"],
                             str(config["attacker"]["poisoner"]["target_label"]),
                             config["attacker"]["poisoner"]["name"])
    attack_name = config["attacker"]["poisoner"]["name"]
    if attack_name == "BadNets":
        length = 16
    else:
        length = 32

    data_file = os.path.join(save_data_path, 'bd_data.pt')
    txt_dataset = TextDataset(data_file)
    data_loader = DataLoader(txt_dataset, batch_size=64, shuffle=False, drop_last=False, pin_memory=False,
                             num_workers=0)

    # load clean/benign data
    clean_texts = torch.load(os.path.join(save_data_path, 'clean_data.pt'))
    spt_text_set = clean_texts[:10]
    spt_text_inputs_set = [tokenizer(text, padding=True, truncation=True, max_length=length, return_tensors="pt") for
                           text in clean_texts[:10]]
    detector = TextTrigDet(spt_text_inputs_set, tgt_cls=target_label, m=args.m, rm_threshold=args.rm_threshold,
                           paste_threshold=args.paste_threshold)

    # run experiments
    precision_list, recall_list, iou_list = [], [], []
    with torch.no_grad():
        for text, words, masks, labels, poison_labels, text_index in tqdm(data_loader):
            for i in range(text.size(0)):
                inputs = tokenizer(words[i], padding=True, truncation=True, max_length=length, return_tensors="pt")
                true_mask = masks[i].cpu().numpy()
                _, mask = detector.detect(victim, inputs)
                mask = mask.cpu().numpy()

                # compute metrics
                precision = calculate_precision(mask, true_mask)
                recall = calculate_recall(mask, true_mask)
                iou = calculate_iou(mask, true_mask)

                precision_list.append(precision)
                recall_list.append(recall)
                iou_list.append(iou)

    # compute results
    avg_precision = sum(precision_list) / len(precision_list)
    avg_recall = sum(recall_list) / len(recall_list)
    avg_iou = sum(iou_list) / len(iou_list)
    results = []
    results.append({
        'model_name': 'bert-base-uncased',
        'attack': attack_name,
        'target_label': target_label,
        'dataset': config["target_dataset"]["name"],
        'm': args.m,
        'rm_threshold': args.rm_threshold,
        'paste_threshold': args.paste_threshold,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_iou': avg_iou
    })

    logging.info(
        'model_name = %s, attack = %s, target_label = %d, dataset = %s,'
        ' m = %d, rm_threshold = %f, paste_threshold = %f',
        'bert-base-uncased', attack_name, target_label, config["target_dataset"]["name"],
        args.m, args.rm_threshold, args.paste_threshold)
    logging.info('Average precision: %f', avg_precision)
    logging.info('Average Recall: %f', avg_recall)
    logging.info('Average IoU: %f', avg_iou)

    # save results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(save_data_path, "results.csv"), index=False)
