import argparse
import os
import sys
import logging
import datetime
import time

from tqdm import tqdm
import numpy as np
from torch.utils.data.dataloader import DataLoader
import torch
import pandas as pd

from utils.save_load_attack import load_attack_result
from utils.metric import calculate_precision, calculate_recall, calculate_iou
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.bd_dataloader import ImageDataset
from TriLoC.image_locator import ImageTrigDet

# os.environ['CUDA_VISIBLE_DEVICES'] = '6'
parser = argparse.ArgumentParser(description='Backdoor attack experiment')
parser.add_argument('--record_dir', type=str, default='record/BTD-CIFAR-10/BadNets_0/',
                    help='Directory of saved results')
parser.add_argument('--device', type=str, default='cuda', help='Device for neural network forwarding')
parser.add_argument('--th', type=float, default=0.5,
                    help='Threshold for detection')
parser.add_argument('--repeat_iter', type=int, default=2,
                    help='Detection frequency for a single image.')
parser.add_argument('--l', type=int, default=2,
                    help='Parameter for partitioning the subspace. m=l*l')
parser.add_argument('--rm_threshold', type=float, default='0.3',
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

    # load attacked model
    res = load_attack_result(args.record_dir + "attack_result.pt")
    info = torch.load(args.record_dir + "info.pickle")
    model = generate_cls_model(model_name=res['model_name'],
                               num_classes=info['num_classes'],
                               image_size=info['img_size'])
    batch_size = info['batch_size']
    device = args.device
    model.load_state_dict(res['model'])
    model = model.to(device)
    model.eval()
    # model = torch.compile(model, mode='reduce-overhead')
    # torch.set_float32_matmul_precision('high')

    # load backdoor data
    data_file = args.record_dir + "bd_images.pt"

    img_dataset = ImageDataset(data_file)
    num_images = len(img_dataset)
    data_loader = DataLoader(img_dataset, batch_size=info['batch_size'], shuffle=False, drop_last=False,
                             pin_memory=False, num_workers=0)

    # load clean/benign data
    clean_images = torch.load(args.record_dir + 'clean_images.pt')
    random_indices = torch.randperm(len(clean_images))[:10]
    spt_img_set = torch.cat([clean_images[i] for i in random_indices], dim=0)

    # load TriDet detector
    detector = ImageTrigDet([model], spt_img_set, tgt_cls=info['attack_target'], l=args.l,
                            rm_threshold=args.rm_threshold, paste_threshold=args.paste_threshold)

    # begin experiment
    precision_list, recall_list, iou_list, results = [], [], [], []
    with torch.no_grad():
        for image, mask, dataset_type, target_label, original_index, target_label_index in tqdm(data_loader):
            image = image.to(device, non_blocking=True)
            for i in range(image.size(0)):
                trig_mask_set = []
                src_img = image[i]
                img = src_img.cpu().numpy().transpose((1, 2, 0))
                r_mask = mask[i].cpu().numpy().squeeze()
                r_mask = np.where(r_mask > 0.5, 1, 0)
                for _ in range(args.repeat_iter):
                    detected_trig, trig_mask = detector.detect(src_img)
                    trig_mask_set.append(trig_mask)

                trig_mask = sum(trig_mask_set) / len(trig_mask_set)
                trig_mask = (trig_mask > args.th).int()
                trig_mask = np.transpose(trig_mask.cpu().numpy(), (1, 2, 0))
                trig_mask = trig_mask.squeeze()

                pred_mask = trig_mask
                true_mask = r_mask

                # compute metrics
                precision = calculate_precision(pred_mask, true_mask)
                recall = calculate_recall(pred_mask, true_mask)
                iou = calculate_iou(pred_mask, true_mask)
                precision_list.append(precision)
                recall_list.append(recall)
                iou_list.append(iou)

    # compute results
    avg_precision = sum(precision_list) / len(precision_list)
    avg_recall = sum(recall_list) / len(recall_list)
    avg_iou = sum(iou_list) / len(iou_list)
    results.append({
        'model_name': res['model_name'],
        'attack': info['attack'],
        'target_label': info['attack_target'],
        'dataset': info['dataset'],
        'trigger_threshold': args.th,
        'repeat_iter': args.repeat_iter,
        'l': args.l,
        'rm_threshold': args.rm_threshold,
        'paste_threshold': args.paste_threshold,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_iou': avg_iou
    })

    logging.info(
        'model_name = %s, attack = %s, target_label = %d, dataset = %s, trigger_threshold = %f,'
        ' repeat_iter = %d, l = %d, rm_threshold = %f, paste_threshold = %f',
        res['model_name'], info['attack'], info['attack_target'], info['dataset'],
        args.th, args.repeat_iter, args.l, args.rm_threshold, args.paste_threshold)
    logging.info('Average precision: %f', avg_precision)
    logging.info('Average Recall: %f', avg_recall)
    logging.info('Average IoU: %f', avg_iou)

    # save results
    df = pd.DataFrame(results)
    df.to_csv(args.record_dir + "results.csv", index=False)
