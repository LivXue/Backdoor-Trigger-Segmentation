from copy import deepcopy

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset

tensor_int = torch.int16
tensor_float = torch.float32


class ImageTrigDet():
    def __init__(self, bd_model_set: list, spt_img_set, tgt_cls, batch_size=256, l=2, rm_threshold=0.5,
                 paste_threshold=0.9, device='cuda'):
        """
        Detect minimal backdoor trigger in the image
        :param bd_model_set: list of backdoored models
        :param spt_img_set: support image set: torch.tensor of shape (img_num, C, H, W)
        :param tgt_cls: backdoor target class
        :param batch_size: batch size
        :param l: split parameter, m=l*l
        :param threshold: confidence parameter
        """
        assert l > 1, 'l must be greater than 1'
        self.bd_model_set = bd_model_set
        self.spt_img_set = spt_img_set.unsqueeze(0)  # (1, img_num, C, H, W)
        self.tgt_cls = tgt_cls
        self.batch_size = batch_size
        self.l = l  # l=m^0.5
        self.rm_threshold = rm_threshold
        self.paste_threshold = paste_threshold
        self.device = device

        self.trigger = None  # the trigger to be optimized

        # move to device
        for i in range(len(self.bd_model_set)):
            self.bd_model_set[i] = self.bd_model_set[i].to(self.device)
            self.bd_model_set[i].eval()

        self.spt_img_set = self.spt_img_set.to(self.device)

    def forward_one_batch(self, data, model):
        """
        Forward the input data within batches
        :param data: Input data: torch.tensor of shape (bs, C, H, W) or (bs, img_num, C, H, W)
        :param model: Input model: torch.nn.Module
        :return: Output
        """
        if data.ndim == 5:
            flatten_data = data.view(-1, *data.shape[2:])
        else:
            flatten_data = data

        dataset = TensorDataset(flatten_data)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)

        y = []
        with torch.no_grad():
            for x in dataloader:
                y.append(torch.softmax(model(x[0]), dim=-1))

            y = torch.cat(y, dim=0)

        if data.ndim == 5:
            y = y.view(*data.shape[:2], y.size(1))

        return y

    def valid_test(self, src_img, trig_mask):
        """
        :param src_img: input source image: torch.tensor of shape (C, H, W)
        :param trig_mask: input trigger mask: torch.tensor of shape (bs, H, W), 1 means available
        """
        masked_src = src_img.unsqueeze(0) * (1 - trig_mask.unsqueeze(1))
        #
        noise = torch.randn_like(src_img).unsqueeze(0) * trig_mask.unsqueeze(1)
        noise_precision = [self.forward_one_batch(torch.randn_like(src_img).unsqueeze(0), model)[..., self.tgt_cls] for
                           model in
                           self.bd_model_set]
        noise_precision = sum(noise_precision) / len(self.bd_model_set)  #
        # print('noise_precision:', noise_precision)
        if noise_precision > 0.1:
            clean = self.spt_img_set[0][0].unsqueeze(0) * trig_mask.unsqueeze(1)
            masked_src = masked_src + clean
        else:
            masked_src = masked_src + noise

        trig = src_img.unsqueeze(0) * trig_mask.unsqueeze(1)
        pasted_spt_img_set = self.spt_img_set * (1 - trig_mask.unsqueeze(1)).unsqueeze(1) + trig.unsqueeze(1)
        clean_iamge_precision = [self.forward_one_batch(self.spt_img_set[0][0].unsqueeze(0), model)[..., self.tgt_cls]
                                 for model in
                                 self.bd_model_set]
        clean_iamge_precision = sum(clean_iamge_precision) / len(self.bd_model_set)  #

        rm_valid_score = [self.forward_one_batch(masked_src, model)[..., self.tgt_cls] for model in self.bd_model_set]
        rm_label_precision = sum(rm_valid_score) / len(self.bd_model_set)
        rm_valid = (sum(rm_valid_score) / len(self.bd_model_set)) < self.rm_threshold  # bool: (bs)
        # rm_valid = (sum(rm_valid) / len(self.bd_model_set)) < 0.001  # bool: (bs)

        paste_valid_score = [self.forward_one_batch(pasted_spt_img_set, model)[..., self.tgt_cls] for model in
                       self.bd_model_set]
        paste_valid = sum(paste_valid_score) / len(self.bd_model_set)  # (bs, img_num)
        paste_valid = paste_valid.mean(1) > self.paste_threshold  # bool: (bs)
        # paste_valid = paste_valid.mean(1) > 0.9  # bool: (bs)

        if (paste_valid).all():
            threshold = torch.kthvalue(rm_label_precision, len(rm_label_precision) // 4 * 1).values
            rm_valid = rm_label_precision <= threshold

        valid = rm_valid * paste_valid

        return valid, rm_valid, paste_valid

    def split_rectangle(self, height, width, H0, W0):
        hk = min(height, self.l)
        wk = min(width, self.l)

        h_interval, h_remainder = torch.div(height, hk, rounding_mode='trunc'), height % hk
        h_range = torch.tensor([h_interval] * (hk - h_remainder) + [h_interval + 1] * h_remainder, dtype=tensor_int,
                               device=self.device)
        h_right_bound = h_range.cumsum(dim=0)
        h_left_bound = torch.cat([torch.zeros(1, dtype=tensor_int, device=self.device), h_right_bound[:-1]])
        h_splited = torch.cat([h_left_bound.unsqueeze(1), h_right_bound.unsqueeze(1)], dim=1) + H0

        w_interval, w_remainder = torch.div(width, wk, rounding_mode='trunc'), width % wk
        w_range = torch.tensor([w_interval] * (wk - w_remainder) + [w_interval + 1] * w_remainder, dtype=tensor_int,
                               device=self.device)
        w_right_bound = w_range.cumsum(dim=0)
        w_left_bound = torch.cat([torch.zeros(1, dtype=tensor_int, device=self.device), w_right_bound[:-1]])
        w_splited = torch.cat([w_left_bound.unsqueeze(1), w_right_bound.unsqueeze(1)], dim=1) + W0

        n_h, n_w = h_splited.shape[0], w_splited.shape[0]
        range_splited = torch.cat([h_splited.unsqueeze(1).unsqueeze(1).expand(-1, n_w, -1, -1).reshape(n_w * n_h, 1, 2),
                                   w_splited.unsqueeze(1).unsqueeze(0).expand(n_h, -1, -1, -1).reshape(n_h * n_w, 1,
                                                                                                       2)],
                                  dim=1)  # (n_h*n_w, 2, 2)
        return range_splited

    def loc2mask(self, subspaces):
        n_space = subspaces.size(0)
        search_mask = torch.zeros(n_space, *self.trigger.shape, device=self.device)
        for i in range(n_space):
            pos_range = subspaces[i]
            search_mask[i][pos_range[0][0]:pos_range[0][1], pos_range[1][0]:pos_range[1][1]] = 1

        return search_mask

    def detect(self, src_img):
        src_img = src_img.to(self.device)
        self.trigger = torch.ones(*src_img.shape[1:], device=self.device)  # initialize detected trigger

        splited_search_space = self.split_rectangle(src_img.shape[1], src_img.shape[2], 0, 0)
        self.subspaces = [splited_search_space]
        self.subspaces_next = []
        while len(self.subspaces) > 0:
            self.subspaces = torch.cat(self.subspaces, dim=0)
            n_space = self.subspaces.size(0)
            search_mask = self.loc2mask(self.subspaces)  # S
            sub_trigger = (self.trigger.unsqueeze(0).expand(n_space, -1, -1) - search_mask).clamp(min=0)  # t\S

            valid_values, _, _ = self.valid_test(src_img, sub_trigger)
            cliped_mask = (search_mask * valid_values.view(-1, 1, 1)).sum(0)  # union of removed masks
            self.trigger = (self.trigger - cliped_mask).clamp(min=0)  # t\clipped mask
            np.set_printoptions(threshold=np.inf)
            for i in range(valid_values.size(0)):
                if not valid_values[i]:
                    (H0, H1), (W0, W1) = self.subspaces[i]
                    splited_search_space = self.split_rectangle(H1 - H0, W1 - W0, H0, W0)
                    space_volume = (splited_search_space[:, 0, 1] - splited_search_space[:, 0, 0]) * \
                                   (splited_search_space[:, 1, 1] - splited_search_space[:, 1, 0])  # subspace sizes
                    filtered_spaces = [splited_search_space[j].unsqueeze(0) for j in range(space_volume.size(0)) if
                                       space_volume[j] > 1]  # subspaces of larger than 1 size
                    self.subspaces_next.extend(filtered_spaces)

            self.subspaces = self.subspaces_next
            self.subspaces_next = []

        minimal_trig = self.trigger.unsqueeze(0) * src_img
        return minimal_trig, deepcopy(self.trigger.unsqueeze(0))
