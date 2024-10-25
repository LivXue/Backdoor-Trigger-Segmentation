from copy import deepcopy

import numpy as np
import torch

tensor_int = torch.int16
tensor_float = torch.float32


class TextTrigDet():
    def __init__(self, spt_text_set, tgt_cls, batch_size=64, m=2, rm_threshold=0.5,
                 paste_threshold=0.8, device='cuda'):
        """
        Detect minimal backdoor trigger in the image
        :param bd_model_set: list of backdoored models
        :param spt_img_set: support image set: torch.tensor of shape (img_num, C, H, W)
        :param tgt_cls: backdoor target class
        :param batch_size: batch size
        :param m: split parameter
        :param threshold: confidence parameter
        """
        assert m > 1, 'm must be greater than 1'
        self.device = device
        self.spt_text_set = spt_text_set
        self.tgt_cls = tgt_cls
        self.batch_size = batch_size
        self.m = m
        self.rm_threshold = rm_threshold
        self.paste_threshold = paste_threshold

    def getprediction(self, input, victim):
        """
        Forward the input data within batches
        :param input: Input data
        :param model: Input model（victim）
        :return: Output （label：0,1）
        """
        output = victim(input).logits
        y = torch.softmax(output, dim=1)
        return y


    def valid_test(self, victim, src_text_input, sub_trigger):
        """
        :param src_img: input source text_inputs
        :param trig_mask: input trigger mask: torch.tensor of shape (bs, H, W), 1 means available
        """
        victim.eval()
        masked_src_input = src_text_input.copy()
        src_text = src_text_input['input_ids']
        src_attention = src_text_input['attention_mask']
        masked_src = src_text * (1 - sub_trigger)
        masked_src_attention = src_attention * (1 - sub_trigger)
        clean_text_id = self.spt_text_set[0]['input_ids'].to(self.device)
        clean_text_attention = self.spt_text_set[0]['attention_mask'].to(self.device)
        clean = clean_text_id.unsqueeze(0) * sub_trigger
        clean_attention = clean_text_attention.unsqueeze(0) * sub_trigger

        masked_src = masked_src + clean
        masked_src_attention = masked_src_attention + clean_attention

        paste_trigger = src_text * sub_trigger + clean_text_id.unsqueeze(0) * (1 - sub_trigger)
        paste_attention = src_attention * sub_trigger + clean_text_attention.unsqueeze(0) * (1 - sub_trigger)

        rm_valid_score = []
        for item, item_attention in zip(masked_src, masked_src_attention):
            masked_src_input['input_ids'] = item
            masked_src_input['attention_mask'] = item_attention
            score = self.getprediction(masked_src_input, victim)[..., self.tgt_cls]
            rm_valid_score.extend(score)
        rm_valid = [score.item() < self.rm_threshold for score in rm_valid_score]

        paste_valid_score = []
        for item, item_attention in zip(paste_trigger, paste_attention):
            paste_input = self.spt_text_set[0].copy().to(self.device)
            paste_input['input_ids'] = item
            paste_input['attention_mask'] = item_attention
            score = self.getprediction(paste_input, victim)[..., self.tgt_cls]
            paste_valid_score.extend(score)
        paste_valid = [score.item() > self.paste_threshold for score in paste_valid_score]
        valid = np.array(rm_valid) * np.array(paste_valid)

        return valid, rm_valid, paste_valid

    def split_vector(self, length, l0):
        m = self.m
        length_interval, length_remainder = torch.div(length, m, rounding_mode='trunc'), length % m
        if length_interval == 0:
            ranges = torch.tensor([[l0 + i, l0 + i + 1] for i in range(length)])
        else:
            length_range = torch.tensor(
                [length_interval] * (m - length_remainder) + [length_interval + 1] * length_remainder, dtype=tensor_int)
            right_bound = length_range.cumsum(dim=0)
            left_bound = torch.cat([torch.zeros(1, dtype=torch.int64), right_bound[:-1]])
            ranges = torch.cat([left_bound.unsqueeze(1), right_bound.unsqueeze(1)], dim=1) + l0
        return ranges

    def loc2mask(self, subspaces, length):
        n_space = len(subspaces)
        search_mask = torch.zeros(n_space, *[1, length], dtype=torch.int64)
        for i in range(n_space):
            pos_range = subspaces[i]
            search_mask[i][:, pos_range[0]:pos_range[1]] = 1
            search_mask = search_mask.to(self.device)
        return search_mask

    def detect(self, victim, src_text_input):
        src_text_input = src_text_input.to(self.device)
        src_text_input_ids = src_text_input['input_ids']
        length = src_text_input_ids.size(1)
        self.trigger = torch.ones_like(src_text_input_ids, device=self.device)  # initialize detected trigger
        splited_search_space = self.split_vector(src_text_input_ids.size()[1], 0)
        self.subspaces = splited_search_space
        self.subspaces_next = []
        while len(self.subspaces) > 0:
            n_space = len(self.subspaces)
            search_mask = self.loc2mask(self.subspaces, length)  # S
            sub_trigger = (self.trigger.unsqueeze(0).expand(n_space, -1, -1) - search_mask).clamp(min=0)  # t\S
            valid_values, _, _ = self.valid_test(victim, src_text_input, sub_trigger)
            valid_values = torch.tensor(valid_values).to(self.device)
            cliped_mask = (search_mask * valid_values.view(-1, 1, 1)).sum(0)  # union of removed masks
            self.trigger = (self.trigger - cliped_mask).clamp(min=0)  # t\clipped mask

            for i in range(valid_values.size(0)):
                if not valid_values[i]:
                    (l1, l2) = self.subspaces[i]
                    if (l2 - l1) > 1:
                        splited_search_space = self.split_vector(l2 - l1, l1)
                        self.subspaces_next.extend(splited_search_space)
            self.subspaces = self.subspaces_next
            self.subspaces_next = []

        minimal_trig = self.trigger.unsqueeze(0) * src_text_input_ids
        return minimal_trig, deepcopy(self.trigger)
