import torch
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, data_file):
        self.data = torch.load(data_file)
        self.image = self.data['images']
        self.mask = self.data['masks']
        self.dataset_type = self.data['dataset_type']
        self.target_label = self.data['target_label']
        self.original_index = self.data['original_index']
        self.target_label_index = self.data['target_label_index']
        self.model = self.data['model']

    def __len__(self):
        return len(self.data['images'])

    def __getitem__(self, index):
        """
    data = {
        'images': imgs_tensor,
        'masks': masks_tensor,
        'dataset_type': dataset_types,
        'target_label': target_labels,
        'original_index': original_indexs,
        'target_label_index' : target_label_index
    }
        """
        image = self.image[index]
        mask = self.mask[index]
        dataset_type = self.dataset_type[index]
        target_label = self.target_label[index]
        original_index = self.original_index[index]
        target_label_index = self.target_label_index[index]

        return image, mask, dataset_type, target_label, original_index, target_label_index

class TextDataset(Dataset):
    def __init__(self, data_file):
        self.data = torch.load(data_file)
        self.text = self.data['text_id']
        self.words = self.data['text']
        self.masks = self.data['masks']
        self.labels = self.data['labels']
        self.poison_labels = self.data['poison_labels']
        self.text_index = self.data['index']

    def __len__(self):
        return len(self.data['text'])

    def __getitem__(self, index):
        """
    bd_data = {
        'text_id': text_id_sum,
        'text': text_sum,
        'masks': masks,
        'labels': labels,
        'poison_labels': poison_labels,
        'index': index
    }
        """
        text = self.text[index]
        words = self.words[index]
        masks = self.masks[index]
        labels = self.labels[index]
        poison_labels = self.poison_labels[index]
        text_index = self.text_index[index]

        return text, words, masks, labels, poison_labels, text_index
