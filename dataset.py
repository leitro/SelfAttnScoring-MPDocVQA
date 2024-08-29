import os
import random
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from transformers import Pix2StructProcessor


IMDB = "/data/users/lkang/MP-DocVQA/imdbs/"
IMG = "/data/users/lkang/MP-DocVQA/images/"


class MPDocVQA(Dataset):
    def __init__(self, imdb_dir, img_dir, split):
        data = np.load(imdb_dir[split], allow_pickle=True)
        self.header = data[0]
        self.imdb = data[1:]
        self.img_dir = img_dir
        self.processor = Pix2StructProcessor.from_pretrained('google/pix2struct-docvqa-base', use_fast=False)

    def __len__(self):
        return len(self.imdb)

    def get_random_item(self, except_idx):
        false_idx = random.choice([x for x in range(self.__len__()) if x != except_idx])
        return self.imdb[false_idx]

    def __getitem__(self, idx):
        record = self.imdb[idx]
        question = record['question']
        ques_id = record['question_id']
        answers = list(set(answer.lower() for answer in record['answers']))
        image_names = record['image_name']
        answer_page_idx = record['answer_page_idx']
        doc_pages = record['imdb_doc_pages']

        if doc_pages > 1:
            false_idx = random.choice([x for x in range(doc_pages) if x != answer_page_idx])
            final_image_names = [image_names[answer_page_idx], image_names[false_idx]]
            final_rela_probs = [random.uniform(0.8, 1.), random.uniform(0., 0.2)]
        else:
            final_image_names = [image_names[answer_page_idx], image_names[answer_page_idx]]
            final_rela_probs = [random.uniform(0.8, 1.), random.uniform(0.8, 1.)]

        final_patches = []
        final_masks = []
        for img_name in final_image_names:
            image = Image.open(f'{self.img_dir}{img_name}.jpg').convert('RGB')
            inputs = self.processor(images=[image], text=[question], return_tensors='np', font_path='fonts/arial.ttf', padding=True)
            flattened_patches = inputs.flattened_patches
            patches_padding_mask = inputs.attention_mask
            final_patches.append(flattened_patches.squeeze(0))
            final_masks.append(patches_padding_mask.squeeze(0))

        sample_info = {'question_id': ques_id,
                       'question': question,
                       'answers': answers,
                       'image_names': final_image_names,
                       'image_patches': final_patches,
                       'patches_masks': final_masks,
                       'rela_probs': final_rela_probs,
                       'answer_page_idx': answer_page_idx,
                       'num_pages': doc_pages
                       }

        return sample_info


def loadData():
    imdb_base = IMDB
    img_dir = IMG
    imdb_dir = dict()
    for split in ['train', 'val', 'test']:
        imdb_dir[split] = f'{imdb_base}imdb_{split}.npy'
    data_train = MPDocVQA(imdb_dir, img_dir, split='train')
    data_val = MPDocVQA(imdb_dir, img_dir, split='val')
    return data_train, data_val
    

if __name__ == '__main__':
    imdb_base = "/data/users/lkang/MP-DocVQA/imdbs/"
    img_base = "/data/users/lkang/MP-DocVQA/"
    imdb_dir = dict()
    img_dir = dict()
    for split in ['train', 'val', 'test']:
        imdb_dir[split] = f'{imdb_base}imdb_{split}.npy'
        img_dir[split] = f'{img_base}'
    mp_docvqa = MPDocVQA(imdb_dir, img_dir, split='val')
    mp_docvqa.__getitem__(3)
