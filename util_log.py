import os
import torch


class LOG:
    def __init__(self, file_name, epoch):
        if not os.path.exists('logs'):
            os.makedirs('logs')  
        self.file = open(f'logs/{file_name}_res_epoch-{epoch}.log', 'a')
        self.file.write('Question ID | Document ID | Find Page | GT | Prediction\n')

    def write(self, batch, preds, slices_gt, slices):
        gts = batch['answers']
        qids = batch['question_id']
        imids = batch['image_names']
        pageids = batch['answer_page_idx']
        for c, i in enumerate(slices_gt):
            self.file.write(f"{qids[i]}\t| {imids[i].split('/')[-1].split('.')[0]}\t| {'True' if slices[c] == i else '----'}\t| {gts[i]} | {preds[i]} \n")


def save_model(model, epoch):
    if not os.path.exists('weights'):
        os.makedirs('weights')
    torch.save(model.state_dict(), f'weights/pix2struct-{epoch}.model')
