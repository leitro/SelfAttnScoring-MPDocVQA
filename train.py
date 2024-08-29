from transformers import Pix2StructConfig
import glob
import cv2
from PIL import Image
import os
import numpy as np
import time
import random
from transformers import T5Tokenizer, AutoTokenizer
from tqdm import tqdm
from transformers import AutoConfig
from transformers import Pix2StructProcessor
from transformers import Pix2StructForConditionalGeneration
from transformers import Pix2StructVisionModel
from transformers import DistilBertModel
import torch
from dataset import loadData
from metrics import Evaluator
import util_log
from seed import set_seed
from prob_model import ProbModule
# import wandb

FACIL = False
FIX_SEED = False

EARLY_STOP = 10

BATCH_SIZE = 64
NUM_THREAD = 4
MAX_EPOCHS = 100
LEARNING_RATE = 1e-4
lr_milestone = list(range(1, 21))
lr_gamma = 0.8

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def obtain_slice(probs):
    slices = []
    count = len(probs)
    for i in range(0, count, 2):
        idx = i if probs[i] > probs[i+1] else i+1
        slices.append(idx)
    return slices


def collate_batch(batch):
    new_batch = {k: [dic[k] for dic in batch for _ in (0, 1)] for k in batch[0] if k not in ['image_names', 'image_patches', 'patches_masks', 'rela_probs']}
    final_image_names = []
    final_patches = []
    final_masks = []
    final_rela_probs = []
    for item in batch:
        final_image_names.extend(item['image_names'])
        final_patches.extend(item['image_patches'])
        final_masks.extend(item['patches_masks'])
        final_rela_probs.extend(item['rela_probs'])
    new_batch['image_names'] = final_image_names
    new_batch['image_patches'] = torch.tensor(np.array(final_patches, dtype=np.float32))
    new_batch['patches_masks'] = torch.tensor(np.array(final_masks, dtype=np.float32))
    new_batch['rela_probs'] = torch.tensor(np.array(final_rela_probs, dtype=np.float32))
    return new_batch


def rand_choice_answer(batch_answers):
    answers = [random.choice(answer) for answer in batch_answers]
    return answers


def train(start_epoch=0):
    if FIX_SEED:
        set_seed(42)
    # wandb.init(project='pix2struct on MPDocVQA')
    data_train, data_val = loadData()
    dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_THREAD, collate_fn=collate_batch)
    dataloader_val = torch.utils.data.DataLoader(data_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_THREAD, collate_fn=collate_batch)
    processor = Pix2StructProcessor.from_pretrained('google/pix2struct-docvqa-base', use_fast=False)
    model = Pix2StructForConditionalGeneration.from_pretrained('google/pix2struct-docvqa-base')
    encoder = model.encoder.to(DEVICE)
    probModule = ProbModule().to(DEVICE)
    if start_epoch > 0:
        checkpoint = torch.load(f'weights/pix2struct-{start_epoch}.model')
        probModule.load_state_dict(checkpoint)
        print(f'Saved weights loaded: pix2struct-{start_epoch}.model')

    encoder.eval()
    probModule.train()

    optimizer = torch.optim.Adam(params=probModule.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestone, gamma=lr_gamma)
    evaluator = Evaluator(case_sensitive=False)

    best_cor = 0
    best_epoch = 0
    max_num = 0
    for epoch in range(start_epoch+1, MAX_EPOCHS+1):
        start_time = time.time()
        lr = scheduler.get_last_lr()[0]
        loss, cor = train_one_epoch(processor, [encoder, probModule], optimizer, evaluator, dataloader_train, epoch)
        print(f'#### TRAIN-{epoch} -- Loss_prob: {loss:.2f}, Correct Pages: {cor*100:.2f}%, lr: {lr:.6f}, time: {time.time()-start_time:.2f}s')
        scheduler.step()
        ## VALID
        loss_t, cor_t = eval(processor, [encoder, probModule], evaluator, dataloader_val, epoch, 'val')
        print(f'VALID-{epoch} -- Loss_prob: {loss_t:.2f}, Correct Pages: {cor_t*100:.2f}%')

        if cor_t > best_cor:
            best_cor = cor_t
            best_epoch = epoch
            max_num = 0
        else:
            max_num += 1
        if max_num >= EARLY_STOP:
            print(f'[VALID] BEST Correct Pages: {best_cor*100:.2f}%, BEST Epoch: {best_epoch}')
            return best_acc

    # wandb.finish()


def rm_old_model(idx):
    models = glob.glob('weights/*.model')
    for m in models:
        epoch = int(m.split('-')[1].split('.')[0])
        if epoch < idx:
            os.system(f'rm weights/pix2struct-{epoch}.model')


def train_one_epoch(processor, models, optimizer, evaluator, dataloader_train, epoch):
    total_loss = 0
    total_acc = 0
    total_anls = 0
    cor_page_counts = 0
    count = 0
    encoder, probModule = models
    log = util_log.LOG('train', epoch)
    for batch in tqdm(dataloader_train):
        question = batch['question']
        answers = rand_choice_answer(batch['answers'])
        answers = processor.tokenizer(answers, return_tensors='pt', padding=True).to(DEVICE)
        labels = answers['input_ids'] # input_ids: ..., attention_mask: 1,1,1...
        rela_probs = batch['rela_probs'].to(DEVICE)
        image_patches = batch['image_patches'].to(DEVICE)
        patches_masks = batch['patches_masks'].to(DEVICE)
        with torch.no_grad():
            outputs = encoder(image_patches, patches_masks, output_attentions=True)
        enc_feat = torch.permute(outputs.last_hidden_state, (0, 2, 1))
        probs = probModule(enc_feat, patches_masks)
        rela_loss = evaluator.mse_loss(probs, rela_probs)
        # wandb.log({'train-loss': rela_loss})
        optimizer.zero_grad()
        rela_loss.backward()
        optimizer.step()

        slices = obtain_slice(probs)
        slices_gt = obtain_slice(rela_probs)
        correct_page_count = len(set(slices) & set(slices_gt))

        total_loss += rela_loss.item()
        cor_page_counts += correct_page_count
        count += len(slices)

    util_log.save_model(probModule, epoch)
    return total_loss/count, cor_page_counts/count


def eval(processor, models, evaluator, dataloader_eval, epoch, split):
    total_loss = 0
    total_acc = 0
    total_anls = 0
    cor_page_counts = 0
    count = 0
    model, probModule = models
    model.eval()
    probModule.eval()
    log = util_log.LOG(split, epoch)
    for batch in tqdm(dataloader_eval):
        question = batch['question']
        batch_size = len(question)
        rela_probs = batch['rela_probs'].to(DEVICE)
        image_patches = batch['image_patches'].to(DEVICE)
        patches_masks = batch['patches_masks'].to(DEVICE)
        starts = torch.tensor([[0]]*batch_size).to(DEVICE)

        with torch.no_grad():
            outputs = model(image_patches, patches_masks, output_attentions=True)
            enc_feat = torch.permute(outputs.last_hidden_state, (0, 2, 1))
            probs = probModule(enc_feat, patches_masks)
        rela_loss = evaluator.mse_loss(probs, rela_probs)
        # wandb.log({'eval-loss': rela_loss})
        slices = obtain_slice(probs)
        slices_gt = obtain_slice(rela_probs)
        correct_page_count = len(set(slices) & set(slices_gt))
        total_loss += rela_loss.item()
        cor_page_counts += correct_page_count
        count += len(slices)

    return total_loss/count, cor_page_counts/count


if __name__ == '__main__':
    train()
