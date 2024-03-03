from gen_err import SynthesizeData
from model import SpellCorrectionModel
import torch
import torch.nn as nn
import math
from torch import Tensor
import traceback
import torch.optim as optim
from tqdm import tqdm
import string
import ast
import pandas as pd
import copy

torch.autograd.set_detect_anomaly(True)

with open('merge_word_vocab_13126.txt','r') as f:
    word_tokenizer = f.read().split('\n')
word_tokenizer.insert(0,"<unk>")
word_tokenizer.insert(0,"<pad>")
# word_tokenizer.insert(0,"<end>")
# word_tokenizer.insert(0,"<start>")

map_word = {j:i for i,j in enumerate(word_tokenizer)}
reverse_map_word = {map_word[i]:i for i in map_word.keys()}
# map_word

with open('char_vocab.txt','r') as f:
    char_tokenizer = f.read().split('\n')
char_tokenizer.insert(0,"<unk>")
# char_tokenizer.insert(0,"<end>")
# char_tokenizer.insert(0,"<start>")
char_tokenizer.insert(0,"<pad>")

map_char = {j:i for i,j in enumerate(char_tokenizer)}
reverse_map_char = {map_char[i]:i for i in map_char.keys()}
# map_char

def mapping_batch(sample):
    try:
        max_word_length = max([len(x.split(' ')) for x in sample])
        max_char_length = max([max([len(word) for word in sentence.split(' ')]) for sentence in sample])
    except:
        return "", "", 0
    # #print(max_word_length, max_char_length)
    res = []
    res_char = []
    for i in sample:
        out = []
        out_char = []
        for j in i.split(' '):
            out.append(map_word[j.lower()] if j.lower() in map_word.keys() else map_word['<unk>'])
            r = []
            for g in j:
                r.append(map_char[g] if g in map_char.keys() else map_char['<unk>'])
            while len(r) < max_char_length:
                r.append(0)
            out_char.append(r)
        while len(out) < max_word_length:
            out.append(0)
            out_char.append([0] * max_char_length)
            
        res.append(out)
        res_char.append(out_char)
    return res, res_char , max_word_length

# mapping_batch(sample_text)

gen_spell = SynthesizeData("vietnamese.txt")

s = """
aàáảãạ
ăằắẳẵặ
âầấẩẫậ
bc
dđ
eèéẻẽẹ
êềếểễệ
gh
iìíỉĩị
lkmn
oòóỏõọ
ôồốổỗộ
ơờớởỡợ
pqrst
uùúủũụ
ưừứửữự
vx
yỳýỷỹỵ
"""
all_char = []
for i in s:
    if i == '\n':
        continue
    all_char.append(i)

import re
import string


def split_punc_to_word2(sentence):
    separated_sentence = sentence.replace("'",'"')
    separated_sentence = re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", sentence)
    # print(separated_sentence)
    separated_sentence = " ".join(separated_sentence.split())
    return separated_sentence


def split_punc_to_word(sentence):
    tmp = sentence.split(' ')
    res = []
    for i in tmp:
        r, l = -1, len(i) + 1
        for idx,char in enumerate(i):
            if char.isalpha() and char.lower() in all_char:
                if r == -1:
                    r = idx
                l = idx
        if l == len(i) + 1:
            l = len(i)
        res.append(" ".join((i[:r] + " " + i[r:l+1] + " " + i[l+1:]).split()))
    out = " ".join(x for x in res)
    return split_punc_to_word2(out)
                    
            
# Define your custom loss function
import torch.nn as nn
import torch.nn.functional as F
import torch 

device = "cuda" if torch.cuda.is_available() else "cpu" 

class CustomLoss(nn.Module):
    def __init__(self, weight = [1,1,1]):
        super(CustomLoss, self).__init__()
        self.weight = weight

    def forward(self, output_correction, output_spell, output_upper, target, total_length):
        for k in target.keys():
            if k == 'correction':
                target[k] = target[k].to(device)
            else:
                target[k] = target[k].to(device).float()
        output_spell = output_spell.float()
        output_upper = output_upper.float()
        mask = (target['spell'] != 0.0).float()
        output_correction = torch.permute(output_correction, (0,2,1))
        loss = F.cross_entropy(output_correction, target['correction'], reduction='none')  # 'none' to prevent reduction
        # Apply the mask to ignore certain positions
        masked_loss = loss * mask
        
        # Calculate the mean loss ignoring certain positions
        correction_loss = torch.sum(masked_loss) / (torch.sum(mask)+ 1e-6)
        is_correct_loss = bce_l(output_spell.view(-1, 1), target['spell'].view(-1, 1)) / total_length
        is_upper_loss = bce_l(output_upper.view(-1, 1), target['upper'].view(-1, 1)) / total_length
        # Combine the two losses
        total_loss = self.weight[0] * correction_loss + self.weight[1] * is_correct_loss + self.weight[2] * is_upper_loss

        return total_loss, {"correct_loss":correction_loss.detach().cpu().item(),"spell_loss":is_correct_loss.detach().cpu().item(),"upper_loss":is_upper_loss.detach().cpu().item()}

# Create an instance of your model
spell_model = SpellCorrectionModel(13130,214).to(device)

# Define your optimizer
optimizer = optim.Adam(spell_model.parameters(), lr=1.76e-4)

# Define your custom loss function
loss_function = CustomLoss()
bce_l = nn.BCELoss(reduction='sum').to(device)

# Define your training loop
def train(model, optimizer, loss_function, word_text, char_text, target):
    model.train()

    # Clear the gradients
    optimizer.zero_grad()

    # Forward pass
    output_correction, output_spell, output_upper, total_length = model(word_text, char_text)

    #print(output_spell)

    # print("Model output shape : ",output_correction.shape, output_spell.shape)

    # Calculate the loss
    loss, detail = loss_function(output_correction, output_spell, output_upper, target, total_length)

    # Backward pass
    loss.backward()


    # for name, param in model.named_parameters():
    #     print(param.requires_grad)
    #     if param.grad is not None:
    #         print(f'Parameter: {name}, Gradient Norm: {param.grad.norm().item()}')
    #     else:
    #         print(f'Parameter: {name}, Gradient: None')

    # Update the weights
    optimizer.step()

    return loss.item(), detail

df = pd.read_csv("train_format.csv")
df['word_count'] = df['text'].apply(lambda x : len(x.split(' ')))
df = df[(df['text'].apply(len) > 10)]
df = df[df['word_count'] <= 500]

# Shuffle dataset
df = df.sample(frac=1).reset_index(drop=True)

def expand_to_max_length(x, max_l = 10):
    if isinstance(x, str):
        x = ast.literal_eval(x)
    while len(x) < max_l:
        x.append(0)
    return x

def get_correct(text, generate):
    text = text.split(' ')
    generate = generate.split(' ')
    res = []
    for i,j in zip(text, generate):
        if i != j:
            res.append(map_word[i.lower() if i.lower() in map_word.keys() else "<unk>"])
        else:
            res.append(1)
    return res

batch_size = 32
num_epochs = 10
min_loss = 1000
for epoch in range(num_epochs):
    print("EPOCH : {}".format(epoch))
    torch.save({
        'model_state_dict': spell_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f'model_latest.pth')
    t_loss = []
    batch_detail = {"correct_loss":[],"spell_loss":[],"upper_loss":[]}
    for i in tqdm(range(0, df.shape[0]//batch_size, 1)):
        # Get the current batch
        tmp_data = df.iloc[batch_size * i: batch_size * (i+1)].to_dict('records')
        # print(tmp_data)

        batch_input_gt = [x['text'] for x in tmp_data]
        batch_input = [x['generate'] for x in tmp_data]

        # print(batch_input, batch_input_gt)

        t, c, max_batch_len = mapping_batch(batch_input)
        if max_batch_len == 0:
            continue
        # print(max_batch_len)
        label_spell = [expand_to_max_length(x['spell_label'], max_batch_len) for x in tmp_data]
        label_cap = [expand_to_max_length(x['cap_label'], max_batch_len) for x in tmp_data]
        label_correction = [get_correct(x['text'], x['generate']) for x in tmp_data]
        label_correction = [expand_to_max_length(x, max_batch_len) for x in label_correction]
        target = {
            "spell" : torch.tensor(label_spell),
            "upper" : torch.tensor(label_cap),
            "correction" : torch.tensor(label_correction)
        }
        
        # break
        try:
            # if t_correction.shape != torch.tensor(t).shape:
            #     continue
            loss, detail = train(spell_model, optimizer, loss_function, t, c, target)
            t_loss.append(loss)
            for detail_k in detail.keys():
                batch_detail[detail_k].append(detail[detail_k])
                
        except:
            # continue
            # print(torch.tensor(label_spell).shape)
            # print(torch.tensor(label_cap).shape)
            # print(torch.tensor(label_correction).shape)
            print(traceback.format_exc())
            # print(batch_train_label.shape, t_correction.shape)
        if i % 5000 == 0:
            print(f"Epoch {epoch+1} - Batch {i} - Avg Loss: {(sum(t_loss)/len(t_loss)):.4f}")
            cur_l = sum(t_loss)/len(t_loss)
            detail = {bdt: sum(batch_detail[bdt])/(len(batch_detail[bdt])) for bdt in batch_detail.keys()}
            print(detail)
            batch_detail = {"correct_loss":[],"spell_loss":[],"upper_loss":[]}
            t_loss = []
            if i % 10000 == 0 and i > 1:
                if cur_l < min_loss:
                    min_loss = cur_l
                    torch.save({
                        'model_state_dict': spell_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, f'model_loss_{min_loss}.pth')


