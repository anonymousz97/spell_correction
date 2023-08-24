import pandas as pd# from create_data import SynthesizeData
from aug_data import *
from utils import check_and_reduce, set_punctuations
from utils import pad_sents, reduce_wrong_word_test, norm_text
import pandas as pd
import argparse

v_model = {}
i = 0
for line in open("vocab.txt"):
    line = norm_text(line.strip())
    if line not in v_model:
        v_model[line.strip()] = i
        i+=1

def check_cap(word, v_model):
    if word.lower() in v_model:
        if len(word) == 1:
        #    if word.isupper() == True:
        #        return True
        #    else:
            return False
        else:
            if word[0].isupper() == True and word[1:].islower() == True:
                return True
            else:
                return False
    return False

def check_and_reduce_word(word):
    if word not in v_model:
        return reduce_wrong_word_test(word)
    else:
        return word

def check_and_reduce_text(text):
    text = text.split()
    text = list(map(check_and_reduce_word, text))
    return " ".join(text)

def compare(check, correct):
    x = check.strip().split()
    y = correct.strip().split()
    label = []
    for i in range(len(x)):
        if x[i] == y[i]:
            label.append("0")
        else:
            label.append("1")
    return  " ".join(label)

def label_cap(correct, v_model):
    x = correct.strip().split()
    label = []
    for word in x:
        if check_cap(word, v_model):
            label.append("1")
        else:
            label.append("0")
    return " ".join(label)

# def random_sent(sent):
#     length = len(sent.split())
#     a = random.choice(range(length))
#     b = random.choice(range(length))
#     if b > a:
#         a,b = b,a
#     i = 0 
#     while(a - b <6):
#         a = random.choice(range(length))
#         b = random.choice(range(length))
#         if b > a:
#             a,b = b,a
#         i+=1
#         if i == 30:
#             return sent
#     return " ".join(sent.split()[b:a+1])
def create(path):
    y = []
    if path == "data_eval.txt":
        for line in open(path):
            y.append(line.strip())
    else:
        for line in open(path):
            if random.random() > 0.75:
                y.append(line.strip())
    data= pd.DataFrame({"Check": ["x"]*len(y), "Correct": y , "Label": ["x"]*len(y)})
    print("Load")
    data["Label_cap"] = data.apply(lambda x: label_cap(x["Correct"], v_model), axis=1)
    data["Correct"] = data.apply(lambda x: x["Correct"].strip().lower(), axis=1)
    data["Check"] = data.apply(lambda x: norm_text(change_sent(x["Correct"])), axis=1)
    data["Sub"] = data.apply(lambda x: len(x["Check"].strip().split()) - len(x["Correct"].strip().split()), axis=1)
    data.drop(data.loc[data["Sub"] != 0].index, axis=0, inplace=True)
    data.drop("Sub", axis=1, inplace=True)
    data["Label"] = data.apply(lambda x: compare(x["Check"], x["Correct"]), axis=1)
    data["Check"] = data.apply(lambda x: norm_text(check_and_reduce_text(x["Check"])), axis=1)
    print("Save")
    path = path[:-4] + ".csv"
    data.to_csv(path, index = False)
    return data
