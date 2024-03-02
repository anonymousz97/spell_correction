# Gen training data

from tqdm.notebook import tqdm
import pandas as pd
import argparse
import re
import string
from gen_err import SynthesizeData

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
                    
            
gen_spell = SynthesizeData("vietnamese.txt")

# Create the parser
parser = argparse.ArgumentParser()

# Add an argument for the input file
parser.add_argument('--input-file', type=str, required=True,
                    help='The input file to process')

parser.add_argument('--max_rows', type=int, default=1000000)
parser.add_argument('--output-file', type=str, default="train_format.csv")

# Parse the arguments
args = parser.parse_args()

# Get the input file from the arguments
input_file = args.input_file
max_rows = args.max_rows
output_file = args.output_file
cnt = 0
lst_gt = []
lst_gen = []
spell_label = []
cap_label = []
with open(input_file, 'r') as file:
    for line in tqdm(file, total=cnt):
        cnt += 1
        s = []
        line = split_punc_to_word(line)
        # print(line)
        res = gen_spell.add_noise(line)
        # print(res)
        lst_gt.append(line)
        lst_gen.append(res[0])
        label_cap = [1 if x[0].isupper() else 0 for x in line.split(' ')]
        label_spell = [0 if x != 1 else 1 for x in res[1]]
        spell_label.append(label_spell)
        cap_label.append(label_cap)
        if cnt == max_rows:
            break

df = pd.DataFrame({"text":lst_gt,'generate':lst_gen,"spell_label":spell_label,"cap_label":cap_label})
# print(df)
df.to_csv(output_file,index=False)
