import numpy as np
import random
from preprocess.handle_text import *
import re
from collections import Counter

typo = {"ă": ["aw"], "â": ["aa"], "á": ["as"], "à": ["af"], "ả": ["ar"],
        "ẫ": ["aax"], "ấ": ["aas"], "ầ": ["aaf"], "ẩ": ["aar"], "ậ": ["aaj"],
        "ã": ["ax"], "ạ": ["aj"], "ắ": ["aws"], "ổ": ["oor"], "ỗ": ["oox"],
        "ộ": ["ooj"], "ơ": ["ow"],
        "ằ": ["awf"], "ẳ": ["awr"], "ẵ": ["awx"], "ặ": ["awj"], "ó": ["os"],
        "ò": ["of"], "ỏ": ["or"], "õ": ["ox"], "ọ": ["oj"], "ô": ["oo"],
        "ố": ["oos"], "ồ": ["oof"],
        "ớ": ["ows"], "ờ": ["owf"], "ở": ["owr"], "ỡ": ["owx"], "ợ": ["owj"],
        "é": ["es"], "è": ["ef"], "ẻ": ["er"], "ẽ": ["ex"], "ẹ": ["ej"],
        "ê": ["ee"], "ế": ["ees"], "ề": ["eef"],
        "ể": ["eer"], "ễ": ["eex"], "ệ": ["eej"], "ú": ["us"], "ù": ["uf"],
        "ủ": ["ur"], "ũ": ["ux"], "ụ": ["uj"], "ư": ["uw"], "ứ": ["uws"],
        "ừ": ["uwf"], "ử": ["uwr"], "ữ": ["uwx"],
        "ự": ["uwj"], "í": ["is"], "ì": ["if"], "ỉ": ["ir"], "ị": ["ij"],
        "ĩ": ["ix"], "ý": ["ys"], "ỳ": ["yf"], "ỷ": ["yr"], "ỵ": ["yj"], "ỹ": ["yx"],
        "đ": ["dd"]}


teencode_dict = {'mình': ['mk', 'mik', 'mjk','m'], 'vô': ['zô', 'zo', 'vo'], 'vậy': ['zậy', 'z', 'zay', 'za'],
                        'phải': ['fải', 'fai', ], 'biết': ['bit', 'biet'], "qđxxst" : ["qdxxst"], "qđst" : ["qdst"], "đắk" : ["đăk", "đắc", "ddawsk"],
                        'rồi': ['rùi', 'ròi', 'r'], 'bây': ['bi', 'bay'], 'giờ': ['h'], "qđ" : ["qd"], "hđxx" : ["hdxx"], "qđxx": ["qdxx"], "csđt" :["csdt"],
                        'không': ['k', 'ko', 'khong', 'hk', 'hong', 'hông', '0', 'kg', 'kh'], "lắk" : ["lắc", "lăk", "lawsk"], "hđtd": ["hdtd"],
                        'đi': ['di', 'dj'], 'gì': ['j'], 'em': ['e'], 'được': ['dc', 'đc'], "qsdđ" : ["qsdd"], "hđ" :["hd"], "heroine" : ["hêrôin"], "qđst" :["qdst"],
                        'tôi': ['t'], 'chồng': ['ck'], 'vợ': ['vk'], 'facebook' : ['fb'], 'đồng' : ['đ'], "hngđ" : ["hngd"]
                        }

def change_to_teencode(word):
    if word in teencode_dict:
        return random.choice(teencode_dict[word])
    else:
        return word
special_char = ["a", "e", "o", "d"]
special_char_2 = ["â", "ê", "ô", "đ"]
errors = [ "typo","change_tone", "remove_tone","change_tone", "random_accent", "remove_char","change_tone","change_last_char",
"key_board", "swap_char","change_tone","fisrt_char_keyboard", "add_char", "change_first_char","change_tone", "change_first_char"]
# def check_syll_vn(txt):
#     if norm_text(txt) in vocabs:
#         return True
#     else:
#         return False
# bỏ dấu 1 từ random
def random_remove_accent(text_src, text_des, index, thresh_hold=1):
    texts = split_word_with_bound(text_src)
    i = index
    if (check_syll_vn(texts[i]) and remove_accent(texts[i]) != texts[i]):
        prob = random.random()
        if prob < thresh_hold:
            texts[i] = remove_accent(texts[i])

    return ' '.join(texts), text_des


lst = [['ã', 'à', 'á', 'ạ'],
            ['ẵ', 'ă', 'ằ', 'ắ', 'ặ'],
            ['â', 'ẫ', 'ầ', 'ấ', 'ậ'],
            ['e', 'è', 'é', 'ẹ', 'ẽ'],
            ['ê', 'ề', 'ế', 'ệ', 'ễ'],
            ['ò', 'ó', 'ọ', 'õ'],
            ['ô', 'ồ', 'ố', 'ộ', 'ỗ'],
            ['ờ', 'ớ', 'ợ'],
            ['ì', 'í', 'ị'],
            ['ủ', 'ù', 'ú', 'ụ'],
            ['ừ', 'ứ', 'ự'],
            ['ỳ', 'ý', 'ỵ']]


dict_change = {'iu': 'ui', 'ỉu': 'ủi', 'ĩu': 'ũi', 'ịu': 'ụi', 'íu': 'úi', 'ìu': 'ùi',
                'ui': 'iu', 'ủi': 'ỉu', 'ũi': 'ĩu', 'ụi': 'ịu', 'úi': 'íu', 'ùi': 'ìu',
                'ia': 'ai', 'ỉa': 'ải', 'ĩa': 'ãi', 'ịa': 'ại', 'ía': 'ái', 'ìa': 'ài',
                'ai': 'ia', 'ải': 'ỉa', 'ãi': 'ĩa', 'ại': 'ịa', 'ái': 'ía', 'ài': 'ìa',
                'ao': 'oa', 'áo': 'oá', 'ào': 'oà', 'ảo': 'oả', 'ão': 'oã', 'ạo': 'oạ',
                'oa': 'ao', 'oá': 'áo', 'oà': 'ào', 'oả': 'ảo', 'oã': 'ão', 'oạ': 'ạo',
                'ua': 'au', 'úa': 'áu', 'ùa': 'àu', 'ủa': 'ảu', 'ũa': 'ãu', 'ụa': 'ạu',
                'au': 'ua', 'áu': 'úa', 'àu': 'ùa', 'ảu': 'ủa', 'ãu': 'ũa', 'ạu': 'ụa',
                'oe': 'eo', 'oé': 'éo', 'oè': 'èo', 'oẻ': 'ẻo', 'oẽ': 'ẽo', 'oẹ': 'ẹo',
                'eo': 'oe', 'èo': 'oè', 'éo': 'oé', 'ẻo': 'oẻ', 'ẽo': 'oẽ', 'ẹo': 'oẹ'}

vocabs_accent = {}
for ls in lst:
    for i in ls:
        list_pop = ls[:]
        list_pop.remove(i)
        vocabs_accent[i] = list_pop


# thay đổi dấu câu
def change_accent(txt_src, txt_des, thresh_hold=1):
    ls_1 = ['ả', 'ẳ', 'ấ', 'ẻ', 'ể', 'ỏ', 'ổ', 'ở', 'ỉ', 'ỷ', 'ủ', 'ử']
    ls_2 = ['ã', 'ẵ', 'ẫ', 'ẽ', 'ễ', 'õ', 'ỗ', 'ỡ', 'ĩ', 'ỹ', 'ũ', 'ữ']

    ls_3 = ['a', 'á', 'à', 'ã', 'ả', 'ạ', 'e', 'é', 'è', 'ẽ', 'ẻ', 'ẹ', 'o', 'ó', 'ò', 'õ', 'ỏ', 'ọ']
    ls_4 = ['â', 'ấ', 'ầ', 'ẫ', 'ẩ', 'ậ', 'ê', 'ế', 'ề', 'ễ', 'ể', 'ệ', 'ô', 'ố', 'ồ', 'ỗ', 'ổ', 'ộ']

    list_vowel = ['o', 'e', 'i', 'u', 'a', 'ê']

    texts = split_word_with_bound(txt_src)
    ls_txt_des = split_word_with_bound(txt_des)

    cnt = 0
    # ls = random.sample(texts, len(texts))
    #
    # prob = random.random()
    # đổi dấu hỏi , ngã
    # if prob < 0.4:
    #     print('1.1')
    #     for k, txt in enumerate(texts):
    #         if txt.isalpha() and 'oov' not in str(txt):
    #             for e, i in enumerate(txt):
    #                 if i in ls_1:
    #                     index = ls_1.index(i)
    #                     texts[k] = texts[k].replace(i, ls_2[index])
    #                     cnt += 1
    #                     break
    #
    #                 if i in ls_2:
    #                     index = ls_2.index(i)
    #                     texts[k] = texts[k].replace(i, ls_1[index])
    #                     cnt += 1
    #                     break
    #
    #             if texts[k] in vocabs_vn_accent:
    #                 texts[k] = texts[k]
    #             else:
    #                 for j, v in enumerate(keys_break_typing):
    #                     if v in texts[k]:
    #                         texts[k] = texts[k].replace(v, values_break_typing[j])
    #
    #                 texts[k] = split_word(texts[k])
    #                 ls_txt_des[k] = split_word(ls_txt_des[k])
    #
    #         if cnt != 0:
    #             break

    # list_sent_src = []
    # list_sent_des = []

    # if prob < 1:
        # print('1.2')
    while cnt < 5:
        i = np.random.randint(len(texts))
        cnt += 1

        if (texts[i].isalpha() and '<oov>' not in texts[i]):
            raw = texts[i]
            if raw.isupper():
                continue
            texts[i] = texts[i].lower()
            a = texts[i]
            if remove_accent(texts[i]) in vocab_confusion_word_key:
                list_word_choice = vocab_confusion_word[remove_accent(texts[i])]

                word_change = random.choice(list_word_choice)
                texts[i] = split_word(word_change)
                ls_txt_des[i] = split_word(a)
                if raw[0].isupper():
                    texts[i] = texts[i].capitalize()

            break

        if cnt == 5:
            break
        #print(list_sent_src, list_sent_des)
    #result = list(zip(list_sent_src, list_sent_des))
    #print(result)
    # for i in result:
    #     sent_src = i[0]
    #     sent_des = i[1]
    texts = ' '.join(texts)
    ls_txt_des = ' '.join(ls_txt_des)
    return texts, ls_txt_des

#def write_list_pair_sentence_to_file(list_pair_sentence , file_write):
    

# xóa 1 từ bất kì nằm trong vocab
# def random_del_word(txt_src, txt_des, thresh_hold=1):
#     texts = split_word_with_bound(txt_src)
#     cnt = 0
#
#     while True:
#         i = np.random.randint(len(texts))
#         cnt += 1
#         if (texts[i].isalpha() and '<oov>' not in str(texts[i])):
#             prob = random.random()
#             if prob < thresh_hold:
#                 texts.remove(texts[i])
#             break
#
#         if cnt == 2:
#             texts = texts
#             break
#
#     return ' '.join(texts), txt_des

# thêm 1 từ bất kì trong vocab
# def random_add_word(txt_src, txt_des, thresh_hold=1):
#     texts = split_word_with_bound(txt_src)
#
#     prob = random.random()
#     if prob < thresh_hold:
#         index = np.random.randint(len(texts))
#         word_random = random.choice(vocabs_vn_accent[32:])
#         texts.insert(index, word_random)
#
#     return ' '.join(texts), txt_des

# thay đổi chữ đầu : n->l , s->x ,...
#def change_first_char(text_src, text_des, thresh_hold=1):
    #text_src = "hello i am"
    #texts = split_word_with_bound(text_src)
    #ls_text_des = split_word_with_bound(text_des)
    #consonants_1 = ['b', 'v', 'l', 'h', 'c', 'n', 'm', 'd', 'đ', 't', 'x', 's', 'r', 'k', 'g']
    #consonants_2 = ['th', 'ch', 'kh', 'ph', 'nh', 'gh', 'qu', 'gi', 'ng', 'tr']
    #consonants_3 = ['ngh']
    #consonants = consonants_1 + consonants_2 + consonants_3
    #idx = np.random.randint(len(texts))
    #k = 0
    #while(texts[idx] not in vocabs_vn_accent):
        #k += 1
        #idx = np.random.randint(len(texts))
        #if k ==10:
            #return " ".join(texts), ' '.join(ls_text_des)
            #break
    #if texts[idx][:3] in consonants_3:
        #consonant = random.choice(consonants)
        #texts[idx] = consonant + texts[idx][3:]
    #elif texts[idx][:2] in consonants_2:
        #consonant = random.choice(consonants)
        #texts[idx] = consonant + texts[idx][2:]
    #elif texts[idx][:1] in consonants_1:
        #consonant = random.choice(consonants)
        #texts[idx] = consonant + texts[idx][1:]
    #return " ".join(texts), ' '.join(ls_text_des)
def change_first_char(text_src, text_des, thresh_hold=1):
    texts = split_word_with_bound(text_src)
    ls_text_des = split_word_with_bound(text_des)
    ls_char_1 = ['x', 's', 'r', 'd', 'n', 'l']
    ls_char_2 = ['ch', 'tr', 'gi']
    cnt = 0

    while cnt < 50:
        i = np.random.randint(len(texts))
        cnt += 1
        raw = texts[i]
        if raw.isupper():
            continue
        texts[i] = texts[i].lower()
        if ('<oov>' not in str(texts[i]) and (texts[i][0] in ls_char_1 or texts[i][:2] in ls_char_2)) or cnt == 50:
            break

    if cnt == 50:
        texts = texts
    else:
        if check_syll_vn(texts[i]):
            prob = random.random()
            txt = list(texts[i])
            if prob < thresh_hold:

                if txt[0] == 's':
                    txt[0] = 'x'
                    txt = ''.join(txt)
                elif txt[0] == 'x':
                    txt[0] = 's'
                    txt = ''.join(txt)

                elif txt[0] == 'r':
                    txt[0] = 'd'
                    txt = ''.join(txt)
                elif txt[0] == 'd':
                    prob = random.random()
                    if prob < 0.5:
                        txt[0] = 'r'
                        txt = ''.join(txt)
                    else:
                        txt[0] = 'gi'
                        txt = ''.join(txt)

                elif ''.join(txt[:2]) == 'gi':
                    txt[0] = 'd'
                    txt[1] = ''
                    txt = ''.join(txt)

                elif txt[0] == 'n':
                    txt[0] = 'l'
                    txt = ''.join(txt)

                elif txt[0] == 'l':
                    txt[0] = 'n'
                    txt = ''.join(txt)

                elif ''.join(txt[:2]) == 'ch':
                    txt[0] = 't'
                    txt[1] = 'r'
                    txt = ''.join(txt)

                elif ''.join(txt[:2]) == 'tr':
                    txt[0] = 'c'
                    txt[1] = 'h'
                    txt = ''.join(txt)

            if check_syll_vn(txt):
                texts[i] = txt
            else:
                texts[i] = split_word(txt)
                ls_text_des[i] = split_word(ls_text_des[i])
            if raw[0].isupper():
                texts[i] = texts[i].capitalize()

    result = ' '.join(texts)

    return result, ' '.join(ls_text_des)
