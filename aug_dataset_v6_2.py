def change_last_char(text):
    words = text.split()
    #for idx in range(len(words)):
        #if words[idx] in vocabs_vn_accent and words[idx][-2:] in dict_change:
            #words[idx] = words[idx][:-2] + dict_change[words[idx][-2:]]
    j = 0
    while j<50 :
        j+=1
        i = np.random.randint(len(words))
        if words[i] not in vocabs_vn_accent:
            continue
        for part_of_w in list(dict_change.keys()):
            if part_of_w in words[i]:
                words[i] = words[i].replace(part_of_w, dict_change[part_of_w])
                j = 50                
                break
    
    return " ".join(words)
# đổi các kí tự trong 1 từ : hôm -> hoom -> hmoo, nay -> nya
def random_swap_char_in_word(txt_src, txt_des, index, thresh_hold=0.5):
    texts = split_word_with_bound(txt_src)
    ls_txt_des = split_word_with_bound(txt_des)

    prob = random.random()
    if prob < thresh_hold:
        if check_syll_vn(texts[index]) and len(texts[index]) > 2:
            texts[index], _ = change_type_telex(texts[index], texts[index], 0)
            texts[index] = texts[index].replace('<oov>', '')
            texts[index] = texts[index].replace('</oov>', '')
            texts[index] = texts[index].replace('_', '')
            texts[index] = texts[index].replace(' ', '')

    else:
        if check_syll_vn(texts[index]) and len(texts[index]) > 2:
            texts[index] = texts[index]

    if len(texts[index]) > 2:

        i = np.random.randint(0, len(texts[index])-1)
        j = np.random.randint(0, len(texts[index])-1)

        ls_texts = list(texts[index])
        ls_texts[i], ls_texts[j] = ls_texts[j], ls_texts[i]

        if check_syll_vn(''.join(ls_texts)):
            texts[index] = ''.join(ls_texts)
            ls_txt_des[index] = ls_txt_des[index]
        else:
            texts[index] = split_word(''.join(ls_texts))
            ls_txt_des[index] = split_word(ls_txt_des[index])

    return ' '.join(texts), ' '.join(ls_txt_des)

# thay thế 1 từ bất kì và vỡ telex với xác suất: vui -> với -> vowis || vui -> với
def random_change_word_and_break(txt_src, txt_des, index, thresh_hold=1):
    texts = split_word_with_bound(txt_src)
    ls_txt_des = split_word_with_bound(txt_des)

    # prob = random.random()
    # if prob < thresh_hold:
    #     if (texts[index] in vocabs_vn_accent):
    #         index_random = np.random.randint(len(vocabs_vn_accent))
    #         texts[index] = vocabs_vn_accent[index_random]
    #         texts[index], _ = change_type_telex(texts[index], texts[index], 0)

    #         if 'oov' in texts[index]:
    #             ls_txt_des[index] = split_word(ls_txt_des[index])

    # else:
    if (texts[index] in vocabs_vn_accent):
        index_random = np.random.randint(len(vocabs_vn_accent))
        #texts[index] = vocabs_common_vn_accent[index_random]
        texts[index] = "<MASK>"

    return ' '.join(texts), ' '.join(ls_txt_des)

def random_change_word_simmilar_and_break(txt_src, txt_des, index, thresh_hold=1):
    texts = split_word_with_bound(txt_src)
    ls_txt_des = split_word_with_bound(txt_des)

    prob = random.random()
    if prob < thresh_hold:
        if (texts[index] in vocabs_vn_accent):
            index_random = vocabs_vn_accent.index(texts[index])
            if index_random > 5:
                texts[index] = random.choice(vocabs_vn_accent[index_random-5:index_random+5])
            else:
                texts[index] = random.choice(vocabs_vn_accent[index_random:index_random+5])

            texts[index], _ = change_type_telex(texts[index], texts[index], 0)

            if 'oov' in texts[index]:
                ls_txt_des[index] = split_word(ls_txt_des[index])

    return ' '.join(texts), ' '.join(ls_txt_des)

# bỏ space giữa 2 từ: anh chị -> anhchij || anh chị -> anhchị
def remove_split_word(txt_src, txt_des, thresh_hold=0.5):
    texts = split_word_with_bound(txt_src)
    ls_txt_des = split_word_with_bound(txt_des)

    cnt = 0
    while True:
        i = np.random.randint(len(texts)-1)
        cnt += 1
        if ('<oov>' not in str(texts[i]) and '<oov>' not in str(texts[i+1]) and check_list_punct(texts[i]) and check_list_punct(texts[i+1]))\
                or cnt == 3:
            break

    if cnt == 3:
        texts = texts
        ls_txt_des = ls_txt_des

    else:
        prob = random.random()
        if prob > thresh_hold:
            a = change_type_telex(texts[i], texts[i], 0, thresh_hold=1)[0].replace('</oov>', '')
            b = change_type_telex(texts[i + 1], texts[i + 1], 0, thresh_hold=1)[0].replace('<oov>', '')
            if a == texts[i] and b == texts[i+1]:
                texts[i] = texts[i] + texts[i+1]
                texts[i] = split_word(texts[i])

            elif a == texts[i]:
                a = split_word(a).replace('</oov>', '')
                b = '_' + remove_multi_space(b)
                texts[i] = a + b

            elif b == texts[i+1]:
                b = '_' + remove_multi_space(split_word(b).replace('<oov>', ''))
                texts[i] = a + b

            else:
                texts[i] = a + '_' + remove_multi_space(b)

            ls_txt_des[i] = remove_multi_space(split_word(ls_txt_des[i]).replace('</oov>', '') + split_word(ls_txt_des[i+1]).replace('<oov>', ''))
            # ls_txt_des[i] = split_word(ls_txt_des[i])

        else:
            texts[i] = split_word(texts[i] + texts[i+1])
            ls_txt_des[i] = split_word(ls_txt_des[i]).replace('</oov>', '') + split_word(ls_txt_des[i+1]).replace('<oov>', '')
            ls_txt_des[i] = remove_multi_space(ls_txt_des[i])

        texts.remove(texts[i + 1])
        ls_txt_des.remove(ls_txt_des[i + 1])

    return ' '.join(texts), ' '.join(ls_txt_des)

# thêm kí tự vào đầu hoặc cuối từ: nay -> nnay || nay -> nayy
def add_char_in_word(txt_src, txt_des, index, thresh_hold=0.5):
    texts = split_word_with_bound(txt_src)
    ls_txt_des = split_word_with_bound(txt_des)
    i = index

    prob = random.random()
    if prob < thresh_hold:
        if check_syll_vn(texts[i]) and texts[i][0].lower() not in special_char_2:
            texts[i] = texts[i][0] + texts[i]

    else:
        if check_syll_vn(texts[i]) and texts[i][-1].lower() not in special_char_2:
            texts[i] = texts[i] + texts[i][-1]


    return ' '.join(texts), ' '.join(ls_txt_des)

# lỗi telex: hôm -> hoom
def change_type_telex(txt_src, txt_des, index, thresh_hold=1):
    ls_txt_des = split_word_with_bound(txt_des)
    texts = split_word_with_bound(txt_src)
    i = index
    raw = texts[i]
    texts[i] = texts[i].lower()
    

    if check_syll_vn(texts[i]):
        prob = random.random()
        if prob < thresh_hold:
            if texts[i] in list(vocabs_telex.keys()):
                ls_wr_txt = vocabs_telex[texts[i]]
                texts[i] = split_word(random.choice(ls_wr_txt))
                ls_txt_des[i] = split_word(ls_txt_des[i])
    if raw[0].isupper() and raw[1:].islower():
        texts[i] = texts[i].capitalize()
    return ' '.join(texts), ' '.join(ls_txt_des)

# mất kí tự cuối hoặc bất kì: nay -> na
def convert_typing_missing_char(txt_src, txt_des, index, thresh_hold=0.6):
    texts = split_word_with_bound(txt_src)
    ls_txt_des = split_word_with_bound(txt_des)
    # i = texts.index(word)
    i = index

    if check_syll_vn(texts[i]) and len(texts[i]) > 1:
        prob = random.random()
        if prob < thresh_hold:
            texts[i] = texts[i][:-1]
            if check_syll_vn(texts[i]):
                texts[i] = texts[i]
            else:
                for k, v in enumerate(keys_break_typing):
                    if v in texts[i]:
                        texts[i] = texts[i].replace(v, values_break_typing[k])

                texts[i] = split_word(texts[i])
                ls_txt_des[i] = split_word(ls_txt_des[i])

        else:
            j = np.random.randint(len(texts[i]))
            texts[i] = texts[i][:j] + texts[i][j+1:]
            if check_syll_vn(texts[i]):
                texts[i] = texts[i]
            else:
                for k, v in enumerate(keys_break_typing):
                    if v in texts[i]:
                        texts[i] = texts[i].replace(v, values_break_typing[k])

                texts[i] = split_word(texts[i])
                ls_txt_des[i] = split_word(ls_txt_des[i])

    return ' '.join(texts), ' '.join(ls_txt_des)

# thay kí tự cuối hoặc bất kì bằng phím gần
def convert_random_word_distance_keyboard(txt_src, txt_des, index, thresh_hold=0.5):
    texts = split_word_with_bound(txt_src)
    ls_txt_des = split_word_with_bound(txt_des)
    i = index
    raw = texts[i]
    texts[i] = texts[i].lower()
    if check_syll_vn(texts[i]):
        prob = random.random()
        if prob < thresh_hold:
            if texts[i][-1] in ls_key_random:
                texts[i] = texts[i][:-1] + random.choice(data_keys_random[texts[i][-1]])
                if not check_syll_vn(texts[i]):
                    for k, v in enumerate(keys_break_typing):
                        if v in texts[i]:
                            texts[i] = texts[i].replace(v, values_break_typing[k])

                    texts[i] = split_word(texts[i])
                    ls_txt_des[i] = split_word(ls_txt_des[i])

        else:
            j = np.random.randint(len(texts[i]))
            if texts[i][j] in ls_key_random:
                texts[i] = texts[i][:j] + random.choice(data_keys_random[texts[i][j]]) + texts[i][j + 1:]
                if not check_syll_vn(texts[i]):
                    for k, v in enumerate(keys_break_typing):
                        if v in texts[i]:
                            texts[i] = texts[i].replace(v, values_break_typing[k])

                    texts[i] = split_word(texts[i])
                    ls_txt_des[i] = split_word(ls_txt_des[i])
    if raw[0].isupper() and raw[1:].islower():
        texts[i] = texts[i].capitalize()
    return ' '.join(texts), ' '.join(ls_txt_des)

def convert_last_char_distance_keyboard(txt_src, txt_des, index, thresh_hold=1):
    texts = split_word_with_bound(txt_src)
    ls_txt_des = split_word_with_bound(txt_des)
    # i = texts.index(word)
    i = index

    prob = random.random()
    if prob < thresh_hold:
        if check_syll_vn(texts[i]) and len(texts[i]) > 1 and texts[i][-1] in ls_keys_last:
            texts[i] = texts[i][:-1] + data_keys_last[texts[i][-1]][0]
            if not check_syll_vn(texts[i]):
                prob = random.random()
                if prob < 0.8:
                    for k, v in enumerate(keys_break_typing):
                        if v in texts[i]:
                            texts[i] = texts[i].replace(v, values_break_typing[k])
                    texts[i] = split_word(texts[i])
                    ls_txt_des[i] = split_word(ls_txt_des[i])

                else:
                    texts[i] = split_word(texts[i])
                    ls_txt_des[i] = split_word(ls_txt_des[i])

    return ' '.join(texts), ' '.join(ls_txt_des)

def convert_first_char_distance_keyboard(txt_src, txt_des, index, thresh_hold=1):
    texts = split_word_with_bound(txt_src)
    ls_txt_des = split_word_with_bound(txt_des)
    # i = texts.index(word)
    i = index
    raw = texts[i]
    texts[i] = texts[i].lower()
    if check_syll_vn(texts[i]) and len(texts[i]) > 1:
        prob = random.random()
        if prob < thresh_hold:
            if texts[i][0] in ls_key_random:
                texts[i] = random.choice(data_keys_random[texts[i][0]]) + texts[i][1:]
                if not check_syll_vn(texts[i]):
                    prob = random.random()
                    if prob < 0.7:
                        for k, v in enumerate(keys_break_typing):
                            if v in texts[i]:
                                texts[i] = texts[i].replace(v, values_break_typing[k])

                        texts[i] = split_word(texts[i])
                        ls_txt_des[i] = split_word(ls_txt_des[i])

                    else:
                        texts[i] = split_word(texts[i])
                        ls_txt_des[i] = split_word(ls_txt_des[i])
    if raw[0].isupper() and raw[1:].islower():
        texts[i] = texts[i].capitalize()
    return ' '.join(texts), ' '.join(ls_txt_des)

def check_word_oov(txt_src, txt_des):
    texts = split_word_with_bound(txt_src)
    ls_txt_des = split_word_with_bound(txt_des)

    for i, txt in enumerate(texts):
        if not check_syll_vn(txt) and check_list_punct(txt) and not txt.isnumeric():
            texts[i] = split_word(texts[i])
            ls_txt_des[i] = split_word(ls_txt_des[i])

        if txt.isnumeric():
            texts[i] = split_numeric(txt)
            ls_txt_des[i] = split_numeric(ls_txt_des[i])

        if not check_all_punct(txt) and not txt.isalnum():
            texts[i] = split_numeric(txt)
            ls_txt_des[i] = split_numeric(ls_txt_des[i])


    return ' '.join(texts), ' '.join(ls_txt_des)

