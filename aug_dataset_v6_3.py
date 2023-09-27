def upper_to_lower(txt_src, txt_des, thresh_hold=1):
    texts = split_word_with_bound(txt_src)
    ls_txt_des = split_word_with_bound(txt_des)
    j = 0
    while(1):
        j+=1
        i = np.random.randint(len(texts))
        if check_syll_vn(texts[i]) and texts[i][0].isupper() and texts[i][1:].islower():
            texts[i] = texts[i].lower()
            break
        if j == 10:
            break

    
    return ' '.join(texts)


def lower_to_upper(txt_src, txt_des, thresh_hold=1):
    texts = split_word_with_bound(txt_src)
    ls_txt_des = split_word_with_bound(txt_des)
    j = 0
    while (1):
        j += 1
        i = np.random.randint(len(texts))
        if check_syll_vn(texts[i]) and texts[i].islower():
            texts[i] = texts[i].capitalize()
            break
        if j == 10:
            break

    return ' '.join(texts)

# dang o day @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def upper_to_lower_all(txt_src, txt_des):
    texts = split_word_with_bound(txt_src)
    ls_txt_des = split_word_with_bound(txt_des)
    for i in range(len(texts)):
        if check_syll_vn(texts[i]) and texts[i][0].isupper() and texts[i][1:].islower():
            texts[i] = texts[i].lower()
        
    return ' '.join(texts)


def lower_to_upper_all(txt_src, txt_des):
    texts = split_word_with_bound(txt_src)
    ls_txt_des = split_word_with_bound(txt_des)
    for i in range(len(texts)):
        if check_syll_vn(texts[i]) and texts[i].islower():
            texts[i] = texts[i].capitalize()

    return ' '.join(texts)


def remove_tone_line(text):

    # texts = split_word_with_bound(text_src)
    # ls_txt_des = split_word_with_bound(text_des)
    # i = index

    intab = 'áàảãạăâắằẳẵặấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôơốồổỗộớờởỡợúùủũụưứừửữựýỳỷỹỵđ'
    outtab = "a"*7 + "ă"*5 + "â"*5 + "e"*6 + "ê"*5 + "i"*5 + "o"*7 + "ô"*5 + "ơ"*5 + "u"*6 + "ư"*5 + "y"*5 + "d"

    
    r = re.compile("|".join(intab))
    replaces_dict = dict(zip(intab, outtab))
    non_dia_str = r.sub(lambda m: replaces_dict[m.group(0)], text)

        # txt = ''.join(non_dia_str)

        # if check_syll_vn(txt):
        #     texts[i] = txt
        # else:
        #     texts[i] = split_word(texts[i])
        #     ls_txt_des[i] = split_word(ls_txt_des[i])

    return non_dia_str

def remove_accent_character(text_src, text_des, index, index_char, thresh_hold=1):

    list_index_char = []
    texts = split_word_with_bound(text_src)
    ls_txt_des = split_word_with_bound(text_des)

    i = index
    k = index_char

    if check_syll_vn(texts[i]):
        # for j in range(len(texts[i])):
        #     if texts[i][j] != unidecode.unidecode(texts[i][j]):
        #         list_index_char.append(j)
        # print(list_index_char)
       #  k = random.choice(list_index_char)
        #print(k)
        a = remove_tone_line(texts[i][k])
        texts[i] = texts[i].replace(texts[i][k] , a)
        #print(texts)
        txt = ''.join(texts)

        # if check_syll_vn(txt):
        #     texts[i] = txt
        # else:
        #     texts[i] = split_word(texts[i])
        #     ls_txt_des[i] = split_word(ls_txt_des[i])

    return ' '.join(texts), ' '.join(ls_txt_des)

def token_character(line):
    ls = split_word_with_bound(line)
    current = ''
    for txt in ls:
        
        if ('_' not in txt) & ("|" not in txt):
            text = list(txt)
            #print(text)
            current += " " + " _".join(text)
        
        else:
            current += " " + txt
    return current

def lower_first_word(sent):
    i = random.choice([0 ,1 ])
    if i == 0 :
        #try: 
        text = split_word_with_bound(sent)

        if text[0][0].isupper():
            text[0] = text[0].lower()
            txt_process_1 =  " ".join(text)
            #print(src_process_1)
        else:
            txt_process_1 = " ".join(text)


    else:
        txt_process_1 = sent
    
    return txt_process_1

def replace_punc(sent):

    src_process = sent.replace(" .", "")

    return src_process



def augment_data(sent, error):
    # print(cnt)
    # sent = re.sub("\s\s+" , " ", sent)
    # sent = sent.replace(u'\u200c','')
    # sent = sent.replace(u'\u200b','')
    origin = sent
    sent = " ".join([t for t in sent.split() if len(t) > 0 ])
    

    ls = []
    # # trace_code = []
    #sent_after_oov = handle_punctuation(sent)

    #sent_after_oov = lower_first_word(sent_after_oov)
    #sent_after_oov = replace_punc(sent_after_oov)

    text_src, text_des = sent, sent
    num_wrong = int(np.round(0.075  * len(text_src.split())))
    if num_wrong ==0:
        num_wrong =1
    if error == None:
        error = random.choice(errors)
    for line in [sent]:
        # print("Line: ",line)
        # print("Length: ", len(line))
        try:
            list_index =[]
            for index, word in enumerate(split_word_with_bound(text_src)):
                if word.isalpha() and word in vocabs_vn_accent:
                    list_index.append(index)
            if len(list_index) == 0:
                ls.append('{} || {}'.format(text_src, text_des))
        #
            if error == "accent":
        # "của" -> "cua" cả câu
                text_src, text_des = remove_accent(text_src), text_des
                return text_src
        #     ls.append('{} || {}'.format(text_src, text_des))
            elif error ==  "upper_to_lower":
                return upper_to_lower(text_src, text_des)

            elif error ==  "lower_to_upper":
                return lower_to_upper(text_src, text_des)
            
            elif error ==  "upper_to_lower_all":
                return upper_to_lower_all(text_src, text_des)

            elif error ==  "lower_to_upper_all":
                return lower_to_upper_all(text_src, text_des)
        
            elif error == "random_accent":
        # "của" -> cua
                for j in range(0,num_wrong):
                    i = random.choice(list_index)
                # # print(i)
                    text_src, text_des = random_remove_accent(text_src, text_des, i)
                # ls.append('{} || {}'.format(text_src, text_des))
                return text_src

            elif error == "typo":
        # "của" -> "cuar"
                for j in range(0,num_wrong):
                    i = random.choice(list_index)
                    text_src, text_des = change_type_telex(text_src, text_des, i)
                return text_src
        #     ls.append('{} || {}'.format(text_src, text_des))
        # "đã" -> "dd"
            elif error == "remove_char":
                for j in range(0,num_wrong):
                    i = random.choice(list_index)
                    text_src, text_des = convert_typing_missing_char(text_src, text_des, i)
                return text_src
            # ls.append('{} || {}'.format(text_src, text_des))
        # "tới" -> "toswj
            elif error == "key_board":
                for j in range(0,num_wrong):
                    i = random.choice(list_index)
                    text_src, text_des = convert_random_word_distance_keyboard(text_src, text_des, i)
                return text_src
        #     ls.append('{} || {}'.format(text_src, text_des))
        # "thức" -> "thcuws"
            elif error == "swap_char":
                for j in range(0,num_wrong):
                    i = random.choice(list_index)
                    text_src, text_des = random_swap_char_in_word(text_src, text_des, i)
                return text_src
        #     ls.append('{} || {}'.format(text_src, text_des))
        # "mức" -> "juwcs"
            elif error == "fisrt_char_keyboard":
                for j in range(0,num_wrong):
                    i = random.choice(list_index)
                    text_src, text_des = convert_first_char_distance_keyboard(text_src, text_des, i)
                return text_src
        #     ls.append('{} || {}'.format(text_src, text_des))
        # "công" -> "coongg"
            elif error == "add_char":
                for j in range(0,num_wrong):
                    i = random.choice(list_index)
                    text_src, text_des = add_char_in_word(text_src, text_des, i)
                return text_src
        #     ls.append('{} || {}'.format(text_src, text_des))
        #
            elif error == "change_first_char":
        #Thay đổi những chứ viết hoa thành viết thường -> bỏ
                for j in range(0, num_wrong):
                    text_src, text_des = change_first_char(text_src, text_des)
                return text_src

            elif error == "change_last_char":
                for j in range(0, num_wrong):
                    text_src = change_last_char(text_src)
                return text_src
                    # ls.append('{} || {}'.format(text_src, text_des))
        #
        #Từ viết thường lên viết hoa -> bỏ
            # i = random.choice(list_index)
            # text_src, text_des = lower_to_upper(text_src, text_des, i)
            # ls.append('{} || {}'.format(text_src, text_des))
            #
            # Viết hoa thành viết thường bất kỳ
            # list_index_upper = []
            # for i in list_index:
            #     list_word_split = split_word_with_bound(text_src1)
            #     if list_word_split[i][0].isupper() == True:
            #         if i != 0:
            #             list_index_upper.append(i)
            # if len(list_index_upper) == 0:
            #     text_src, text_des = text_src1, text_des1
            # if len(list_index_upper) > 0:
            #     i = random.choice(list_index_upper)
            #     text_src, text_des = upper_to_lower(text_src1, text_des1, i)
            # ls.append('{} || {}'.format(text_src, text_des))
            #
            # nhỏ ->  nho,  nhổ -> nhô
            elif error == "remove_tone":
                for i in list_index:
                    list_word_split = split_word_with_bound(text_src)
                    list_index_char = []
                    for j in range(len(list_word_split[i])):
                        if list_word_split[i][j] != remove_tone_line(list_word_split[i][j]):
                            list_index_char.append(j)

                    prob = random.random()
                    if (len(list_index_char) == 1) and (prob <= 0.3):
                        k = random.choice(list_index_char)
                        text_src, text_des = remove_accent_character(text_src, text_des, i, k)
                    if (len(list_index_char) >= 2) and (prob <= 0.5):
                        randomlist = random.sample(list_index_char, 2)
                        for k in randomlist:
                            text_src, text_des = remove_accent_character(text_src, text_des, i, k)
                    else:
                        text_src, text_des = text_src, text_des
                return text_src
            #
            elif error == "change_tone":
                for j in range(0, num_wrong):
                    text_src, text_des = change_accent(text_src, text_des)
                return text_src
            # result = list(zip(text_src, text_des))
            # for i in result:
            #     sent_src = i[0]
            #     sent_des = i[1]
            # ls.append('{} || {}'.format(text_src, text_des))

            elif error == "not_change":
                return origin

            elif error == "teen_code":
                new_word = list(map(change_to_teencode,text_src.strip().lower().split()))
                return " ".join(new_word)
    
        except:
            return origin


def create_data(text, error = None):
    text = augment_data(text, error)
    text = augment_data(text, error)
    return text
      
    # ls_text = list(set(ls))
    
    # return ls_text

    # ls_text = list(set(ls))
    #
    #
    # return ls_text

# if __name__ == '__main__':
#     # s = '<oov> hôm </oov> hỏi <oov> gấu <oov> <oov> ham </oov> <oov> miến </oov> <oov> trôi </oov> <oov> đi </oov> <oov> hoobc </oov>'
#     s = "Dodge đã công bố giá bán của mẫu sedan-cỡ nhỏ Dart hoàn toàn mới trước khi (nó) chính thức được bán ra vào tháng 6 tới đây."
#     #s = 'Lilly nhận được một cuộc gọi từ một khách hàng tiềm năng , hỏi ông có thể tới chỗ cô không'
#     #s = 'Hải Dương phải tự đặt trong tình trạng như đà nẵng trước đây,  Bộ y tế sẽ chi viện tối đa, tuy nhiên hải dương phải tập trung cao độ, làm sao để trong 10 ngày phải khoanh vùng triệt để'
#     #s = 'Ngân hàng Nhà nước Việt Nam'
#     #s = ['Cach', '2', '_:', 'Nguyen', 'lieu', '_:', 'De', 'lam', 'mon', 'thit', 'rang', 'chay', 'canh', 'ban', 'can', 'chuan', 'bi', 'nhung', 'nguyen', 'lieu', 'sau', '_:', '-', '<oov> 4 _0 _0 _g </oov>', 'thit', 'ba', 'roi', '(', 'hoac', 'thit', 'nac', 'dam', 'co', 'dat', 'mo', '_)', '-', '2', '_-', '3', 'cu', 'sa', '-', '1', 'thia', 'mam', 'tom', '-', '1', 'thia', 'nuoc', 'mam', '-', '', '1/2', '', 'thia', 'duong', '-', 'Vai', 'cu', 'hanh', 'tim', 'Chuan', 'bi', '_:', 'Thit', 'loc', 'bo', 'bi', '_,', 'thai', 'mieng', 'vua', 'an', '_.']
#     # text_src =
#
#     # text_des = text_src
#     # print(len(s.split()))
#     # src, des = augment_data(s, s, 4)
#
#     data = augment_data(s, 15000)
#     #print('src: ', src)
#     print('data: ', data)
#
#     data = handle_punctuation(s)
#     print(data)
    
 

