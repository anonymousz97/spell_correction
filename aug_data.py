import random
import unidecode
from utils import norm_text
v = {}
i = 0
for line in open("all-vietnamese-syllables.txt"):
#     line = norm_text(line.strip())
    if line not in v:
        v[line.strip()] = i
        i+=1
      
    
typo = {"ă": "aw", "â": "aa", "á": "as", "à": "af", "ả": "ar",
        "ẫ": "aax", "ấ": "aas", "ầ": "aaf", "ẩ": "aar", "ậ": "aaj",
        "ã": "ax", "ạ": "aj", "ắ": "aws", "ổ": "oor", "ỗ": "oox",
        "ộ": "ooj", "ơ": "ow",
        "ằ": "awf", "ẳ": "awr", "ẵ": "awx", "ặ": "awj", "ó": "os",
        "ò": "of", "ỏ": "or", "õ": "ox", "ọ": "oj", "ô": "oo",
        "ố": "oos", "ồ": "oof",
        "ớ": "ows", "ờ": "owf", "ở": "owr", "ỡ": "owx", "ợ": "owj",
        "é": "es", "è": "ef", "ẻ": "er", "ẽ": "ex", "ẹ": "ej",
        "ê": "ee", "ế": "ees", "ề": "eef",
        "ể": "eer", "ễ": "eex", "ệ": "eej", "ú": "us", "ù": "uf",
        "ủ": "ur", "ũ": "ux", "ụ": "uj", "ư": "uw", "ứ": "uws",
        "ừ": "uwf", "ử": "uwr", "ữ": "uwx",
        "ự": "uwj", "í": "is", "ì": "if", "ỉ": "ir", "ị": "ij",
        "ĩ": "ix", "ý": "ys", "ỳ": "yf", "ỷ": "yr", "ỵ": "yj", "ỹ": "yx",
        "đ": "dd"}

key_board = {
    "q": ["w", "a", "s"],
    "w": ["q", "e", "a", "s", "d"],
    "e": ["w", "r", "s", "d", "f"],
    "r": ["e", "t", "d", "f", "g"],
    "t": ["r", "y", "f", "g", "h"],
    "y": ["t", "u", "g", "h", "j"],
    "u": ["y", "h", "j", "k", "i"],
    "i": ["o", "j", "k", "l", "u"],
    "o": ["i", "p", "k", "l"],
    "p": ["o", "l"],
    "a": ["q", "w", "s", "z", "x"],
    "s": ["q", "w", "e", "a", "d", "z", "x", "c"],
    "d": ["w", "e", "r", "s", "f", "x", "c", "v"],
    "f": ["e", "r", "t", "d", "g", "c", "v", "b"],
    "g": ["r", "t", "y", "f", "v", "b", "n", "h"],
    "h": ["t", "y", "u", "j", "g", "n", "m", "b"],
    "j": ["y", "u", "i", "h", "k", "n", "m"],
    "k": ["u", "i", "o", "j", "l", "m"],
    "l": ["i", "o", "p", "k"],
    "z": ["a", "s", "x"],
    "x": ["a", "s", "d", "z", "c"],
    "c": ["s", "d", "f", "x", "v"],
    "v": ["d", "f", "g", "c", "b"],
    "b": ["f", "g", "h", "v", "n"],
    "n": ["g", "h", "j", "b", "m"],
    "m": ["h", "j", "k", "n"]
    }

last_key_board = {"m": "n",
                "n": "m",
                "h": "g",
                "u": "i",
                "g": "h",
                "i": "u"
                }

word_confusion = {}
for line in open("model_all-vietnamese-words-confusionset.txt"):
    t = line.strip()
    t = t.split(":")
    if t[0] not in word_confusion:
        word_confusion[t[0]] = t[1].split("|")
        if t[0] not in word_confusion[t[0]]:
            word_confusion[t[0]].append(t[0])
            
change_char_still_vocab = {}
for line in open("change_char_still_vocab.txt"):
    t = line.strip()
    t = t.split(":")
    if t[0] not in change_char_still_vocab:
        change_char_still_vocab[t[0]] = t[1].split("|")
        

add_char_still_dict = {}
for line in open("add_char_still_dict.txt"):
    t = line.strip()
    t = t.split(":")
    if t[0] not in add_char_still_dict:
        add_char_still_dict[t[0]] = t[1].split("|")

confused_word_same_error = {}
for line in open("confused_word_same_error.txt"):
    t = line.strip()
    t = t.split(":")
    if t[0] not in confused_word_same_error:
        confused_word_same_error[t[0]] = t[1].split("|")

def change_tone_word(word):
    """
    cá -> cã
    cá -> ca
    """
    try:
        return (random.choice(word_confusion[unidecode.unidecode(word)])), True
    except:
        return word, False
    
def change_first_char(word):
    """
    hà nội -> hà lội
    """
    ls_char_1 = ['x', 's', 'r', 'd', 'n', 'l']
    ls_char_2 = ['ch', 'tr', 'gi']
    try:
        if word[0] not in ls_char_1 and word[:2] not in ls_char_2:
            return word, False
    except:
        return word, False
    txt = list(word)
    
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

    elif word[:2] == 'gi':
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

    return txt, True
def change_char(word):
    """
    án treo -> án troe
    """
    try:
        w = "".join(sorted(word))
        if w in change_char_still_vocab:
            word_rep = random.choice(change_char_still_vocab[w])
            while(word_rep == word):
                word_rep = random.choice(change_char_still_vocab[w])
        return word_rep, True
    except:
        return word, False
def typo_error(word):
    """
    chả cá -> char cas
    """
    try:
        chars = list(word)
        for i in range(len(chars)):
            if chars[i] in typo:
                chars[i] = typo[chars[i]]
        return "".join(chars), True
    except:
        return word, False

def add_char(word):
    """
    anh -> annh
    ánh -> aqnh
    """
    if word in add_char_still_dict:
        if random.random() < 0.2:
            return random.choice(add_char_still_dict[word]), True
    if random.random() < 0.3:
        if random.random() < 0.4:
            list_char = list(word)
            i = random.choice(range(len(list_char)))
            word = word[0:i] + word[i] + word[i:]
            return word, True
        else:
            word,_ = typo_error(word)
            list_char = list(word)
            i = random.choice(range(len(list_char)))
            word = word[0:i] + word[i] + word[i:]
            return word, True
    else:
        if random.random() < 0.4:
            list_char = list(word)
            j =0
            while(j<7):
                try:
                    i = random.choice(range(len(list_char)))
                    if random.random() < 0.5:
                        word = word[0:i] + random.choice(key_board[word[i]]) + word[i:]
                    else:
                        word = word[0:i+1] + random.choice(key_board[word[i]]) + word[i+1:]
                    break
                except:
                    j+=1
            if j ==7:
                return word, False
            return word, True
        else:
            word,_ = typo_error(word)
            list_char = list(word)
            i = random.choice(range(len(list_char)))
            if random.random() < 0.5:
                word = word[0:i] + random.choice(key_board[word[i]]) + word[i:]
            else:
                word = word[0:i+1] + random.choice(key_board[word[i]]) + word[i+1:]
            return word, True
        
    return word, False

def remove_char(word):
    """
    tín -> tí
    tín -> tisn -> tsn
    """
    if random.random() < 0.4:
        word = word
    else:
        word, _ = typo_error(word)
    i = random.choice(range(len(word)))
    if i == 0:
        try:
            word = word[1:]
            return word, True
        except:
            return word, False
    elif i == len(word) -1:
        try:
            word = word[:-1]
            return word, True
        except:
            return word, False
    else:
        try:
            word = word[:i] + word[i+1:]
            return word, True
        except:
            return word, False
def change_char_keyboard(word):
    """
    tuế -> tiế
    """
    if word[-1] in last_key_board:
        if random.random() < 0.3:
            word = word[:-1] + last_key_board[word[-1]]
            return word, True
    j = 0
    if random.random() < 0.3:
        while(j<7):
            try:
                i = random.choice(range(len(word)))
                word = word[:i] + random.choice(key_board[word[i]]) + word[i+1:]
                return word, True
            except:
                j+=1
    else:
        word, _ = typo_error(word)
        i = random.choice(range(len(word)))
        word = word[:i] + random.choice(key_board[word[i]]) + word[i+1:]
        return word, True
    return word, False

def word_same_error(word):
    try:
        return random.choice(confused_word_same_error[word]), True
    except:
        return word, False
    
def add_unknown(word):
    return '<unk>', True

def not_change_word(word):
    return word, True

def not_change_sent(sent):
    return sent, True

def unidecode_span_sent(sent):
    sent = sent.split()
    i = random.choice(range(len(sent)))
    j = random.choice(range(len(sent)))
    if i >= j:
        i,j = j,i
    sent = " ".join(sent[:i]) + " " +  unidecode.unidecode(" ".join(sent[i:j+1])) + " " +  " ".join(sent[j+1:])
    return sent.strip(), True

def typo_span_sent(sent):
    sent = sent.split()
    i = random.choice(range(len(sent)))
    j = random.choice(range(len(sent)))
    if i >= j:
        i,j = j,i
    new_text, _ = typo_error(" ".join(sent[i:j+1]))
    sent = " ".join(sent[:i]) + " " +  new_text + " " +  " ".join(sent[j+1:])
    return sent.strip(), True

def change_sent_span(sent):
    prob = random.random()
#     if prob < 0.1:
    return not_change_sent(sent)
#     elif prob < 0.4:
#         return typo_span_sent(sent)
#     else:
#         return unidecode_span_sent(sent)
prob_create_data = [change_tone_word]* 10 + [change_first_char]*8 + [change_char]*2 + [typo_error] + [add_char]*2 + [remove_char]*2 +[change_char_keyboard]*2 + [not_change_word] * 3
def change_word_once(word):
    response = False
    while (response == False):
        func = random.choice(prob_create_data)
        t, response = func(word)
    return t

def change_word_twice(word):
    while(1):
        try:
            word = change_word_once(word)
            word = change_word_once(word)
            break
        except:
            continue
    return word

def change_word(word, unknown = False):
    if unknown:
        prob = random.random()
        if prob < 0.05:
            return add_unknown(word)
        elif prob < 0.5:
            return change_word_once(word), True
        else:
            return change_word_twice(word), True
    else:
        prob = random.random()
        #if prob < 0.5:
        return change_word_once(word), True
        #else:
        #    return change_word_twice(word), True
def count_word_in_vocab(sent):
    sent = sent.split()
    count = 0
    for w in sent:
        if w in v:
            count +=1
    return count

def change_sent(sent, unknown = False):
    count = count_word_in_vocab(sent)
    if count < 5:
        return sent
    if random.random() < 0.4:
        sent, _ = change_sent_span(sent)
        return sent
    else:
        num_word_wrong = round(count * 0.15)
        rand_1 = random.random()
        if rand_1 < 0.35:
            rand = random.random()
            if rand < 0.1:
                num_word_wrong = 1
            elif rand < 0.2:
                num_word_wrong = 2
            elif rand < 0.4:
                num_word_wrong +=1
            elif rand < 0.6:
                num_word_wrong -=1
            else:
                num_word_wrong = num_word_wrong
        elif rand_1 < 0.70:
            num_word_wrong = random.choice(range(num_word_wrong+1))
        else:
            num_word_wrong = 1
        sent = sent.split()
        index_v = [i for i in range(len(sent)) if sent[i] in v]
        index_word_wrong = []
        while(num_word_wrong>0):
            idx_wrong = random.choice(index_v)
            if idx_wrong not in index_word_wrong:
                index_word_wrong.append(idx_wrong)
                num_word_wrong = num_word_wrong - 1 
        for idx in index_word_wrong:
            word_temp, _ = change_word(sent[idx], unknown)
            if len(word_temp.strip()) == 0:
                sent[idx] = sent[idx]
            else:
                sent[idx] = word_temp
        return " ".join(sent)
