from docx import Document
import random
from underthesea import ner, pos_tag
import pandas as pd
from tqdm import tqdm
import os
import unicodedata
import re

def norm_text(text):
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r"òa", "oà", text)
    text = re.sub(r"óa", "oá", text)
    text = re.sub(r"ỏa", "oả", text)
    text = re.sub(r"õa", "oã", text)
    text = re.sub(r"ọa", "oạ", text)
    text = re.sub(r"òe", "oè", text)
    text = re.sub(r"óe", "oé", text)
    text = re.sub(r"ỏe", "oẻ", text)
    text = re.sub(r"õe", "oẽ", text)
    text = re.sub(r"ọe", "oẹ", text)
    text = re.sub(r"ùy", "uỳ", text)
    text = re.sub(r"úy", "uý", text)
    text = re.sub(r"ủy", "uỷ", text)
    text = re.sub(r"ũy", "uỹ", text)
    text = re.sub(r"ụy", "uỵ", text)
    text = re.sub(r"Ủy", "Uỷ", text)
    return text


with open("all-vietnam.txt" , "r", encoding='utf-8')  as f:
	vocab = f.readlines()
	vocab = [word.replace("\n","") for word in vocab]
	vocab = [norm_text(word) for word in vocab]

def check(s):
	s2 = norm_text(s)
	tmp = s2.split(' ')
	for idx, i in enumerate(tmp):
		if i not in vocab:
			tmp[idx] = '<oov>'
	res = ' '.join(x for x in tmp)
	return s, res


BANG_XOA_DAU_FULL = str.maketrans(
    "ÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴáàảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ",
    "A"*17 + "D" + "E"*11 + "I"*5 + "O"*17 + "U"*11 + "Y"*5 + "a"*17 + "d" + "e"*11 + "i"*5 + "o"*17 + "u"*11 + "y"*5,
    chr(774) + chr(770) + chr(795) + chr(769) + chr(768) + chr(777) + chr(771) + chr(803) # 8 kí tự dấu dưới dạng unicode chuẩn D
)

def xoa_dau_full(txt: str) -> str:
    return txt.translate(BANG_XOA_DAU_FULL)

from_char = "àáãảạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệđùúủũụưừứửữựòóỏõọôồốổỗộơờớởỡợìíỉĩịäëïîöüûñçýỳỹỵỷ"
to_char   = "aaaaaăăăăăăââââââeeeeeêêêêêêđuuuuuưưưưưưoooooôôôôôôơơơơơơiiiiiaeiiouuncyyyyy"


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

def check_syll_vn(txt):
	if norm_text(txt) in vocabs:
		return True
	else:
		return False
# bỏ dấu 1 từ random
def random_remove_accent(text_src, text_des, index, thresh_hold=1):
	texts = split_word_with_bound(text_src)
	i = index
	if (check_syll_vn(texts[i]) and remove_accent(texts[i]) != texts[i]):
		prob = random.random()
		if prob < thresh_hold:
			texts[i] = remove_accent(texts[i])

	return ' '.join(texts), text_des

error = ['capital_error','spell_error','remove_tone','remove_char','add_char','swap_char','typo','duplicate_char']

error_addition = ["name","place"]


lst_change_tone = [
	['a','ă','â','á','à','ả','ã','ạ','ắ','ằ','ẳ','ẵ','ặ','ấ','ầ','ẩ','ẫ','ậ'],
	['d','đ'],
	['e','ê','é','è','ẻ','ẽ','ẹ','ế','ề','ể','ễ','ệ'],
	['i','í','ì','ỉ','ĩ','ị'],
	['u','ư','ú','ù','ủ','ũ','ụ','ứ','ừ','ử','ữ','ự'],
	['o','ô','ơ','ó','ò','ỏ','õ','ọ','ố','ồ','ổ','ỗ','ộ','ớ','ờ','ở','ỡ','ợ'],
	['y','ý','ỳ','ỷ','ỹ','ỵ']
]

def duplicate_position(s, pos, n):
	if pos < 0 or pos > len(s):
		return s
	if n < 1:
		return s
	return s[:pos] + s[pos] * n + s[pos+1:]

def gen_err(i):
	if len(i) < 2:
		return i
	choice = random.sample(error,1)[0]
	print(choice)
	if choice == 'capital_error':
		print("input : ",i)
		if i[0].isupper():
			print("output : ",i.lower())
			return i.lower()
		else:
			print("output : ",i.capitalize())
			return i.capitalize()
	elif choice == 'remove_tone':
		print("input : ",i)
		print("output : ",xoa_dau_full(i))
		return xoa_dau_full(i)
	elif choice == 'spell_error':
		print("input : ",i)
		change_idx = []
		for idx, j in enumerate(i):
			for g in lst_change_tone:
				if j in g:
					change_idx.append(idx)
		if len(change_idx) == 0:
			return i
		c = random.sample(change_idx,1)[0]
		for g in lst_change_tone:
			if i[c] in g:
				r = [x for x in g if x != i[c]]
				r2 = random.sample(r,1)[0]
				i = i[:c] + r2 + i[c+1:]
		print("output : ",i)
		return i
	elif choice == 'remove_char':
		print("input : ",i)
		n_remove = random.sample([1,2],1)[0]
		if len(i)-1 < n_remove:
			return i
		else:
			tmp = list(i)
			pos = random.sample([x for x in range(len(tmp))], n_remove)
			tmp = [x for idx, x in enumerate(tmp) if idx not in pos]
			res = ''.join(v for v in tmp)
			print("output : ", res)
			return res 
	elif choice == 'add_char':
		keyboard = "qwertyuiop\[\]';lkjhgfdsazxcvbnm,./QWERTYUIOP\{\}\":LKJHGFDSAZXCVBNM<>?"
		print("input : ",i)
		pos = random.randint(0, len(i)-1)
		tmp = list(i)
		c = tmp[pos]
		try:
			k = keyboard.index(c)
		except:
			return i
		new_c = random.sample([1,-1],1)[0]
		c = keyboard[k+new_c]
		res = tmp[:pos] + [c] + tmp[pos+1:]
		res = ''.join(g for g in res)
		print("output : ",res)
		return res
	elif choice == 'swap_char':
		print("input : ",i)
		tmp = list(i)
		pos = random.sample([x for x in range(len(tmp))], 2)
		v = tmp[pos[0]]
		tmp[pos[0]] = tmp[pos[1]]
		tmp[pos[1]] = v
		res = ''.join(v for v in tmp)
		print("output : ", res)
		return res 
	elif choice == 'typo':
		print("input : ",i)
		tmp = list(i)
		for idx, j in enumerate(tmp):
			if j in list(typo.keys()):
				res = tmp[:idx] + typo[j] + tmp[idx+1:]
				res = ''.join(g for g in res)
				print("output : ", res)
				return res
		return i

	elif choice == 'duplicate_char':
		print("input : ",i)
		pos = random.randint(0,len(i)-1)
		r = random.randint(2,3)
		res = duplicate_position(i,pos, r)
		print("output : ", res)
		return res

	return ""

def generate(text, rate=0.15, addition_err=True):
	text = text.strip()
	tmp = text.split(' ')
	lst_pos = [0]
	label = [0] * len(tmp)
	for i in tmp[:-1]:
		lst_pos.append(len(i)+lst_pos[-1]+1)

	if addition_err:
		output_err = ner(text, deep=True)
		for i in output_err:
			if i['start'] in lst_pos:
				if i['word'][0].isupper():
					i['word'] = i['word'].lower()
					idx = lst_pos.index(i['start'])
					tmp[idx] = i['word']
					label[idx] = 1
				else:
					continue

	
	for idx, i in enumerate(tmp):
		rand = random.random()
		if rand < rate:
			r = gen_err(i)
			# print(i,r)
			if r == i:
				continue
			tmp[idx] = r
			label[idx] = 1
	
	res = ' '.join(x for x in tmp)
	return text, res, label

n_gen = 5
lst_gt = []
lst_gen = []
lst_label = []

lst_file = os.listdir('./')
for file in tqdm(lst_file):
	# print(file)
	if not file.endswith('.docx') or '~' in file:
		continue
	document = Document(file)
	for i in document.paragraphs:
		for j in range(n_gen):
			if len(i.text) < 5:
				continue
			text, tmp, label = generate(i.text)
			lst_gt.append(i.text)
			lst_gen.append(tmp)
			lst_label.append(label)


df = pd.DataFrame({"text":lst_gt,'generate':lst_gen,"label":lst_label})
print(df.shape)
df.to_excel('gen_err.xlsx',index=False)
