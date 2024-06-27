from fastapi import FastAPI
import nest_asyncio
import uvicorn
from pydantic import BaseModel
from utils import *
from typing import List


from transformers import pipeline

corrector = pipeline("text2text-generation", model="minhbui/spell_correction")

app = FastAPI()
MAX_LENGTH = 512
batch_size = 32

@app.get("/")
async def hello():
    return "Hello"

@app.get("/health")
async def health_check():
    return {"status": "OK"}

class Input(BaseModel):
    id: int
    text: str
    meta_data: dict

class Output(BaseModel):
    index: int
    type: str
    suggestion: str
    
class Correct(BaseModel):
    lst_str: List[str]

@app.post("/check_spell/")
async def check_spell(lst: List[Input]):
    res = []
    lst_text = []
    for i in lst:
        lst_text.append(separate_punctuation_with_space(norm_text(i.text.lower())).strip())
    predictions = corrector(lst_text, max_length=MAX_LENGTH, batch_size=batch_size)
    for item, pred in zip(lst, predictions):
        reverse_mapping, process_text = reverse(item.text)
        result = []
        for idx, (i, j) in enumerate(zip(process_text.split(' '), pred['generated_text'].lower().split(' '))):
            if i != j and "..." not in j:
                result.append({"index": reverse_mapping[idx], "type": "GRAMMAR", "suggestion": j})
        if len(result) > 0:
            res.append({"id": item.id, "text": item.text, "result": result, "meta_data": item.meta_data})
    return res

@app.post("/check_sent/")
async def check_sent(lst: Correct):
    lst.lst_str = [separate_punctuation_with_space(norm_text(x.lower())).strip() for x in lst.lst_str]
    res = {}
    predictions = corrector(lst.lst_str, max_length=MAX_LENGTH, batch_size=batch_size)
    for idx1, (text, pred) in enumerate(zip(lst.lst_str, predictions)):
        # print(text, pred)
        text = separate_punctuation_with_space(norm_text(text.lower())).strip()
        try:
            lst_err_pos = []
            detail = []
            for idx, (i,j) in enumerate(zip(text.lower().split(' '), pred['generated_text'].lower().split(' '))):
                if i != j:
                    if "..." in j:
                        continue
                    lst_err_pos.append(idx)
                    detail.append(i+" -> "+j)
            if len(lst_err_pos) > 0:
                res[idx1] = {"sentence": text, "generated":pred['generated_text'], "error_index" : lst_err_pos, "suggestion":detail}
        except Exception as e:
            print(e)
            continue
    return {"status": "Success", "data" : res}

