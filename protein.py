import pandas as pd
import numpy as np

def step0():
    df = pd.read_parquet('data.parquet')
    pl = list(set(df['seq']))

    seq = [i.upper() for i in pl]
    protein = list(set(seq))
    df = pd.DataFrame({
        'seq':protein
    })
    df['index'] = df.index
    df.to_csv('seqIndex.csv', sep=',', index= False)
step0()

def step1():
    data = pd.read_table('seqIndex.csv', sep=',') 
    with open("protein.txt",'a') as f:
        gap = min(list(data['reindex']))
        for i,d in enumerate(data['seq']):
            f.write('>proteins%s' % (i + gap) + '\n')
            f.write(d.upper() + '\n')
step1()

import json
def save_json(save_path,data):
    assert save_path.split('.')[-1] == 'json'
    with open(save_path,'w') as file:
        json.dump(data,file)
 
def load_json(file_path):
    assert file_path.split('.')[-1] == 'json'
    with open(file_path,'r') as file:
        data = json.load(file)
    return data

# different json data format
def IndexSeq():
    data = pd.read_table('seqIndex.csv', sep=',')
    s = list(data['seq'])
    dic = dict()
    gap = min(list(data['reindex']))
    for i in range(len(s)):
        dic[i+gap] = s[i]
    save_json("IndexSeq.json", dic)
IndexSeq()

def SeqIndex(): # step3
    dic = dict()
    data = load_json('IndexSeq.json')
    for key,value in data.items():
        dic[value] = key
    save_json("SeqIndex.json", dic)####改生成的文件命名
SeqIndex()