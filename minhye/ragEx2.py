from collections import defaultdict
import io
import os
import pickle

from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_ollama import OllamaEmbeddings, ChatOllama
from PIL import Image
from tqdm.notebook import tqdm

import settings as sets


# 定数定義
DATA_PAR_PATH = os.path.join('..','..','data')
INPUT_DATA_PATH = os.path.join(DATA_PAR_PATH,'output.pkl')

NEO4J_URI: str = sets.NEO4J_URI
NEO4J_USER: str = sets.NEO4J_USER
NEO4J_PASSWORD: str = sets.NEO4J_PASSWORD

MODEL_NAME = 'elyza:8b'
CHUNK_SIZE = 500

with open(INPUT_DATA_PATH, 'rb') as rf:
    elements = pickle.load(rf)

# データを確認してみる
page_no = 1
for elem in elements[page_no]:
    cat = elem['category']
    data = elem['data']
    print(cat)
    
    if cat == 'image':
        binary_image = io.BytesIO(data)
        image = Image.open(binary_image)
        display(image)
    else:
        print(data)

embeddings = OllamaEmbeddings(model=MODEL_NAME)

total = len(elements.keys())
embed_info = []
text = ''
reference_text = ''

def add_embed_info(text):
    text = text.rstrip('\n')
    embed_text = embeddings.embed_query(text)  # str -> vector(dim=4096)
    embed_info.append(
        {
            'text': text,
            'reference_text': reference_text,
            'embedding': embed_text
        }
    )

for page_no in tqdm(elements.keys(), total=total):    
    for elem in elements[page_no]:
        cat = elem['category']
        data = elem['data']
        
        if cat == 'text':
            detail_cat = elem['detail']
            
            if detail_cat == 'Header':
                if len(text) == 0:
                    continue
                
                add_embed_info(text)
                
                text = data
                reference_text = data
            elif len(text) < CHUNK_SIZE:
                text += data + '\n'
            else:
                add_embed_info(text)
                
                text = data

if len(text) > 0:
    add_embed_info(text)

# データの確認
embed_info[1]

%%time

text_embeddings = [(e['text'], e['embedding']) for e in embed_info]
metadatas = [{'reference_text': e['reference_text']} for e in embed_info]

graph_db = Neo4jVector.from_embeddings(
    text_embeddings=text_embeddings,
    embedding=embeddings,
    metadatas=metadatas,
    url=NEO4J_URI,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD
)
