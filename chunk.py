import base64
from collections import defaultdict
import io
import os
import pickle
import re

from PIL import Image
from tqdm.notebook import tqdm
from unstructured.partition.pdf import partition_pdf
from IPython.display import display


# 定数定義
DATA_PAR_PATH = os.path.join('..','..','data')
INPUT_DATA_PATH = os.path.join(DATA_PAR_PATH,'安全なウェブサイトの作り方.pdf')
OUTPUT_DATA_PATH = os.path.join(DATA_PAR_PATH,'output.pkl')

res = defaultdict(list)

# %%time

raw_pdf_elements = partition_pdf(
    languages=['jpn'],
    filename=INPUT_DATA_PATH,
    infer_table_structure=True,
    strategy='hi_res',
    extract_images_in_pdf=True,
    extract_image_block_types=['Image', 'Table'],
    extract_image_block_to_payload=True
)

for elem in tqdm(raw_pdf_elements):
    page_no = elem.metadata.page_number
    cat = elem.category
    print(f"{page_no = }")
    
    if cat in ['Image', 'Table']:
        image_base64 = elem.metadata.image_base64
        decoded_image = base64.b64decode(image_base64)
        binary_image = io.BytesIO(decoded_image)
        image = Image.open(binary_image)
        display(image)

        res[page_no].append(
            {
                'category': 'image',
                'data': decoded_image  # バイナリ形式で出力させる
            }
        )
    else:
        text = elem.text
        cleaned_text = text.replace(' ', '')

        # URLは除去
        url_pattern = 'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+'
        text_without_url = re.sub(url_pattern, '', cleaned_text)
        
        print(text_without_url)

        res[page_no].append(
            {
                'category': 'text',
                'detail': cat,
                'data': text_without_url
            }
        )
with open(OUTPUT_DATA_PATH, 'wb') as wf:
    pickle.dump(res, wf)
