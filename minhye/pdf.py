import base64
from collections import defaultdict
import io
import os
import pickle
import re

from PIL import Image
from tqdm.notebook import tqdm
from unstructured.partition.pdf import partition_pdf


# 定数定義
DATA_PAR_PATH = os.path.join('..','..','data')
INPUT_DATA_PATH = os.path.join(DATA_PAR_PATH,'安全なウェブサイトの作り方.pdf')
OUTPUT_DATA_PATH = os.path.join(DATA_PAR_PATH,'output.pkl')

