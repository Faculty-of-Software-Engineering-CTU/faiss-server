import time
import numpy as np
from embedding import Embedding
from vector_store import VectorStore

start_time = time.time()

embedd_model = Embedding()

text = "Hiệu trưởng trường Đại học Cần Thơ 2024 đến nay"

result = embedd_model.embed_query(text)

vector_store = VectorStore()