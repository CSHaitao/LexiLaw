#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: create_knowledge.py
@time: 2023/04/18
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: - emoji：https://emojixd.com/pocket/science
"""
import os
import pandas as pd
from langchain.schema import Document
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from tqdm import tqdm
# 中文Wikipedia数据导入示例：
embedding_model_name = './text2vec'
docs_path = './cache/legal_article'
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)



docs = []

for doc in tqdm(os.listdir(docs_path)):
    if doc.endswith('.txt'):

        f=open(f'{docs_path}/{doc}','r',encoding='utf-8')


        for line in f.readlines():

            docs.append(Document(page_content=''.join(line.strip()), metadata={"source": f'doc_id_{doc}'}))

vector_store = FAISS.from_documents(docs, embeddings)
vector_store.save_local('./cache/legal_article')



