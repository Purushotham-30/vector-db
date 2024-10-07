import re
import os
from chromadb import PersistentClient
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter


def read_pdf(path):
    with pdfplumber.open(path) as file:
        page_list = file.pages
        text = ""
        for page in page_list:
            text += page.extract_text()
        return text

def create_chunks(text):
    splitter =  RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap = 20)
    list_sentences = splitter.split_text(text)
    sentence_list = []
    for sentence in list_sentences:
        sentence = re.sub(r"\\u[e-f][0-9a-z]{3}|\n", " ", sentence)
        sentence_list.append(re.sub(r'[^\w\s]', '', sentence.lower()))
    return sentence_list

def chroma(query):
    text = read_pdf("document.pdf")
    sentences = create_chunks(text)
    client = PersistentClient("/data")
    collection = client.get_or_create_collection(name="my_collection")
    collection.upsert(documents=sentences, ids=[f'id{idx}' for idx in range(1, len(sentences)+1)])
    results = collection.query(query_texts=query,
                     n_results=3)

    return results["documents"]


if __name__ == '__main__':
    result = chroma("explain corpora")
    

