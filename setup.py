import re
import collections
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import numpy as np
from annoy import AnnoyIndex
import pickle

def read_pdf(path):
    with pdfplumber.open(path) as file:
        page_list = file.pages
        text = ""
        for page in page_list:
            text += page.extract_text()
        return text

def create_chunks(text):
    text_spiltter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    list_sentences = text_spiltter.split_text(text)
    sentence_list = []
    for sentence in list_sentences:
        sentence = re.sub(r"\\u[e-f][0-9a-z]{3}|\n", " ", sentence)
        sentence_list.append(re.sub(r'[^\w\s]', '', sentence.lower()))
    return sentence_list

def create_embeddings(list_text : list[str]):
    tokens = [simple_preprocess(sentence, deacc=True, min_len=1, max_len=20) for sentence in list_text]
    model = Word2Vec(tokens, vector_size=100, window=5, min_count=1, workers=5)
    list_vectors = []
    for text in list_text:
        if text:
            vectors = np.mean([model.wv[word] for word in text.split() if word in model.wv], axis=0)
        else:
            vectors = np.zeros(model.vector_size)
        list_vectors.append(vectors)
    return list_vectors, model
 
def create_index(dimension):
    return AnnoyIndex(dimension, 'angular')

def store_vector_embeddings(index, list_sentence, embeddings, index_name = 'annoy_index.ann', pickle_file = "test_purush.pkl"):
    list_sentences_emneddings = collections.defaultdict(list)
    for i, (text, embed) in enumerate(zip(list_sentence, embeddings)):
        list_sentences_emneddings[i].append(text)
        list_sentences_emneddings[i].append(embed)
        index.add_item(i, embed)
    index.build(n_trees=10)
    index.save(index_name)
    with open(pickle_file, 'wb') as file:
        pickle.dump(list_sentences_emneddings, file)
    
def query_embeddings(query, model):
    tokens = query.lower().split()
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if vectors:
        mean_vectors = np.mean(vectors, axis=0)
    else:
        mean_vectors = np.zeros(model.vector_size)   
    return mean_vectors

def query(question, top_results, index : AnnoyIndex, model, index_name = 'annoy_index.ann'):
    index.load(index_name)
    embeddings = query_embeddings(question, model)
    neighbours = index.get_nns_by_vector(vector=embeddings, n=top_results, include_distances=True)
    return neighbours

def get_result(neighbours):
    with open("test_purush.pkl", 'rb') as file :
        data = pickle.load(file)
    text = ""
    for idx, dist in zip(*neighbours):
        text+=data[idx][0]
    return text

def main(pdf, question, top_values = 3):
    text = read_pdf(pdf)
    chunks = create_chunks(text)
    embed, model = create_embeddings(chunks)
    index = create_index(len(embed[0]))
    store_vector_embeddings(index, chunks, embed)
    neighbours = query(question, top_values, index, model)
    print(neighbours)
    result = get_result(neighbours)
    print(result)
    
if __name__ == '__main__':
    main("document.pdf", "explain corpora")
