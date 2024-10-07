import re
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
import numpy as np
from annoy import AnnoyIndex
import pickle
import collections
import random

class DocumentRead:

    def __init__(self, path : str):
        self.path = path
    
    def read_pdf(self):
        with pdfplumber.open(self.path) as file:
            page_list = file.pages
            text = ""
            for page in page_list:
                text += page.extract_text()
            return text
        
class TextSplitter:

    def __init__(self, splitter : object | None = None):
        if splitter is None:
            self.splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        
    def create_chunks(self, text):
        text_spiltter = self.splitter
        list_sentences = text_spiltter.split_text(text)
        sentence_list = []
        for sentence in list_sentences:
            sentence = re.sub(r"\\u[e-f][0-9a-z]{3}|\n", " ", sentence)
            sentence_list.append(re.sub(r'[^\w\s]', '', sentence.lower()))
        return sentence_list

class Embeddings:
    def __init__(self, sentences : list[str] = None):
        self.sentences = sentences
        tokens = None
        if sentences is not None:
            tokens = [simple_preprocess(sentence, deacc=True, min_len=1, max_len=20) for sentence in self.sentences]
        self.model = Word2Vec(tokens, vector_size=100, window=5, min_count=1, workers=5)
        self.model.save('word2vec.samplemodel')

    def create_embeddings(self):
        list_vectors = []
        for text in self.sentences:
            if text:
                vectors = np.mean([self.model.wv[word] for word in text.split() if word in self.model.wv], axis=0)
            else:
                vectors = np.zeros(self.model.vector_size)
            list_vectors.append(vectors)
        return list_vectors
    
    def query_embeddings(self, query : str):
        self.model = Word2Vec.load('word2vec.samplemodel')
        tokens = query.lower().split()
        vectors = [self.model.wv[word] for word in tokens if word in self.model.wv]
        if vectors:
            mean_vectors = np.mean(vectors, axis=0)
        else:
            mean_vectors = np.zeros(self.model.vector_size)   
        return mean_vectors

class StoreEmbeddings:
    _name = None

    def __init__(self, file_name = None) -> None:
        if file_name is None:
            self.__class__._name = f'pickle_test{random.randrange(1, 100)}.pkl'
        else:
            self.__class__._name = file_name

    @classmethod
    def store_(cls, data):
        with open(cls._name, 'wb') as file:
            pickle.dump(data, file)
    
    @classmethod
    def read_(cls):
        with open(cls._name, 'rb') as file :
            data = pickle.load(file)
        return data

class Index:
    def __init__(self, dimension : int, index_name = None, pickle_file_name = None):
        self.index = AnnoyIndex(dimension, 'angular')
        if index_name is None:
            self.index_name = f'annoy_index{random.randrange(1, 1000)}.ann'
        else:
            self.index_name = index_name
        if pickle_file_name is None:
            self.pickle_file = StoreEmbeddings()
        else:
            self.pickle_file = StoreEmbeddings(pickle_file_name)
    
    def store_vector_embeddings(self, list_sentence, embeddings):
        list_sentences_emneddings = collections.defaultdict(list)
        for i, (text, embed) in enumerate(zip(list_sentence, embeddings)):
            list_sentences_emneddings[i].append(text)
            list_sentences_emneddings[i].append(embed)
            self.index.add_item(i, embed)
        self.index.build(n_trees=10)
        self.index.save(self.index_name)
        self.pickle_file.store_(list_sentences_emneddings)

    def query(self, question, embedding_instance, top_results):
        self.index.load(self.index_name)
        if isinstance(embedding_instance, Embeddings):
            embeddings = embedding_instance.query_embeddings(question)
            neighbours = self.index.get_nns_by_vector(vector=embeddings, n=top_results, include_distances=True)
            return neighbours
        else:
            raise Exception("class mismatch")
        
    def result(self, neighbours):
        data = self.pickle_file.read_()
        text = ""
        for idx, dist in zip(*neighbours):
            text+=data[idx][0]
        return text
    
def process():
    obj = DocumentRead("document.pdf")
    text = obj.read_pdf()
    splitter_obj = TextSplitter()
    sentences = splitter_obj.create_chunks(text)
    embedding_obj = Embeddings(sentences)
    embeddings = embedding_obj.create_embeddings()
    index = Index(dimension=len(embeddings[0]))
    index.store_vector_embeddings(sentences, embeddings)
    neighbour = index.query("explain corpora", embedding_obj, 3)
    text = index.result(neighbour)
    print(text)
    
def get_result_through_query(query):
    embedding_obj = Embeddings()
    index = Index(embedding_obj.model.vector_size, index_name='annoy_index429.ann', pickle_file_name='pickle_test67.pkl')
    neighbour = index.query(query, embedding_obj, 3)
    print(neighbour)
    text = index.result(neighbour)
    print(text)

if __name__ == '__main__':
    process()
    