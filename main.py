from langchain.document_loaders import PyMuPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import collections
import numpy as np


def read_pdf(path):
    pdf_obj = PyMuPDFLoader(path)
    pages = pdf_obj.load()
    text = ''
    for page in pages:
        text+=page.page_content
    return text
read_pdf("C:\\Users\\NAGARURU PURUSHOTHAM\\Downloads\\natural_language_processing_tutorial.pdf")


def chunks(data):
    data = data.replace('\n', '')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=20)
    chunk_data = text_splitter.split_text(data)
    return chunk_data

def embeddings(chunk_data):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embedding_data = model.encode(chunk_data)
    return embedding_data

def store_embeddings(file_name : str, chunk_data, embeddings):
    data = collections.defaultdict(list)
    with open(f'{file_name}.pkl', 'wb') as file:
        for index, (chunks, vectors) in enumerate(zip(chunk_data, embeddings)):
            data[index].append(chunks)
            data[index].append(vectors)
        pickle.dump(data, file)
    return data

def read_file(file_name):
    with open(f'{file_name}.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

def get_relevent_text(query_embedding, embedding_vectors,texts, top_n):
      similarities = cosine_similarity([query_embedding], embedding_vectors).flatten()
      top_indices = similarities.argsort()[-top_n:][::-1]
      relevant_texts = [(texts[i], similarities[i]) for i in top_indices]
      return relevant_texts


if __name__ == '__main__':
    # pdf_data = read_pdf("C:\\Users\\NAGARURU PURUSHOTHAM\\Downloads\\natural_language_processing_tutorial.pdf")
    # chunks_data = chunks(pdf_data)
    # print(len(chunks_data))
    # embeddings_data = embeddings(chunks_data)
    # print(len(embeddings_data))
    # print(store_embeddings('test',chunks_data, embeddings_data))
    # print(len(store_embeddings))
    data = read_file('test')
    embedding_vectors = np.array([data[i][1] for i in data])
    text = [data[i][0] for i in data]
    query_vetors = embeddings('abc')
    texts = get_relevent_text(query_vetors, embedding_vectors, text, 3)
    print(texts)
    # print(data)