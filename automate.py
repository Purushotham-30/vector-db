import pandas as pd
from chroma_db import chroma
from main import query_process
import collections

questions = ["explain corpora",
             'explain Natural language Processing',
             'what is verbnet',
             'what is wordnet',
            'history of NLP',
            'explain Machine Translation Phase',
            'explain lexical ambiguity',
            'explain Semantic Ambiguity',
            'explain corpus']

q_a = collections.defaultdict(list)

for question in questions:
    q_a[question].append(chroma(question))
    q_a[question].append(query_process(question))

data = pd.DataFrame(q_a)

trans = data.transpose()

trans.to_excel('result.xlsx', header=False)
print("sucess")
