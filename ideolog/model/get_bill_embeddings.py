from ideolog.model import EmbeddingGenerator as eg
import pandas as pd

'''A simple script for generating the embeddings of the summaries of every bill in the bill_table.csv'''

bills = pd.read_csv('../../data/bill_table.csv')
embedder = eg.EmbeddingGenerator()

for index, bill in bills.iterrows():
    if isinstance(bill['summary'], str):
        print(embedder.get_embeddings(bill['summary'], id=bill['id'], willSave=True).shape)
        print(bill['id'])
