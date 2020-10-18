import EmbeddingGenerator as eg
import pandas as pd
import torch


class VoteClassifier:
    def __init__(self):
        self.embedder = eg.EmbeddingGenerator()

    def get_legislator_id(self):
        '''Gets '''

    def classify_vote(self, prompt, legislator):
        # retrieve the ID of the legislator
        legislators = pd.read_csv('./data/house_table.csv')
        leg_id = legislators[legislators['name'] == legislator].iloc[0]['id']

        # get a set of bills which the legislator voted on
        vote_history = pd.read_csv('./data/vote_table.csv')
        bill_df = vote_history.loc[vote_history['person'] == leg_id]

        # load the embeddings of all the bills the legislator voted on
        yeas = set()
        nays = set()
        for index, row in bill_df.iterrows():
            # add embeddings to the right set
            try:
                tensor = torch.load('data/tensors/' + row['bill'] + ".pt")
                if row['vote'] == 1:
                    yeas.add((tensor, row['bill']))
                else:
                    nays.add((tensor, row['bill']))
            except Exception:
                continue

        # use embedder to turn the prompt into embeddings
        prompt_embed = self.embedder.get_embeddings(prompt, willSave=False)
        prompt_embed = torch.flatten(prompt_embed)

        # determine whether the prompt embedding is more similar to the bills voted yea on
        # or the bills voted nay on
        best_sim = 1
        best_class = None
        closest_bill = None
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        for bill in yeas:
            flat = torch.flatten(bill[0])
            sim = cos(flat, prompt_embed)
            if sim < best_sim:
                best_class = 'yea'
                closest_bill = bill[1]
        for bill in nays:
            flat = torch.flatten(bill[0])
            sim = cos(flat, prompt_embed)
            if sim < best_sim:
                best_class = 'nay'
                closest_bill = bill[1]

        return best_class, closest_bill


