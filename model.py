from ..generate_embeddings import EmbeddingGenerator
import pandas as pd
import torch


class VoteClassifier:
    def __init__(self):
        self.embedder = EmbeddingGenerator()

    def get_legislator_id(self):
        '''Gets '''

    def classify_vote(self, prompt, legislator):
        # retrieve the ID of the legislator
        legislators = pd.load_csv('./house_table.csv')
        leg_id = legislators[legislators['name'] == legislator].iloc[0]['id']

        # if the legislator doesn't exist, TODO: Do something
        # if legislators.empty:

        # get a set of bills which the legislator voted on
        vote_history = pd.load_csv('./data/vote_table.csv')
        bill_df = vote_history.loc[vote_history['person'] == leg_id]

        # load the embeddings of all the bills the legislator voted on
        yeas = set()
        nays = set()
        for index, row in bill_df.iterrows():
            # add embeddings to the right set
            tensor = torch.load('data/tensors/' + row['bill'])
            if row['vote'] == 1:
                yeas.add(tensor)
            else:
                nays.add(tensor)

        # use embedder to turn the prompt into embeddings
        prompt_embed = self.embedder.get_embeddings(prompt, willSave=False)

        # determine whether the prompt embedding is more similar to the bills voted yea on
        # or the bills voted nay on


