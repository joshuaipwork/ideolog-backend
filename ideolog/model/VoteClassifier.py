import EmbeddingGenerator as eg
import pandas as pd
import torch
import json

torch.cuda.set_device(0)

class VoteClassifier:
    """ VoteClassifier is a class which predicts the way a legislator will vote based on their past records. """

    def __init__(self):
        self.embedder = eg.EmbeddingGenerator()
        self.best_matches = []
        self.reset_best_matches()

        self.legislators = pd.read_csv('../data/house_table.csv')
        self.vote_history = pd.read_csv('../data/vote_table.csv')

    def reset_best_matches(self):
        ''' Used internally to reset the record of the best matches '''
        self.best_matches = [('', 1, ''), ('', 1, ''), ('', 1, ''), ('', 1, ''), ('', 1, '')]

    def classify_vote(self, prompt, legislator):
        '''
        This function predicts the way a legislator will vote on a prompt. Before running, you must use
        EmbeddingsGenerator to get embeddings for all the bills that will be included in the legislator's record.

        :param prompt: a string provided by the user which outlines an position on a policy issue.
            Examples include "Obamacare must be repealed" and "medicare must be provided for all"
        :param legislator: must match the legislator's name on house_table.csv
        :return: a prediction 'yea' or 'nay' on how the legislator will vote on a bill with that issue.
        '''
        # retrieve the ID of the legislator
        legislators = self.legislators
        leg_id = legislators[legislators['name'] == legislator].iloc[0]['id']

        # get a set of bills which the legislator voted on
        vote_history = self.vote_history
        bill_df = vote_history.loc[vote_history['person'] == leg_id]

        # load the embeddings of all the bills the legislator voted on
        yeas = []
        nays = []
        for index, row in bill_df.iterrows():
            # add embeddings to the right set
            try:
                tensor = torch.load('../data/tensors/' + row['bill'] + ".pt")
                if row['vote'] == 1:
                    yeas.append((tensor, row['bill']))
                else:
                    nays.append((tensor, row['bill']))
            except Exception:
                continue

        # use embedder to turn the prompt into embeddings
        prompt_embed = self.embedder.get_embeddings(prompt, willSave=False)
        prompt_embed = torch.flatten(prompt_embed)

        # determine whether the prompt embedding is more similar to the bills voted yea on
        # or the bills voted nay on
        self.reset_best_matches()

        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        for bill in yeas:
            flat = torch.flatten(bill[0])
            sim = cos(flat, prompt_embed)
            sim = sim.item()
            if sim < self.best_matches[0][1]:
                self.best_matches[0] = (bill[1], sim, 'yea')
                self.best_matches.sort(key=lambda x: -x[1])
                print(self.best_matches)

        for bill in nays:
            flat = torch.flatten(bill[0])
            sim = cos(flat, prompt_embed)
            sim = sim.item()
            if sim < self.best_matches[0][1]:
                self.best_matches[0] = (bill[1], sim, 'nay')
                self.best_matches.sort(key=lambda x: -x[1])
                print(self.best_matches)

        print(self.best_matches)
        return self.best_matches[-1][2]

    def return_best_matches(self):
        ''' returns a json string which represents the best five matches
            of the last query. '''
        result = []
        bills = pd.read_csv('../data/bill_table.csv')

        for i in range(5):
            if self.best_matches[i][0]:
                entry = {}
                entry['name'] = self.best_matches[i][0]
                entry['vote'] = self.best_matches[i][2]
                entry['summary'] = bills[bills['id'] == self.best_matches[i][0]].iloc[0]['summary']
                # score = self.best_match_scores[i]
                # if score < 0.1:
                #     entry['confidence'] = 'high confidence'
                # elif 0.1 <= score < 0.25:
                #     entry['confidence'] = 'moderate confidence'
                # else:
                #     entry['confidence'] = 'low confidence'
                result.append(entry)

        return json.dumps(result)
