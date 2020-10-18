import VoteClassifier as vc
import pandas as pd

classifier = vc.VoteClassifier()
print(classifier.classify_vote("This bill bans abortion under all circumstances.", "Maria Cantwell"))

