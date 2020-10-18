from ideolog.model import VoteClassifier as vc

classifier = vc.VoteClassifier()
print(classifier.classify_vote("legalize marijuana.", "Maria Cantwell"))
print("best matches: " + classifier.return_best_matches())
