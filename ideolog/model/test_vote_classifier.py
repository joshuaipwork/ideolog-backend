from ideolog.model import VoteClassifier as vc

classifier = vc.VoteClassifier()
#print(classifier.classify_vote("legalize marijuana.", "Maria Cantwell"))
print(classifier.classify_all_legislators("Prohibit abortion under all circumstances."))
#print("best matches: " + classifier.return_best_matches())
