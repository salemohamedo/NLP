import copy
import sys
import numpy
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
lmtzr = WordNetLemmatizer()

## For feature template 2  - POS tags
## only accepts adverbs/adjectives/nouns/conjunctions
pos_accepted = ["RB", "RBS", "RBR", "NN", "NNS", "CC", "JJ", "JJR", "JJS"]


## OPEN TRAIN/DEV FILES ##

with open('sst3/sst3.train') as train:
	train_lines = train.readlines()
	train_lines = [x.strip() for x in train_lines] 

with open('sst3/sst3.dev') as dev:
	dev_lines = dev.readlines()
	dev_lines = [x.strip() for x in dev_lines] 

with open('sst3/sst3.devtest') as devtest:
	devtest_lines = devtest.readlines()
	devtest_lines = [x.strip() for x in devtest_lines] 


labels = [0, 1, 2]

## HELPER FUNCTIONS ##

## create feature hashmap with all weights set to 0
def populate_features(train_corpus):
	features = {}
	for line in train_corpus:
		parts = line.split("\t")
		y = parts[1]
		words = parts[0].split(" ")
		for word in words:
			feature = word + "$%$" + y
			if(feature not in features):
				features[feature] = float(0)
	return features


## feature template 1 - lemmas

def populate_lemma_features(train_corpus):
	features = {}
	lemma_map = {}
	for line in train_corpus:
		parts = line.split("\t")
		y = parts[1]
		words = parts[0].split(" ")
		for word in words:
			root_word = lmtzr.lemmatize(word)
			lemma_map[word] = root_word
			feature = root_word + "$%$" + y
			if(feature not in features):
				features[feature] = float(0)
	return (features, lemma_map)

## feature template 2 - pos
def populate_pos_features(train_corpus):
	features = {}
	for line in train_corpus:
		parts = line.split("\t")
		y = parts[1]
		words = parts[0]
		tokens = word_tokenize(words)
		tokens_annotated = pos_tag(tokens)
		relevant_words = []
		for token in tokens_annotated:
			pos = token[1]
			if pos in pos_accepted:
				relevant_words.append(token[0])
		for word in relevant_words:
			feature = word + "$%$" + y		
			if(feature not in features):
				features[feature] = float(0)
	return features

## if hinge == true, calculated hinge loss
## else calculate perceptron loss
def classify(words, gold_standard, hinge, features, lemma_map, mode):
	scores = [0.0, 0.0, 0.0]
	for label in labels:
		score = float(0)
		for word in words:
			if(mode == 1):
				if(word in lemma_map):
					root_word = lemma_map[word]
				else:
					root_word = word
				feature = root_word + "$%$" + str(label)
			else:
				feature = word + "$%$" + str(label)
			if feature in features:
				score += features[feature]
		if hinge:
			if label != gold_standard:
				score += 1
		scores[label] = score
	max_score = max(scores)
	argmax = scores.index(max_score)
	return argmax


## test accuracy on dev/devtest
def test_accuracy(dev_corpus, features, mode, lemma_map, report_errors):
	num_correct = 0
	for line in dev_corpus:
		words = line.split("\t")[0].split(" ")
		gold_standard = int(line.split("\t")[1])
		predicted_label = classify(words, gold_standard, False, features, lemma_map, mode)
		if(gold_standard == predicted_label):
			num_correct += 1
		else:
			if report_errors:
		  		print(line + " " + str(predicted_label)) 
	return float(float(num_correct) / float(len(dev_corpus)))

## return top x features for each label
def top_features(features, x):
	desc_sort = sorted(features, key=features.get, reverse=True)
	top = {0:[],1:[],2:[]}
	for label in labels:
		count = 0
		for feature in desc_sort:
			if(count >= x):
				break
			else:
				feature_label = int(feature.split("$%$")[1])
				if(feature_label == label):
					top[label].append(feature.split("$%$")[0] + str(features[feature]))
					count += 1
	return top

def avg_feature_weight(features):
	return numpy.mean(list(features.values()))

## run classifier on given feature template mode.
## mode 0 - standard unigram binary features
## mode 1 - lemmatize training examples
## mode 2 - only create features for certain POS
def run_classifier(epochs, test_interval, hinge, mode, report_errors):
	num_examples = 0
	best_dev = {}
	lemma_map = {}
	best_dev_accuracy = float(0)
	best_devtest_accuracy = float(0)
	## initialize empty feature map with weights = 0
	if(mode == 0): ##standard feature template
		features = populate_features(train_lines)
	elif(mode == 1): ##lemmatize feature template
		(features, lemma_map) = populate_lemma_features(train_lines)
	elif(mode == 2): ##POS feature template
		features = populate_pos_features(train_lines)
	for i in range(0,epochs):
		for line in train_lines:

			##calculate accuracies every 20,000 examples

			if((num_examples != 0) and ((num_examples%test_interval) == 0)):
				new_dev_accuracy = test_accuracy(dev_lines, features, mode, lemma_map, False)
				if(new_dev_accuracy > best_dev_accuracy):
					best_dev.clear()
					best_dev = copy.deepcopy(features)
					best_dev_accuracy = new_dev_accuracy
					new_devtest_accuracy = test_accuracy(devtest_lines, features, mode, lemma_map, report_errors)
					if(new_devtest_accuracy > best_devtest_accuracy):
						best_devtest_accuracy = new_devtest_accuracy

			## extract gold and predicted labels

			words = line.split("\t")[0].split(" ")
			gold_standard = int(line.split("\t")[1])
			predicted_label = classify(words, gold_standard, hinge, features, lemma_map, mode)

			## every word in each training example combined
			## with its gold standard is a feature, and combined
			## with the predicted label may also be a feature

			for word in words:
				relevant_features = []
				if(mode == 1):
					root_word = lemma_map[word]
					gold_feature = root_word + "$%$" + str(gold_standard)
					predicted_feature = root_word + "$%$" + str(predicted_label)
				else:
					gold_feature = word + "$%$" + str(gold_standard)
					predicted_feature = word + "$%$" + str(predicted_label)
				if(gold_feature in features):
					relevant_features.append(gold_feature)
				if((predicted_feature in features) and (predicted_feature != gold_feature)):
					relevant_features.append(predicted_feature)

				## calculate subgradient perceptron/hinge loss
				## and update respective weights
					
				for feature in relevant_features:
					current_weight = features[feature]
					feature_label = int(feature.split("$%$")[1])
					f_1 = 0
					f_2 = 0
					if(feature_label == gold_standard):
						f_1 = 1
					if(feature_label == predicted_label):
						f_2 = 1
					new_weight = current_weight + (0.01 * f_1) - (0.01 * f_2)
					features[feature] = new_weight

			num_examples += 1

		##reset number of examples at end of epoch

		num_examples = 0

		##calculate accuracies at end of an epoch

		new_dev_accuracy = test_accuracy(dev_lines, features, mode, lemma_map, False)
		if(new_dev_accuracy > best_dev_accuracy):
			best_dev.clear()
			best_dev = copy.deepcopy(features)
			best_dev_accuracy = new_dev_accuracy
			new_devtest_accuracy = test_accuracy(devtest_lines, features, mode, lemma_map, report_errors)
			if(new_devtest_accuracy > best_devtest_accuracy):
				best_devtest_accuracy = new_devtest_accuracy
	print({"max_dev_accuracy": best_dev_accuracy, "max_devtest_accuracy": best_devtest_accuracy})
	return best_dev


## OUTPUT FOR REPORT ##


# OUTPUT FOR 1.1

print("BEST PERCEPTRON ACCURACIES")
run_classifier(20, 20000, False, 0, False)

# OUTPUT FOR 1.2

print("BEST HINGE ACCURACIES")
best_hinge = run_classifier(20, 20000, True, 0, False)

# OUTPUT FOR 1.3

top_10 = top_features(best_hinge, 10)
print(top_10)

# OUTPUT FOR 1.4

# sets error reporting to true
run_classifier(20, 20000, True, 0, True)

# OUTPUT FOR 1.5

print("BEST FEATURE 1 TEMP HINGE ACCURACIES")
run_classifier(20, 20000, True, 1, False)


print("BEST FEATURE 2 TEMP HINGE ACCURACIES")
run_classifier(20, 20000, True, 2, False)


