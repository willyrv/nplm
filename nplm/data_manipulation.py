import numpy as np
import pandas as pd
import re

def one_hot_encode_vector(x, dict_size):
    a = np.array(x)
    v = np.zeros((dict_size, len(x)))
    v[a, np.arange(len(x))] = 1
    return v

def one_hot_encode_matrix(A, dict_size):
    return np.array([one_hot_encode_vector(x, dict_size)
                        for x in A])

class dictionary(dict):
    """
    Extends python dictionary in order to have
    index --> word
    but also
    word --> index
    """
    def __init__(self):
        super(dictionary, self).__init__()
        self.index = {}
        self.size = 0
    
    def __setitem__(self, key, value):
        super(dictionary, self).__setitem__(key, value)
        self.index[value] = key
        self.size += 1
    
    def __delitem__(self, key):
        value = super().pop(key)
        ignore = self.index.pop(value)
        self.size -=1

def process_corpus(corpus, context_size, dictionary, fixed_dictionary=False):
    list_of_points = []
    for document in corpus:
        list_of_points += process_document(document, context_size, dictionary, fixed_dictionary)
    return list_of_points


def process_document(document, context_size, dictionary, fixed_dictionary=False):
    """
    Given a dictionary, extract the tuples of words of length equal to
    context_size. Each word is represented by a unique integer number.
    If fixed_dictionary is True, only take consecutive tuples of words 
    being (all of them) in the dictionary.
    Example: 
        document = "This is a new document"
        context_size = 4
        dictionary = {
            0: "this",
            1: "is",
            2: "a",
            3: "new",
            4: "document"
        }
        
        return
            [(0, 1, 2, 3), (1, 2, 3, 4)]
    """
    text = document.lower()
    p = re.compile("[a-z]+")
    tokens = p.findall(text)
    list_of_points = []
    for i in range(len(tokens) - context_size + 1):
        data_point = [0 for l in range(context_size)]
        add_new_data_point = True
        for j in range(context_size):
            k = i+j
            if tokens[k] not in dictionary.index:
                if fixed_dictionary:
                    # only takes series of words in the dictionary
                    add_new_data_point = False
                    break
                else:
                    new_Ix = dictionary.size
                    dictionary[new_Ix] = tokens[k]
            data_point[j] = dictionary.index[tokens[k]]
        if add_new_data_point:
            list_of_points.append(tuple(data_point))
    return list_of_points

def create_training_dataset_arXiv(path2file, context_size, dict_size):
    """
    Create the training data set from the arXiv data for a given
    context_size and dict_size
    """
    data = pd.read_csv("./arxiv_articles.csv", sep="|")
    mydict = dictionary()
    # Process the corpus one first time to create the dictionary
    dataset = process_corpus(data['summary'], context_size, mydict)
    data_df = pd.DataFrame(dataset)
    word_counts = data_df.iloc[:, 0].value_counts()
    # We sort the words, starting from the most frequent ones
    words2keep = word_counts.keys()[:dict_size]
    # Create a new clean dictionary with the
    # words selected in the previous step
    new_dictionary = dictionary()
    for i in range(len(words2keep)):
        new_dictionary[i] = mydict[words2keep[i]]
    # Build the new training dataset using the new dictionary
    # and the series of context_size words appearing in the text
    new_dataset = process_corpus(data['summary'], context_size, new_dictionary, fixed_dictionary=True)
    return new_dataset, new_dictionary





