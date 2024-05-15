from sklearn.feature_extraction.text import CountVectorizer
texts = ['i have a cat', 
        'he has a dog', 
        'he has a dog and i have a cat']

vect = CountVectorizer()
X = vect.fit_transform(texts)
print('words in dictionary: ', vect.get_feature_names())
print(X.toarray())
