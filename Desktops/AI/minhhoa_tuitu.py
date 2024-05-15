from functools import reduce
import numpy as np
# Đầu vào là một texts bao gồm 3 câu văn:
texts = [['i', 'have', 'a', 'cat'], 
        ['he', 'has', 'a', 'dog'], 
        ['he', 'he','has', 'a', 'dog', 'and', 'i', 'have', 'a', 'cat']]
# B1: Xây dựng từ điển
dictionary = list(enumerate(set(reduce(lambda x, y: x + y, texts))))
# B2: Mã hoá câu sang véc tơ tần suất
def bag_of_word(sentence):
    # Khởi tạo một vector có độ dài bằng với từ điển.
    vector = np.zeros(len(dictionary))
    # Đếm các từ trong một câu xuất hiện trong từ điển.
    for i, word in dictionary:
        count = 0
        # Đếm số từ xuất hiện trong một câu.
        for w in sentence:
            if w == word:
                count += 1
        vector[i] = count
    return vector
            
for i in texts:
    print(bag_of_word(i))
