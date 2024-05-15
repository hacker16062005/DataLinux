from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
 	'tôi thích ăn bánh mì nhân thịt',
	'cô ấy thích ăn bánh mì, còn tôi thích ăn xôi',
	'thị trường chứng khoán giảm làm tôi lo lắng',
	'chứng khoán sẽ phục hồi vào thời gian tới. danh mục của tôi sẽ tăng trở lại',
  'dự báo thời tiết hà nội có mưa vào chiều và tối. tôi sẽ mang ô khi ra ngoài'
]

# Tính tfidf cho mỗi từ. max_df để loại bỏ stopwords xuất hiện ở hơn 90% các câu
vectorizer = TfidfVectorizer(max_df = 0.9)
# Tokenize các câu theo tfidf
X = vectorizer.fit_transform(corpus)
print('words in dictionary:')
print(vectorizer.get_feature_names())
print('X shape: ', X.shape)
