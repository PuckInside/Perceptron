from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "Кошка спит на солнце",
    "Собака играет на улице",
    "На улице люди гуляют с собаками"
]

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(documents)

feature_names = vectorizer.get_feature_names_out()

tfidf_array = tfidf_matrix.toarray()

print("Фичи (Слова):", feature_names)
print("TF-IDF Матрица:")
print(tfidf_array)