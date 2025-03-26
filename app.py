import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('stopwords')
nltk.download('punkt_tab')

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def preprocess_text(text):
  text = text.lower()

  text = text.translate(str.maketrans("", "", string.punctuation))

  tokens = word_tokenize(text)

  stop_words = set(stopwords.words('russian'))
  tokens = [word for word in tokens if word not in stop_words]

  return " ".join(tokens)

st.title('Проверка текмта на токсичность')

user_input = st.text_area('Ведите ваш комментарий:', "")

if st.button("Проверить"):
    
    if user_input:
        text = preprocess_text(user_input)
        text_vectorized = vectorizer.transform([text])
        
        prediction = model.predict(text_vectorized)
        
        if prediction == 1:
            st.write('Вы пытаетесь отправить токсичный комментарий!')
        else:
            st.write(f'Ваш комментарий опубликован: {user_input}')