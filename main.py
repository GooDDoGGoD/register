from flask import request, Flask
import os
import onnxruntime as rt
from transformers import DistilBertTokenizer
import numpy as np


app = Flask(__name__)
posts = []

tokenizer = DistilBertTokenizer.from_pretrained(
    'distilbert-base-uncased'
)
model = rt.InferenceSession(
    'models/model-quantized.onnx',
    providers=['CPUExecutionProvider']
)


def proverka(text):
    m_i = tokenizer(text, truncation=True, padding=True)
    m_i = {k: np.array([v]) for k, v in m_i.items()}
    return model.run(None, m_i)[0][0].argmax()


@app.route('/dialog', methods=['GET'])
def dialog():
    return '<br>'.join(posts)


@app.route('/add', methods=['GET'])
def add():
    post = str(request.query_string).replace('%20', ' ')[2:-1]
    if proverka(post) == 1:
        posts.append(' '.join(['*' * len(x) for x in post.split()]))
        return '<h1>Сообщение удаленно</h1>'
    posts.append(post)
    return '<h1>Сообщение отправленно</h1>'


app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 500)))
