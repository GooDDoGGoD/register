from flask import request, Flask
import os
import onnxruntime as rt
from transformers import DistilBertTokenizer
import numpy as np
import urllib


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
    outp = model.run(None, m_i)
    return outp[0][0].argmax()


@app.route('/dialog', methods=['GET'])
def dialog():
    return '<br>'.join(posts)


@app.route('/add', methods=['GET'])
def add():
    post = str(request.query_string).replace('%20', ' ')[2:-1]
    post = urllib.parse.unquote(post)
    if proverka(post) == 0:
        posts.append('*** ' * len(post))
        return f'<h1>Сообщение "{post}" удаленно</h1>'
    posts.append(post)
    return f'<h1>Сообщение "{post}" отправленно</h1>'


app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
