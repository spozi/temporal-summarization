#%% Flask tutorial
from elasticsearch import Elasticsearch
from elasticsearch import helpers
from flask_sqlalchemy import SQLAlchemy
es = Elasticsearch(http_compress=True)
from flask import Flask

app = Flask(__name__)

@app.route('/')

def home():
    return "Hey there!"

if __name__ == '__main__':
    app.run(debug=True)