import numpy as np
import pandas as pd
import pickle
from elasticsearch import Elasticsearch
from elasticsearch import helpers
es = Elasticsearch(http_compress=True)

def populate_data(path_to_publications_dataframe, index_name, keys):
    use_these_keys = keys
    publications_path = path_to_publications_dataframe
    publications_df = pickle.load(open(publications_path, "rb"))
    publications_df = publications_df[use_these_keys]

    def filterKeys(document):
        return {key: document[key] for key in use_these_keys }

    def doc_generator(df):
        df_iter = df.iterrows()
        for index, document in df_iter:
            yield {
                    "_index": index_name,
                    # "_type": "document",
                    "_id" : index,
                    "_source": filterKeys(document),
                }
    helpers.bulk(es, doc_generator(publications_df))

populate_data('assets/publications.p', 'cord19', ['paper_id', 'metadata.title', 'all_body_text', 'publish_datetime'])
populate_data('assets/acl_publications.p', 'acl', ['Year', 'Title', 'Body_Text'])