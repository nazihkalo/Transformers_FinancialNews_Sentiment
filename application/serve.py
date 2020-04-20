# from finbert .ner_model import NERModel
# from model.config import Config
#from model.utils import align_data
from __future__ import absolute_import, division, print_function
import csv

import numpy as np
from pytorch_pretrained_bert.modeling import *
from pytorch_pretrained_bert.optimization import *
from bert_sentiment_utils import predict
from textblob import TextBlob

def align_data(data):
    """Given dict with lists, creates aligned strings
    Adapted from Assignment 3 of CS224N
    Args:
        data: (dict) data["x"] = ["I", "love", "you"]
              (dict) data["y"] = ["O", "O", "O"]
    Returns:
        data_aligned: (dict) data_align["x"] = "I love you"
                           data_align["y"] = "O O    O  "
    """
    spacings = [max([len(seq[i]) for seq in data.values()])
                for i in range(len(data[list(data.keys())[0]]))]
    data_aligned = dict()

    # for each entry, create aligned string
    for key, seq in data.items():
        str_aligned = ""
        for token, spacing in zip(seq, spacings):
            str_aligned += token + " " * (spacing - len(token) + 1)

        data_aligned[key] = str_aligned

    return data_aligned

model = BertForSequenceClassification.from_pretrained('finbert-sentiment',cache_dir=None,  num_labels=3)

# text = "Later that day Apple said it was revising down its earnings expectations in \
# the fourth quarter of 2018, largely because of lower sales and signs of economic weakness in China. \
# The news rapidly infected financial markets. Apple’s share price fell by around 7% in after-hours \
# trading and the decline was extended to more than 10% when the market opened. The dollar fell \
# by 3.7% against the yen in a matter of minutes after the announcement, before rapidly recovering \
# some ground. Asian stockmarkets closed down on January 3rd and European ones opened lower. \
# Yields on government bonds fell as investors fled to the traditional haven in a market storm."
# prediction  = predict(text, model)
# print(prediction.to_dict())

# prediction.size

# string_dict = prediction.apply(lambda x: str(x)).to_dict()
# output_data = align_data(string_dict)

def get_model_api():
    """Returns lambda function for api"""
    # 1. initialize model once and for all and reload weights
    model = BertForSequenceClassification.from_pretrained('finbert-sentiment',cache_dir=None,  num_labels=3)
    
    def model_api(input_data):
        #Get custom model prediction
        prediction  = predict(input_data, model)
        #Add textblob prediction to the dataframe
        blob = TextBlob(input_data)
        prediction['textblob_prediction'] = [sentence.sentiment.polarity for sentence in blob.sentences]
        output_data = prediction.astype(str) 
        # 4. process the output
        output_data = {"input": str(input_data), "output": output_data} #align_data(output_data)
    
        # 5. return the output for the api
        return output_data

    return model_api

#Output needs to be dictionary
#value = string
