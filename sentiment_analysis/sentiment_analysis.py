# import sys 
# sys.path.append("..") 

# from finbert.finbert import predict
# import argparse
# import os
# import pandas as pd
# import numpy as np
# import datetime
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
# date_format = '%Y-%m-%d %H:%M:%S'

# parser = argparse.ArgumentParser(description='Sentiment analyzer')
# parser.add_argument('-a', action="store_true", default=False)
# parser.add_argument('--table_path', type=str, help='Path to the dataset table.')
# parser.add_argument('--output_dir', type=str, help='Where to write the results')
# parser.add_argument('--model_path', type=str, help='Path to classifier model')
# parser.add_argument('--mode', type=str, default="check", help='Path to classifier model')

# args = parser.parse_args()

# if not os.path.exists(args.output_dir):
#     os.mkdir(args.output_dir)

# head, tail = os.path.split(args.table_path)
# output = "predictions_"+tail[:-4]+"csv"
# print("output stored in: ", os.path.join(args.output_dir,output))

# df = pd.read_excel(args.table_path)

# def store(dict, date, score_title, score_summary):
#     dict["date"].append(date)
#     dict["score_title"].append(score_title)
#     dict["score_summary"].append(score_summary)

# def check_before_start():
#     title_df = pd.notna(df["title"])
#     summary_df = pd.notna(df["summary"])
#     print("have empty title", df.index[title_df == False].tolist())
#     print("have empty summary", df.index[summary_df == False].tolist())

# def predict_dataset():
#     # initialize
#     model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
#     # df = df.sort_values(by=['date', 'platform'])
#     last_date = df["date"][0]
#     cul_score_title, cul_score_summary, counter = 0, 0, 0
#     output_dict = {"date": [],
#                 "score_title": [],
#                 "score_summary": []}

#     length = len(df)
#     # length = 10 # for debug testing

#     for i in range(length):
#         date, platform, title, summary = df["date"][i], df["platform"][i], df["title"][i], df["summary"][i]
        
#         # store if comes to a new day
#         if ((last_date) != (date) and counter > 0):
#             store(output_dict, last_date, cul_score_title/counter, cul_score_summary/counter)

#             # initialize again
#             cul_score_title, cul_score_summary, counter = 0, 0, 0

#             # fix the gap if date is not continious
#             prev_date = last_date - datetime.timedelta(days=1)
#             while date < prev_date:
#                 store(output_dict, prev_date, 0, 0)
#                 print("fixed empty date", prev_date)
#                 prev_date -= datetime.timedelta(days=1)

#         try:
#             # get score for the new title and summary
#             score_title = predict(title, model, write_to_csv=False)
#             cul_score_title += score_title
#             score_summary = predict(summary, model, write_to_csv=False)
#             cul_score_summary += score_summary
#             counter += 1
#             last_date = date
#         except Exception as error:
#             print("Error happened in handeling line", i, "of file", args.table_path)
#             print("title: ", title)
#             print("summary: ", summary)
#             print("Error: ", error)
#             # break

#         # showing progress
#         if i % 100 == 0:
#             print("finish", i, "/", length)

#     if counter > 0:
#         store(output_dict, last_date, cul_score_title/counter, cul_score_summary/counter)
#     df_out = pd.DataFrame.from_dict(output_dict)
#     print(df_out)
#     df_out.to_csv(os.path.join(args.output_dir,output))

# if args.mode == "check":
#     check_before_start()
# elif args.mode == "run":
#     predict_dataset()

import os
import torch
import csv
from tqdm import tqdm
from transformers import AlbertTokenizer, AlbertForSequenceClassification
from transformers import XLNetTokenizer, XLNetForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def sentiment_albert(document_path):
    # Load pre-trained model and tokenizer
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    model = AlbertForSequenceClassification.from_pretrained('albert-base-v2')
    
    # Read the document
    with open(document_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Encode and prepare inputs
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract logits
    logits = outputs.logits
    scores = torch.nn.functional.softmax(logits, dim=-1).squeeze().tolist()
    
    return scores

def sentiment_xlnet(document_path):
    # Load pre-trained model and tokenizer
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased')
    
    # Read the document
    with open(document_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Encode and prepare inputs
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract logits
    logits = outputs.logits
    scores = torch.nn.functional.softmax(logits, dim=-1).squeeze().tolist()
    
    return scores

def sentiment_ernie(document_path):
    # Load pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('nghuyong/ernie-2.0-en')
    model = AutoModelForSequenceClassification.from_pretrained('nghuyong/ernie-2.0-en')
    
    # Read the document
    with open(document_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Encode and prepare inputs
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract logits
    logits = outputs.logits
    scores = torch.nn.functional.softmax(logits, dim=-1).squeeze().tolist()
    
    return scores

def load_models():
    models = {
        "albert": "albert-base-v2" ,
        "xlnet": "xlnet-base-cased",
        "ernie": "nghuyong/ernie-2.0-en",
        "bert": "bert-base-uncased",
        "distilbert": "distilbert-base-uncased",
        "roberta": "roberta-base"
    }
    tokenizers_and_models = {}
    for model_name, model_path in models.items():
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizers_and_models[model_name] = (tokenizer, model)
    return tokenizers_and_models

def compute_sentiments(text, tokenizers_and_models):
    results = {}
    for model_name, (tokenizer, model) in tokenizers_and_models.items():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        scores = torch.nn.functional.softmax(logits, dim=-1).squeeze().tolist()
        results[model_name] = scores
    return results

def write_to_csv(company, data, base_path):
    headers = ["Date"]
    model_names = ["albert", "xlnet", "ernie", "bert", "distilbert", "roberta"]
    for model_name in model_names:
        headers.extend([f"{model_name}_positive", f"{model_name}_neutral", f"{model_name}_negative"])

    csv_file = os.path.join(base_path, f"{company}_sentiment.csv")
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for title, scores in data:
            date = '-'.join(title.split("-")[:-1])
            row = [date]
            for model_name in model_names:
                row.extend(scores.get(model_name, [None, None, None]))
            writer.writerow(row)

def process_transcripts(base_path):
    results = {}
    tokenizers_and_models = load_models()

    for root, dirs, files in os.walk(base_path):
        print("stock:", dirs)
        for file in tqdm(files):
            if file.endswith(".txt"):
                company = root.split(os.sep)[-1].upper()  # Get company name from folder
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()

                sentiment_scores = compute_sentiments(text, tokenizers_and_models)

                if company not in results:
                    results[company] = []
                results[company].append((file, sentiment_scores))

    # Write each company's results to its own CSV file
    for company, data in results.items():
        write_to_csv(company, data, base_path)

# Usage
base_path = "./../dataset_transcript/Transcripts"
process_transcripts(base_path)