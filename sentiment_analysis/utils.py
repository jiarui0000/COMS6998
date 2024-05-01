
import pandas as pd
import numpy as np
import os
import glob

def dataloader():
    dfs = []
    labels = []
    pdf = pd.DataFrame()
    # for company in ['AAPL', 'AMD', 'AMZN', 'ASML','CSCO', 'GOOGL', 'INTC', 'MSFT', 'NVDA']:

    companies = ['AAPL', 'AMZN', 'GOOGL', 'MSFT', 'NVDA']
    for company in companies:
        df_numerical = pd.read_csv('../dataset_transcript/EPS/'+company+'_EPS.csv')
        df_textural = pd.read_csv('../dataset_transcript/Transcripts/'+company+'_sentiment.csv')
        df_comment = pd.read_csv('../dataset_transcript/Comments/predictions_'+company+'.csv', index_col=0)
        df_index = pd.read_csv('../dataset_transcript/Stock Values and Sector Index/NASDAQ.csv')
        df_result = pd.read_csv('../dataset_transcript/Stock Values and Sector Index/'+company+'.csv')
        df_numerical['Date'] = pd.to_datetime(df_numerical['Date'])
        df_textural['Date'] = pd.to_datetime(df_textural['Date'])
        df_comment['Date'] = pd.to_datetime(df_comment['Date'])
        df_index['Date'] = pd.to_datetime(df_index['Date'])
        df_result['Date'] = pd.to_datetime(df_result['Date'])
        df_numerical = df_numerical.sort_values(by='Date').reset_index(drop=True)
        df_textural = df_textural.sort_values(by='Date').reset_index(drop=True)
        df_comment = df_comment.sort_values(by='Date').reset_index(drop=True)
        df_index = df_index.sort_values(by='Date').reset_index(drop=True)
        df_result = df_result.sort_values(by='Date').reset_index(drop=True)

        df_combined = pd.merge(df_textural, df_numerical, how="left", on=["Date"])
        df_combined = pd.merge(df_combined, df_index, how="left", on=["Date"])
        df_combined = pd.merge(df_combined, df_comment, how="left", on=["Date"]).fillna(0)

        mask = df_result['Date'].isin(df_combined['Date'])
        indices_of_interest = df_result.index[mask]
        
        previous_indices = indices_of_interest - 1
        previous_indices = previous_indices[previous_indices >= 0]
        df_previous = df_result.iloc[previous_indices].copy().reset_index(drop=True)
        df_previous = df_previous.rename(columns={col: 'prev_' + col for col in df_previous.columns})
        df_previous['Date'] = df_result.iloc[indices_of_interest].reset_index(drop=True)['Date']
        
        df_combined = pd.merge(df_combined, df_previous, how="left", on=["Date"])
        df_combined.fillna(0, inplace = True) 

        df_result = pd.merge(df_result, df_combined['Date'], how='right', on=["Date"])
        df_combined=df_combined.drop(columns=['Fiscal Quarter End', 'Date', 'prev_Date'])
        pdf = df_combined
        dfs.append(df_combined)
        labels.append(df_result['Close'])
    print("labels[0].shape: ", labels[0].shape)
    print("dfs[0].shape: ", dfs[0].shape)

    return dfs, labels, pdf, dfs[0].shape[1], companies

def binary_accuracy(predict_change, truth):
    y_prediction = predict_change / np.abs(predict_change)
    diff2 = np.array(truth[1:])-np.array(truth[:-1])
    truth_labels = diff2/ np.abs(diff2)

    negative = -1.0
    positive = 1.0

    tp = np.sum(np.logical_and(y_prediction == positive, truth_labels == positive))
    tn = np.sum(np.logical_and(y_prediction == negative, truth_labels == negative))
    fp = np.sum(np.logical_and(y_prediction == positive, truth_labels == negative))
    fn = np.sum(np.logical_and(y_prediction == negative, truth_labels == positive))
    s = tp+tn+fp+fn
    return tp/s, tn/s, fp/s, fn/s

def setup(path):
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

    files = glob.glob(path+'/*')
    for f in files:
        os.remove(f)


def create_sequences(X, Y, window_size=3):
    X_new, Y_new = [], []
    if len(X) < window_size + 1:
        print("Insufficient data points for the given window size.")
        return np.array(X_new), np.array(Y_new)
    for i in range(len(X) - window_size):
        X_new.append(X[i:i + window_size])
        percentage_change = (Y[i + window_size] - Y[i + window_size - 1]) / Y[i + window_size - 1]
        Y_new.append(percentage_change)
    return np.array(X_new), np.array(Y_new)
