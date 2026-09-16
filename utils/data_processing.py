import pandas as pd
from underthesea import word_tokenize, text_normalize

from configs.arg_parser import get_args

def process_dataframe(df):
    df['quesion'] = [word_tokenize(text_normalize(x), format='text') for x in df['question']]
    df['answer'] = [word_tokenize(text_normalize(str(x)), format='text') for x in df['answer']]
    return df

def preprocess_data(args):
    train_csv_path = args.train_csv_path
    test_csv_path = args.test_csv_path
    dev_csv_path = args.dev_csv_path
    # Load data
    df_train = pd.read_csv(train_csv_path)
    df_dev = pd.read_csv(dev_csv_path)
    df_test = pd.read_csv(test_csv_path)
    
    # Preprocess data

    df_train = process_dataframe(df_train)
    df_dev = process_dataframe(df_dev)

    return df_train, df_dev, df_test

if __name__=="__main__":
    args = get_args()

    df_train, df_dev, df_test = preprocess_data(args)
    print(df_train.head())