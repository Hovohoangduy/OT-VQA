import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training the model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--img_path", type=str, default="data/images/st_images", help="Path to image folder")
    parser.add_argument("--train_csv_path", type=str, default="data/csv/ViTextVQA_train.csv", help="Path to training CSV file")
    parser.add_argument("--test_csv_path", type=str, default="data/csv/ViTextVQA_test.csv", help="Path to testing CSV file")
    parser.add_argument("--dev_csv_path", type=str, default="data/csv/ViTextVQA_dev.csv", help="Path to development CSV file")
    parser.add_argument("--model_path", type=str, default="data", help="Path to save trained model")
    parser.add_argument("--json_folder_path", type=str, default="data/json", help="Path to folder containing JSON files")
    parser.add_argument("--csv_folder_path", type=str, default="data/csv", help="Path to folder where CSV files will be saved")
    
    return parser.parse_args()