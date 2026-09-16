import json
import os
import pandas as pd
from configs.arg_parser import get_args

args = get_args()

input_dir = args.json_folder_path
output_dir = args.csv_folder_path

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for file_name in os.listdir(input_dir):
    if file_name.endswith('.json'):
        json_file_path = os.path.join(input_dir, file_name)

        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        rows = []

        for anno in data["annotations"]:
            anno_id = anno["id"]
            image = f"{anno['image_id']}.jpg" 
            question = anno["question"]
            answer = ", ".join(anno["answers"])
            
            rows.append([anno_id, image, question, answer])

        df = pd.DataFrame(rows, columns=["anno_id", "image", "question", "answer"])

        csv_file_name = os.path.splitext(file_name)[0] + '.csv'
        csv_file_path = os.path.join(output_dir, csv_file_name)
        
        df.to_csv(csv_file_path, index=False)
        
        print(f"CSV file '{csv_file_name}' has been created successfully.")