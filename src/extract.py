import csv

def extract_data(file_path="data/sample_data.csv"):
    # Generator function that reads one row at a time.
    
    with open(file_path, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            yield row