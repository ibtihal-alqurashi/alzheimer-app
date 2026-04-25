import csv
from datetime import datetime

def save_result(label, confidence):
    file_exists = False

    try:
        open("results.csv", "r")
        file_exists = True
    except:
        file_exists = False

    with open("results.csv", "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["time", "prediction", "confidence"])

        writer.writerow([
            datetime.now(),
            label,
            confidence
        ])