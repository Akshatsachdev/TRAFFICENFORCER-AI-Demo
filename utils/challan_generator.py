import pandas as pd
from datetime import datetime


def log_challan(plate, helmet_status):
    data = {
        "Plate Number": [plate],
        "Helmet Status": [helmet_status],
        "Timestamp": [datetime.now()]
    }

    df = pd.DataFrame(data)

    file_path = "data/challan_details.xlsx"

    try:
        existing = pd.read_excel(file_path)
        df = pd.concat([existing, df], ignore_index=True)
    except:
        pass

    df.to_excel(file_path, index=False)
