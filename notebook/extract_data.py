import zipfile
import pandas as pd

# Step 1: Extract the zip
with zipfile.ZipFile("car-dataset.zip", "r") as z:
    z.extractall("car_data")  # folder ka naam ham kuch bhi rakh sakta hai

# Step 2: Load CSV
df = pd.read_csv("car_data/Car Sell Dataset.csv")
print(df.head())
import pandas as pd

# Load the CSV file from the data folder
df = pd.read_csv("data/data.csv")

# Print first 5 rows to confirm it's working
print(df.head())


# note that this code is changed after