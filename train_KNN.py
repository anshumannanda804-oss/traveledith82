import sqlite3
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import joblib
import os
import sqlite3

import sqlite3
import os

db_path = os.path.abspath("users.db")
print("Database:", db_path)

conn = sqlite3.connect( r"C:\Users\ANSHUMAN\OneDrive\Desktop\FRONTEND TRAVEL\backend\users.db")

cursor = conn.cursor()

cursor.execute("""
SELECT name
FROM sqlite_master
WHERE type='table'
""")

print("Tables:", cursor.fetchall())

# Read booking data
df = pd.read_sql_query(
    "SELECT hotel_name, travelers, days, total_price FROM hotel_bookings",
    conn
)

conn.close()

print(df.head())
if len(df) < 5:
    print("Not enough booking data for KNN training")
    exit()
# Prepare data for KNN
X = df[["travelers", "days", "total_price"]]
y = df["hotel_name"]
# Train KNN model
knn = NearestNeighbors(n_neighbors=3)
knn.fit(X)
# Save the model
joblib.dump(knn, "knn_model.pkl")
print("Total Bookings:", len(df))
print("KNN model trained and saved as knn_model.pkl")
