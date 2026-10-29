import sqlite3

conn = sqlite3.connect(
    r"C:\Users\ANSHUMAN\OneDrive\Desktop\FRONTEND TRAVEL\backend\users.db"
)

cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM hotel_bookings")
print("Total Bookings:", cursor.fetchone()[0])

cursor.execute("SELECT * FROM hotel_bookings")
print(cursor.fetchall())

conn.close()