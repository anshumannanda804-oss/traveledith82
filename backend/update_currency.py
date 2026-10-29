import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_NAME = os.path.join(BASE_DIR, "users.db")

updates = [
    ('Paris', 'EUR'),
    ('London', 'GBP'),
    ('Dubai', 'AED'),
    ('Singapore', 'SGD'),
    ('Bangkok', 'THB'),
    ('Bali', 'IDR'),
    ('Tokyo', 'JPY'),
    ('New York', 'USD'),
    ('Los Angeles', 'USD'),
    ('Barcelona', 'EUR'),
    ('Rome', 'EUR'),
    ('Sydney', 'AUD'),
    ('Amsterdam', 'EUR'),
    ('Maldives', 'MVR'),
    ('Iceland', 'ISK')
]

conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()

for name, curr in updates:
    cursor.execute("UPDATE destinations SET currency = ? WHERE name = ?", (curr, name))

conn.commit()
conn.close()
print("Currencies updated successfully.")
