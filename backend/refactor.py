import os

app_path = r"c:\Users\ANSHUMAN\OneDrive\Desktop\FRONTEND TRAVEL\backend\app.py"

with open(app_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
skip = False

# Replace imports and GDACS/KNN at the top (lines 6 to 90)
import_started = False
for i, line in enumerate(lines):
    if i == 5:  # line 6 (0-indexed 5)
        new_lines.append("from flask import Flask, request, jsonify\n")
        new_lines.append("from flask_cors import CORS\n")
        new_lines.append("import sqlite3\n")
        new_lines.append("import requests\n")
        new_lines.append("import re\n")
        new_lines.append("import random\n")
        new_lines.append("import pandas as pd\n")
        new_lines.append("import numpy as np\n")
        new_lines.append("from sklearn.ensemble import RandomForestClassifier\n")
        new_lines.append("from sklearn.neural_network import MLPClassifier\n\n")
        new_lines.append("from database import init_db\n")
        new_lines.append("from currency import convert_currency\n")
        new_lines.append("from currency import currency_symbol as get_symbol\n\n")
        new_lines.append("from travel_utils import (\n")
        new_lines.append("    fetch_tomtom_hotels,\n")
        new_lines.append("    recommend_hotels_knn,\n")
        new_lines.append("    has_airport,\n")
        new_lines.append("    calculate_distance,\n")
        new_lines.append("    estimate_transport_cost,\n")
        new_lines.append("    HOTEL_SEARCH_RADIUS_KM,\n")
        new_lines.append("    normalize_place_name\n")
        new_lines.append(")\n")
        new_lines.append("from imd import fetch_india_sachet_alerts\n")
        new_lines.append("from gdacs import get_gdacs_alerts\n\n")
        skip = True
    
    if i == 90:  # line 91 starts with `# ---------------- DEEP LEARNING`
        skip = False
    
    if i == 250: # `def calculate_distance` starts at 251 (0-indexed 250)
        skip = True
        
    if i == 817: # `# ---------------- REGISTER ----------------` starts at 818 (0-indexed 817)
        skip = False

    if not skip:
        new_lines.append(line)

with open(app_path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("app.py successfully pruned!")
