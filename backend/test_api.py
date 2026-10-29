import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app

with app.test_request_context(json={"source":"Delhi","destination":"London","days":3,"numbersOfPerson":1}):
    from app import trip_details
    response = trip_details()
    print("Response Status:", response.status_code)
    print("Response JSON:", response.get_json())
