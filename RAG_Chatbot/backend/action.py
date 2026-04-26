import json
import os

BOOKINGS_FILE = "bookings.json"

def book_table(name, date, people):
    booking_data = {
        "name": name,
        "date": date,
        "people": people
    }
    
    # Load existing bookings
    bookings = []
    if os.path.exists(BOOKINGS_FILE):
        with open(BOOKINGS_FILE, "r") as f:
            try:
                bookings = json.load(f)
            except json.JSONDecodeError:
                bookings = []
    
    # Add new booking
    bookings.append(booking_data)
    
    # Save back to file
    with open(BOOKINGS_FILE, "w") as f:
        json.dump(bookings, f, indent=4)
        
    return f"Success! Table booked for {name} on {date} for {people} guests. You can see this in bookings.json."