"""
Airport data for major Indian airports
"""

def get_airports():
    """Get list of all airports with their details"""
    return [{"code": code, "name": info["name"], **info} for code, info in INDIAN_AIRPORTS.items()]

def get_airport_info(code):
    """Get detailed information for a specific airport"""
    return INDIAN_AIRPORTS.get(code, {})

def get_all_airport_data():
    """Get complete airport dataset"""
    return INDIAN_AIRPORTS

INDIAN_AIRPORTS = {
    "DEL": {
        "name": "Indira Gandhi International Airport",
        "city": "Delhi",
        "lat": 28.5562,
        "lon": 77.1000,
        "elevation": 237,
        "runway_directions": ["09L/27R", "09R/27L", "11/29"]
    },
    "BOM": {
        "name": "Chhatrapati Shivaji Maharaj International Airport",
        "city": "Mumbai",
        "lat": 19.0896,
        "lon": 72.8656,
        "elevation": 11,
        "runway_directions": ["09/27", "14/32"]
    },
    "BLR": {
        "name": "Kempegowda International Airport",
        "city": "Bangalore",
        "lat": 13.1986,
        "lon": 77.7066,
        "elevation": 920,
        "runway_directions": ["09/27", "16/34"]
    },
    "HYD": {
        "name": "Rajiv Gandhi International Airport",
        "city": "Hyderabad",
        "lat": 17.2313,
        "lon": 78.4298,
        "elevation": 532,
        "runway_directions": ["09L/27R", "09R/27L"]
    },
    "MAA": {
        "name": "Chennai International Airport",
        "city": "Chennai",
        "lat": 12.9941,
        "lon": 80.1709,
        "elevation": 16,
        "runway_directions": ["07/25", "12/30"]
    },
    "CCU": {
        "name": "Netaji Subhash Chandra Bose International Airport",
        "city": "Kolkata",
        "lat": 22.6546,
        "lon": 88.4467,
        "elevation": 5,
        "runway_directions": ["01L/19R", "01R/19L"]
    }
}

# Common flight routes between Indian airports
INDIAN_ROUTES = [
    {"origin": "DEL", "destination": "BOM", "distance": 1138, "typical_altitude": 37000},
    {"origin": "DEL", "destination": "BLR", "distance": 1740, "typical_altitude": 39000},
    {"origin": "DEL", "destination": "HYD", "distance": 1296, "typical_altitude": 37000},
    {"origin": "DEL", "destination": "MAA", "distance": 1759, "typical_altitude": 39000},
    {"origin": "DEL", "destination": "CCU", "distance": 1318, "typical_altitude": 37000},
    {"origin": "BOM", "destination": "BLR", "distance": 843, "typical_altitude": 35000},
    {"origin": "BOM", "destination": "HYD", "distance": 617, "typical_altitude": 33000},
    {"origin": "BOM", "destination": "MAA", "distance": 1027, "typical_altitude": 35000},
    {"origin": "BOM", "destination": "CCU", "distance": 1648, "typical_altitude": 39000},
    {"origin": "BLR", "destination": "HYD", "distance": 499, "typical_altitude": 31000},
    {"origin": "BLR", "destination": "MAA", "distance": 290, "typical_altitude": 29000},
    {"origin": "BLR", "destination": "CCU", "distance": 1561, "typical_altitude": 37000},
    {"origin": "HYD", "destination": "MAA", "distance": 504, "typical_altitude": 31000},
    {"origin": "HYD", "destination": "CCU", "distance": 1185, "typical_altitude": 37000},
    {"origin": "MAA", "destination": "CCU", "distance": 1366, "typical_altitude": 37000}
]

def get_airport_info(airport_code):
    """Get detailed information about an airport"""
    return INDIAN_AIRPORTS.get(airport_code.upper())

def get_all_airports():
    """Get all airport codes and names"""
    return [(code, info["name"]) for code, info in INDIAN_AIRPORTS.items()]

def get_routes_for_airport(airport_code):
    """Get all routes involving a specific airport"""
    routes = []
    for route in INDIAN_ROUTES:
        if route["origin"] == airport_code or route["destination"] == airport_code:
            routes.append(route)
    return routes

def calculate_great_circle_distance(lat1, lon1, lat2, lon2):
    """Calculate great circle distance between two points"""
    import math
    
    # Convert latitude and longitude to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Radius of earth in kilometers
    r = 6371
    
    return c * r
