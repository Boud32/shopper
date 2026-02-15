import json
import random

# Seed Data
categories = {
    "Wireless Headphones": [
        {"title": "Sony WH-1000XM5", "base_price": 348.00, "tags": ["Noise Cancelling", "Top Rated"]},
        {"title": "Bose QuietComfort 45", "base_price": 329.00, "tags": ["Comfort", "Traveling"]},
        {"title": "Apple AirPods Max", "base_price": 549.00, "tags": ["Premium", "Apple"]},
        {"title": "Sennheiser Momentum 4", "base_price": 349.95, "tags": ["Audiophile", "Battery Life"]},
        {"title": "Anker Soundcore Space Q45", "base_price": 149.99, "tags": ["Budget", "Value"]},
        {"title": "Sony WF-1000XM5", "base_price": 299.99, "tags": ["Earbuds", "Compact"]},
        {"title": "Bose QC Earbuds II", "base_price": 279.00, "tags": ["Earbuds", "ANC"]},
        {"title": "Apple AirPods Pro 2", "base_price": 249.00, "tags": ["ios", "transparency"]},
        {"title": "Google Pixel Buds Pro", "base_price": 199.99, "tags": ["Android", "Assistant"]},
        {"title": "Samsung Galaxy Buds2 Pro", "base_price": 229.99, "tags": ["Samsung", "360 Audio"]},
        {"title": "Bowers & Wilkins Px7 S2", "base_price": 399.00, "tags": ["Luxury", "Design"]},
        {"title": "Bang & Olufsen Beoplay HX", "base_price": 499.00, "tags": ["Luxury", "Premium Materials"]},
        {"title": "Master & Dynamic MW75", "base_price": 599.00, "tags": ["High-End", "Build Quality"]},
        {"title": "Jabra Elite 85h", "base_price": 249.99, "tags": ["Durable", "Calls"]},
        {"title": "Audio-Technica ATH-M50xBT2", "base_price": 199.00, "tags": ["Studio", "Flat Response"]},
        {"title": "Beats Studio Pro", "base_price": 349.99, "tags": ["Bass", "Fashion"]},
        {"title": "Shure AONIC 50 Gen 2", "base_price": 349.00, "tags": ["Pro Audio", "Wired/Wireless"]},
        {"title": "Focal Bathys", "base_price": 799.00, "tags": ["Hi-Fi", "DAC Mode"]},
        {"title": "1More SonoFlow", "base_price": 79.99, "tags": ["Budget", "LDAC"]},
        {"title": "Edifier W820NB Plus", "base_price": 59.99, "tags": ["Budget", "Wired"]}
    ],
    "Smartwatches": [
        {"title": "Apple Watch Series 9", "base_price": 399.00, "tags": ["iOS", "Health"]},
        {"title": "Apple Watch Ultra 2", "base_price": 799.00, "tags": ["Rugged", "Dive"]},
        {"title": "Samsung Galaxy Watch 6", "base_price": 299.99, "tags": ["Android", "Sleep"]},
        {"title": "Google Pixel Watch 2", "base_price": 349.99, "tags": ["Fitbit", "Sleek"]},
        {"title": "Garmin Fenix 7 Pro", "base_price": 799.99, "tags": ["Multisport", "Solar"]},
        {"title": "Garmin Forerunner 965", "base_price": 599.99, "tags": ["Running", "AMOLED"]},
        {"title": "Garmin Venu 3", "base_price": 449.99, "tags": ["Lifestyle", "Voice"]},
        {"title": "Fitbit Charge 6", "base_price": 159.95, "tags": ["Tracker", "Lightweight"]},
        {"title": "Fitbit Sense 2", "base_price": 249.95, "tags": ["Stress", "ECG"]},
        {"title": "Amazfit GTR 4", "base_price": 199.99, "tags": ["Value", "Battery"]},
        {"title": "Xiaomi Smart Band 8", "base_price": 49.99, "tags": ["Budget", "Basic"]},
        {"title": "Withings ScanWatch 2", "base_price": 349.95, "tags": ["Hybrid", "Classic"]},
        {"title": "Suunto Race", "base_price": 449.00, "tags": ["Maps", "Adventure"]},
        {"title": "Polar Vantage V3", "base_price": 599.95, "tags": ["Heart Rate", "Recovery"]},
        {"title": "Coros Pace 3", "base_price": 229.00, "tags": ["Lightweight", "Running"]},
        {"title": "Mobvoi TicWatch Pro 5", "base_price": 349.99, "tags": ["WearOS", "Dual Display"]},
        {"title": "Huawei Watch GT 4", "base_price": 249.00, "tags": ["Design", "Ecosystem"]},
        {"title": "OnePlus Watch 2", "base_price": 299.99, "tags": ["WearOS", "Battery"]},
        {"title": "CMF Watch Pro", "base_price": 69.00, "tags": ["Budget", "Design"]},
        {"title": "Casio G-Shock Move", "base_price": 299.00, "tags": ["Tough", "Solar"]}
    ],
    "Mechanical Keyboards": [
        {"title": "Keychron Q1 Pro", "base_price": 199.00, "tags": ["Custom", "Aluminum"]},
        {"title": "Keychron V1", "base_price": 84.00, "tags": ["Value", "Knob"]},
        {"title": "NuPhy Air75 V2", "base_price": 119.95, "tags": ["Low Profile", "Mac"]},
        {"title": "Logitech G915 Lightspeed", "base_price": 249.99, "tags": ["Gaming", "Wireless"]},
        {"title": "Razer BlackWidow V4", "base_price": 169.99, "tags": ["Gaming", "RGB"]},
        {"title": "Corsair K70 MAX", "base_price": 229.99, "tags": ["Magnetic", "Rapid Trigger"]},
        {"title": "SteelSeries Apex Pro", "base_price": 199.99, "tags": ["OLED", "Adjustable"]},
        {"title": "Wooting 60HE", "base_price": 174.99, "tags": ["Esports", "Analog"]},
        {"title": "Glorious GMMK Pro", "base_price": 169.99, "tags": ["Barebones", "75%"]},
        {"title": "HHKB Professional Hybrid", "base_price": 307.00, "tags": ["Topre", "Compact"]},
        {"title": "Drop CTRL", "base_price": 200.00, "tags": ["Hot-Swap", "RGB"]},
        {"title": "Akko 3068B Plus", "base_price": 89.99, "tags": ["Budget", "Wireless"]},
        {"title": "Epomaker TH80 Pro", "base_price": 99.00, "tags": ["75%", "Volume Knob"]},
        {"title": "Royal Kludge RK61", "base_price": 49.99, "tags": ["Entry Level", "60%"]},
        {"title": "Ducky One 3 Mini", "base_price": 119.00, "tags": ["Typing", "Acoustics"]},
        {"title": "Das Keyboard 4 Pro", "base_price": 169.00, "tags": ["Work", "Volume Knob"]},
        {"title": "IQUNIX F97", "base_price": 229.00, "tags": ["96%", "Design"]},
        {"title": "MelGeek Mojo84", "base_price": 199.00, "tags": ["See-through", "Gasket"]},
        {"title": "Lofree Flow", "base_price": 159.00, "tags": ["Low Profile", "Smooth"]},
        {"title": "Anne Pro 2", "base_price": 89.00, "tags": ["60%", "Software"]}
    ],
    "Gaming Mice": [
        {"title": "Logitech G Pro X 2", "base_price": 159.00, "tags": ["Esports", "Lightweight"]},
        {"title": "Razer DeathAdder V3 Pro", "base_price": 149.99, "tags": ["Ergo", "Performance"]},
        {"title": "Razer Viper V2 Pro", "base_price": 149.99, "tags": ["Ambidextrous", "Speed"]},
        {"title": "Logitech G502 X Plus", "base_price": 159.99, "tags": ["Features", "Buttons"]},
        {"title": "SteelSeries Aerox 3", "base_price": 99.99, "tags": ["Holes", "RGB"]},
        {"title": "HyperX Pulsefire Haste 2", "base_price": 79.99, "tags": ["Value", "Light"]},
        {"title": "Glorious Model O 2", "base_price": 99.99, "tags": ["RGB", "Shape"]},
        {"title": "Pulsar X2V2", "base_price": 94.95, "tags": ["Claw Grip", "Minimal"]},
        {"title": "Lamzu Atlantis OG V2", "base_price": 89.99, "tags": ["Grippy", "Colors"]},
        {"title": "Ninjutso Sora 4K", "base_price": 119.99, "tags": ["No Holes", "4K Hz"]},
        {"title": "Finalmouse UltralightX", "base_price": 189.00, "tags": ["Carbon Fiber", "Hype"]},
        {"title": "Zowie EC2-CW", "base_price": 149.99, "tags": ["Shape King", "Stable"]},
        {"title": "Corsair Dark Core RGB", "base_price": 99.99, "tags": ["Qi Charging", "Grip"]},
        {"title": "ASUS ROG Harpe Ace", "base_price": 139.99, "tags": ["Aim Lab", "Light"]},
        {"title": "Roccat Kone Pro Air", "base_price": 119.99, "tags": ["Optical Switch", "German"]},
        {"title": "Vaxee XE Wireless", "base_price": 119.99, "tags": ["Competitive", "Firmware"]},
        {"title": "Endgame Gear XM2we", "base_price": 79.99, "tags": ["Coatings", "Claw"]},
        {"title": "Keychron M3", "base_price": 49.00, "tags": ["Budget", "Work/Play"]},
        {"title": "Alienware 720M", "base_price": 129.99, "tags": ["Design", "Magnetic"]},
        {"title": "MSI Clutch GM51", "base_price": 99.99, "tags": ["Ergo", "Charging Dock"]}
    ]
}

def generate_catalog():
    catalog = []
    
    for category, products in categories.items():
        for i, prod in enumerate(products):
            # Generate random variations
            rating = round(random.uniform(3.5, 5.0), 1)
            review_count = random.randint(50, 5000)
            
            # Attributes per category
            attrs = {}
            if category == "Wireless Headphones":
                attrs["battery_life"] = f"{random.randint(20, 60)} hours"
                attrs["weight"] = f"{random.randint(200, 350)}g"
                attrs["connection"] = random.choice(["Bluetooth 5.2", "Bluetooth 5.3", "Bluetooth 5.4"])
            elif category == "Smartwatches":
                attrs["battery_life"] = f"{random.randint(18, 72)} hours" if random.random() > 0.3 else f"{random.randint(7, 30)} days"
                attrs["water_resistance"] = random.choice(["5ATM", "10ATM", "IP68"])
                attrs["sensors"] = random.choice(["HR, SpO2", "HR, SpO2, ECG", "HR, SpO2, ECG, Temp"])
            elif category == "Mechanical Keyboards":
                attrs["switch_type"] = random.choice(["Red (Linear)", "Blue (Clicky)", "Brown (Tactile)", "Magnetic"])
                attrs["layout"] = random.choice(["60%", "65%", "75%", "TKL", "Full-size"])
                attrs["connectivity"] = "Wired" if "Wired" in prod["tags"] else "Wireless/Bluetooth/2.4G"
            elif category == "Gaming Mice":
                attrs["dpi"] = f"{random.randint(16000, 32000)}"
                attrs["weight"] = f"{random.randint(40, 100)}g"
                attrs["polling_rate"] = random.choice(["1000Hz", "4000Hz", "8000Hz"])

            item = {
                "id": f"prod_{category.split()[-1].lower()}_{i+1:03d}",
                "category": category,
                "title": prod["title"],
                "base_price": prod["base_price"],
                "rating": rating,
                "review_count": review_count,
                "attributes": attrs,
                "tags": prod["tags"]
            }
            catalog.append(item)
            
    return catalog

if __name__ == "__main__":
    catalog = generate_catalog()
    with open("data/seed_catalog.json", "w") as f:
        json.dump(catalog, f, indent=2)
    print(f"Generated {len(catalog)} items in data/seed_catalog.json")
