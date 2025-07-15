import joblib
import numpy as np
import gradio as gr

# Load the model
model = joblib.load("model/decision_tree_model.pkl")

# Load feature names
with open("model/feature_names.txt", "r") as f:
    feature_names = [line.strip() for line in f]

# City features
cities = ['City_Alameda', 'City_Alamo', 'City_Albany', 'City_Alhambra', 'City_Anaheim', 'City_Antioch', 'City_Aptos', 'City_Arcadia', 'City_Arcata', 'City_Bakersfield', 'City_Baldwin Park', 'City_Banning', 'City_Bella Vista', 'City_Belmont', 'City_Belvedere Tiburon', 'City_Ben Lomond', 'City_Berkeley', 'City_Beverly Hills', 'City_Bodega Bay', 'City_Bonita', 'City_Boulder Creek', 'City_Brea', 'City_Brisbane', 'City_Burlingame', 'City_Calabasas', 'City_Camarillo', 'City_Campbell', 'City_Canoga Park', 'City_Capistrano Beach', 'City_Capitola', 'City_Cardiff By The Sea', 'City_Carlsbad', 'City_Carpinteria', 'City_Carson', 'City_Castro Valley', 'City_Ceres', 'City_Chatsworth', 'City_Chico', 'City_Chino', 'City_Chino Hills', 'City_Chula Vista', 'City_Citrus Heights', 'City_Claremont', 'City_Clearlake', 'City_Clovis', 'City_Concord', 'City_Costa Mesa', 'City_Crestline', 'City_Culver City', 'City_Cupertino', 'City_Cypress', 'City_Daly City', 'City_Danville', 'City_Davis', 'City_Diamond Bar', 'City_Edwards', 'City_El Dorado Hills', 'City_El Segundo', 'City_El Sobrante', 'City_Elk Grove', 'City_Emeryville', 'City_Encinitas', 'City_Escondido', 'City_Eureka', 'City_Fairfield', 'City_Fallbrook', 'City_Fawnskin', 'City_Folsom', 'City_Fremont', 'City_Fresno', 'City_Fullerton', 'City_Garden Grove', 'City_Gilroy', 'City_Glendale', 'City_Glendora', 'City_Goleta', 'City_Greenbrae', 'City_Hacienda Heights', 'City_Half Moon Bay', 'City_Hawthorne', 'City_Hayward', 'City_Hermosa Beach', 'City_Highland', 'City_Hollister', 'City_Hopland', 'City_Huntington Beach', 'City_Imperial', 'City_Inglewood', 'City_Irvine', 'City_La Jolla', 'City_La Mesa', 'City_La Mirada', 'City_La Palma', 'City_Ladera Ranch', 'City_Laguna Hills', 'City_Laguna Niguel', 'City_Lake Forest', 'City_Larkspur', 'City_Livermore', 'City_Loma Linda', 'City_Lomita', 'City_Lompoc', 'City_Long Beach', 'City_Los Alamitos', 'City_Los Altos', 'City_Los Angeles', 'City_Los Gatos', 'City_Manhattan Beach', 'City_March Air Force Base', 'City_Marina', 'City_Martinez', 'City_Menlo Park', 'City_Merced', 'City_Milpitas', 'City_Mission Hills', 'City_Mission Viejo', 'City_Modesto', 'City_Monrovia', 'City_Montague', 'City_Montclair', 'City_Montebello', 'City_Monterey', 'City_Monterey Park', 'City_Moraga', 'City_Morgan Hill', 'City_Moss Landing', 'City_Mountain View', 'City_Napa', 'City_National City', 'City_Newbury Park', 'City_Newport Beach', 'City_North Hills', 'City_North Hollywood', 'City_Northridge', 'City_Norwalk', 'City_Novato', 'City_Oak View', 'City_Oakland', 'City_Oceanside', 'City_Ojai', 'City_Orange', 'City_Oxnard', 'City_Pacific Grove', 'City_Pacific Palisades', 'City_Palo Alto', 'City_Palos Verdes Peninsula', 'City_Pasadena', 'City_Placentia', 'City_Pleasant Hill', 'City_Pleasanton', 'City_Pomona', 'City_Portola Valley', 'City_Poway', 'City_Rancho Cordova', 'City_Rancho Cucamonga', 'City_Rancho Palos Verdes', 'City_Redding', 'City_Redlands', 'City_Redondo Beach', 'City_Redwood City', 'City_Reseda', 'City_Richmond', 'City_Ridgecrest', 'City_Rio Vista', 'City_Riverside', 'City_Rohnert Park', 'City_Rosemead', 'City_Roseville', 'City_Sacramento', 'City_Salinas', 'City_San Anselmo', 'City_San Bernardino', 'City_San Bruno', 'City_San Clemente', 'City_San Diego', 'City_San Dimas', 'City_San Francisco', 'City_San Gabriel', 'City_San Jose', 'City_San Juan Bautista', 'City_San Juan Capistrano', 'City_San Leandro', 'City_San Luis Obispo', 'City_San Luis Rey', 'City_San Marcos', 'City_San Mateo', 'City_San Pablo', 'City_San Rafael', 'City_San Ramon', 'City_San Ysidro', 'City_Sanger', 'City_Santa Ana', 'City_Santa Barbara', 'City_Santa Clara', 'City_Santa Clarita', 'City_Santa Cruz', 'City_Santa Monica', 'City_Santa Rosa', 'City_Santa Ynez', 'City_Saratoga', 'City_Sausalito', 'City_Seal Beach', 'City_Seaside', 'City_Sherman Oaks', 'City_Sierra Madre', 'City_Simi Valley', 'City_Sonora', 'City_South Gate', 'City_South Lake Tahoe', 'City_South Pasadena', 'City_South San Francisco', 'City_Stanford', 'City_Stinson Beach', 'City_Stockton', 'City_Studio City', 'City_Sunland', 'City_Sunnyvale', 'City_Sylmar', 'City_Tahoe City', 'City_Tehachapi', 'City_Thousand Oaks', 'City_Torrance', 'City_Trinity Center', 'City_Tustin', 'City_Ukiah', 'City_Upland', 'City_Valencia', 'City_Vallejo', 'City_Van Nuys', 'City_Venice', 'City_Ventura', 'City_Vista', 'City_Walnut Creek', 'City_Weed', 'City_West Covina', 'City_West Sacramento', 'City_Westlake Village', 'City_Whittier', 'City_Woodland Hills', 'City_Yorba Linda', 'City_Yucaipa']

def predict(age, exp, income, family, ccavg, edu, mort, sa, cd, online, credit, city):
    city_encoding = [1 if city == c else 0 for c in cities]
    base_features = [age, exp, income, family, ccavg, edu, mort, int(sa), int(cd), int(online), int(credit)]
    features = np.array([base_features + city_encoding])

    if features.shape[1] != len(feature_names):
        return f"❌ Feature mismatch: expected {len(feature_names)}, got {features.shape[1]}"

    pred = model.predict(features)[0]
    return "✅ Will Buy Loan" if pred == 1 else "❌ Will Not Buy Loan"

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Age"),
        gr.Number(label="Experience"),
        gr.Number(label="Income"),
        gr.Number(label="Family Members"),
        gr.Number(label="CCAvg"),
        gr.Radio([1, 2, 3], label="Education Level"),
        gr.Number(label="Mortgage"),
        gr.Checkbox(label="Securities Account"),
        gr.Checkbox(label="CD Account"),
        gr.Checkbox(label="Online"),
        gr.Checkbox(label="Credit Card"),
        gr.Dropdown(choices=cities, label="City")
    ],
    outputs="text",
    title="🏦 Loan Purchase Predictor"
)

if __name__ == "__main__":
    demo.launch()
