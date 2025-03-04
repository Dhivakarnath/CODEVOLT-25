from flask import Flask, render_template, request, session, url_for, jsonify
import joblib
import pandas as pd
import numpy as np
from groq import Groq
import traceback

app = Flask(__name__)
app.secret_key = 'simple_poc_key_2023'

try:
    speed_model = joblib.load(r"D:\Codevolt\models\speed_limit.pkl")
    wear_tear_model = joblib.load(r"D:\Codevolt\models\wear_tear.pkl")
    bhi_model = joblib.load(r"D:\Codevolt\models\bhi_model.pkl")
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    raise

groq_client = Groq(api_key="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")  

@app.before_request
def initialize_session():
    if 'history' not in session:
        session['history'] = {
            'speed_limit': [],
            'wear_tear': [],
            'bhi': []
        }
    if 'speed_limit' not in session:
        session['speed_limit'] = 0  

@app.route('/', methods=['GET'])
def speed_limit_estimator():
    speed_limit = session.get('speed_limit', 0) 
    return render_template('speed_limit.html', speed_limit=speed_limit, history=session['history']['speed_limit'])

@app.route('/predict_speed', methods=['POST'])
def predict_speed():
    try:
        data = request.json
        print(f"Received data: {data}")
        acceleration = float(data.get('acceleration', 0))
        distance_km = float(data.get('distance_km', 0))
        battery_level = float(data.get('battery_level', 0))
        terrain_elevation = int(data.get('terrain_elevation', 0))
        weather_condition = int(data.get('weather_condition', 0))

        if acceleration == 0 or distance_km == 0 or battery_level == 0:
            speed_limit = 0  
            print("One or more key inputs (acceleration, distance, battery) are 0, setting speed_limit to 0")
        else:
            speed_inputs = pd.DataFrame(
                [[acceleration, distance_km, battery_level, terrain_elevation, weather_condition]],
                columns=["acceleration", "distance_km", "battery_level", "terrain_elevation", "weather_condition"]
            )
            speed_limit = speed_model.predict(speed_inputs)[0]
            print(f"Model predicted speed_limit: {speed_limit}")

        session['speed_limit'] = float(speed_limit)

        history_entry = {
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'speed_limit': float(speed_limit)
        }
        session['history']['speed_limit'].append(history_entry)
        if len(session['history']['speed_limit']) > 10:
            session['history']['speed_limit'].pop(0)
        session.modified = True

        print(f"Final speed_limit sent: {speed_limit}")
        return jsonify({
            'speed_limit': float(speed_limit),
            'history': session['history']['speed_limit']
        })
    except Exception as e:
        print(f"Error in predict_speed: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/wear_tear_bhi', methods=['GET', 'POST'])
def wear_tear_bhi():
    battery_health = session.get('battery_health')
    brake_health = session.get('brake_health')
    tyre_health = session.get('tyre_health')
    bhi = session.get('bhi')
    
    if request.method == 'POST':
        if 'predict_wear_tear' in request.form:
            wear_tear_pred = wear_tear_model.predict(pd.DataFrame(
                [[np.random.uniform(0, 10), np.random.uniform(0, 10), np.random.uniform(20, 120), 
                  np.random.uniform(5, 150), np.random.choice([0, 1, 2]), np.random.choice([-2, -1, 0, 1, 2])]],
                columns=["braking_intensity", "harsh_acceleration", "average_speed", "distance_driven", 
                         "weather_condition", "terrain_type"]
            ))[0]
            battery_health, brake_health, tyre_health = wear_tear_pred
            session.update({
                'battery_health': float(battery_health), 
                'brake_health': float(brake_health), 
                'tyre_health': float(tyre_health)
            })
            
            history_entry = {
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'battery_health': float(battery_health),
                'brake_health': float(brake_health),
                'tyre_health': float(tyre_health)
            }
            session['history']['wear_tear'].append(history_entry)
            if len(session['history']['wear_tear']) > 10:
                session['history']['wear_tear'].pop(0)
            session.modified = True
            
        elif 'predict_bhi' in request.form:
            bhi = bhi_model.predict(pd.DataFrame(
                [[np.random.uniform(10, 25), 22.5, np.random.uniform(180, 240), 180, 240, 
                  np.random.uniform(100, 500), np.random.uniform(80, 500), np.random.uniform(0.01, 0.2), 
                  np.random.uniform(10, 100), np.random.uniform(230, 240), np.random.uniform(180, 240), 
                  np.random.randint(1, 30), np.random.uniform(50, 100), np.random.uniform(0, 100), 
                  np.random.uniform(85, 100), np.random.uniform(0.01, 0.5), np.random.uniform(0.01, 5)]],
                columns=["Current_Capacity", "Rated_Capacity", "Current_Voltage", "Min_Voltage", "Max_Voltage",
                         "Charge_Energy", "Discharge_Energy", "Voltage_Drop", "Current", "Initial_Voltage",
                         "Final_Voltage", "Time_Days", "SoH", "SoC", "CE", "IR", "SDR"]
            ))[0]
            session['bhi'] = float(bhi)
            
            history_entry = {
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'bhi': float(bhi)
            }
            session['history']['bhi'].append(history_entry)
            if len(session['history']['bhi']) > 10:
                session['history']['bhi'].pop(0)
            session.modified = True

    return render_template('wear_tear_bhi.html', battery_health=battery_health, brake_health=brake_health, 
                          tyre_health=tyre_health, bhi=bhi, wear_tear_history=session['history']['wear_tear'], 
                          bhi_history=session['history']['bhi'])

@app.route('/summary_report')
def summary_report():
    speed_limit = session.get('speed_limit', 'Not available')
    battery_health = session.get('battery_health', 'Not available')
    brake_health = session.get('brake_health', 'Not available')
    tyre_health = session.get('tyre_health', 'Not available')
    bhi = session.get('bhi', 'Not available')

    prompt = f"""
    You are an AI assistant crafting a rider summary for an EV vehicle owner. Based on the following metrics, provide a concise, engaging, personalized, advisable, and logical summary of their vehicle's performance and tailored maintenance/driving tips:

    - Predicted speed limit: {speed_limit if isinstance(speed_limit, float) else 'Not available'} km/h
    - Battery health: {battery_health if isinstance(battery_health, float) else 'Not available'}%
    - Brake health: {brake_health if isinstance(brake_health, float) else 'Not available'}%
    - Tyre health: {tyre_health if isinstance(tyre_health, float) else 'Not available'}%
    - Battery Health Index (BHI): {bhi if isinstance(bhi, float) else 'Not available'}%

    Guidelines for personalization:
    - If battery health < 50%, suggest optimizing charging habits (e.g., avoid full discharges) or scheduling a service.
    - If brake health < 60%, recommend checking brake pads/fluid and adjusting braking habits.
    - If tyre health < 70%, advise tyre rotation, pressure checks, or alignment.
    - If speed limit > 100 km/h, suggest safer speeds for efficiency and safety.
    - For metrics ≥ 80%, offer praise (e.g., "You've got a star performer here!").
    - Weave in the user's metrics naturally for a conversational tone.

    Format the response as a rider summary starting with a greeting, followed by a performance overview, and ending with actionable tips. Keep it under 300 words for brevity and appeal.
    """

    try:
        completion = groq_client.chat.completions.create(
            model="llama3-8b-8192", messages=[{"role": "user", "content": prompt}], 
            temperature=0.7, max_tokens=300, top_p=0.9
        )
        recommendations = completion.choices[0].message.content
    except Exception as e:
        recommendations = f"Error generating rider summary: {str(e)}"

    health_data = {
        'Battery Health': battery_health if isinstance(battery_health, float) else 0,
        'Brake Health': brake_health if isinstance(brake_health, float) else 0,
        'Tyre Health': tyre_health if isinstance(tyre_health, float) else 0,
        'BHI': bhi if isinstance(bhi, float) else 0
    }
    return render_template('summary_report.html', recommendations=recommendations, health_data=health_data, 
                          speed_limit=speed_limit, history=session['history'])

@app.route('/clear_history', methods=['POST'])
def clear_history():
    tab = request.json.get('tab')
    if tab == 'all':
        session['history'] = {'speed_limit': [], 'wear_tear': [], 'bhi': []}
    elif tab in session['history']:
        session['history'][tab] = []
    session.modified = True
    return '', 204

if __name__ == '__main__':
    app.run(debug=True)