from flask import Flask, render_template, request, session, redirect, url_for
import joblib
import pandas as pd
import numpy as np
from groq import Groq  # Assuming you're using the Groq Python SDK

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'simple_poc_key_2023'  # POC secret key

# Load pre-trained models (ensure these .pkl files are in the same directory)
speed_model = joblib.load(r"D:\Codevolt\speed_limit.pkl")
wear_tear_model = joblib.load(r"D:\Codevolt\wear_tear.pkl")
bhi_model = joblib.load(r"D:\Codevolt\bhi_model.pkl")

# Initialize Groq client with your API key
groq_client = Groq(api_key="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")

# Route for Speed Limit Estimator page
@app.route('/', methods=['GET', 'POST'])
def speed_limit_estimator():
    speed_limit = None
    if request.method == 'POST':
        # Collect user inputs from the form
        acceleration = float(request.form['acceleration'])
        distance_km = float(request.form['distance_km'])
        battery_level = float(request.form['battery_level'])
        terrain_elevation = int(request.form['terrain_elevation'])
        weather_condition = int(request.form['weather_condition'])

        # Prepare input DataFrame for prediction
        speed_inputs = pd.DataFrame(
            [[acceleration, distance_km, battery_level, terrain_elevation, weather_condition]],
            columns=["acceleration", "distance_km", "battery_level", "terrain_elevation", "weather_condition"]
        )

        # Predict speed limit
        speed_limit = speed_model.predict(speed_inputs)[0]

        # Store prediction in session
        session['speed_limit'] = float(speed_limit)  # Convert to float for serialization

    return render_template('speed_limit.html', speed_limit=speed_limit)

# Route for Wear and Tear and BHI Prediction page
@app.route('/wear_tear_bhi', methods=['GET', 'POST'])
def wear_tear_bhi():
    battery_health = brake_health = tyre_health = bhi = None

    if request.method == 'POST':
        if 'predict_wear_tear' in request.form:
            # Generate random inputs for Wear and Tear
            braking_intensity = np.random.uniform(0, 10)
            harsh_acceleration = np.random.uniform(0, 10)
            average_speed = np.random.uniform(20, 120)
            distance_driven = np.random.uniform(5, 150)
            weather_condition = np.random.choice([0, 1, 2])
            terrain_type = np.random.choice([-2, -1, 0, 1, 2])

            wear_tear_inputs = pd.DataFrame(
                [[braking_intensity, harsh_acceleration, average_speed, distance_driven, weather_condition, terrain_type]],
                columns=["braking_intensity", "harsh_acceleration", "average_speed", "distance_driven", "weather_condition", "terrain_type"]
            )

            # Predict Wear and Tear
            wear_tear_pred = wear_tear_model.predict(wear_tear_inputs)[0]
            battery_health, brake_health, tyre_health = wear_tear_pred

            # Store predictions in session
            session['battery_health'] = float(battery_health)
            session['brake_health'] = float(brake_health)
            session['tyre_health'] = float(tyre_health)

        elif 'predict_bhi' in request.form:
            # Generate random inputs for BHI
            current_capacity = np.random.uniform(10, 25)
            rated_capacity = 22.5  # Fixed value
            current_voltage = np.random.uniform(180, 240)
            min_voltage = 180
            max_voltage = 240
            charge_energy = np.random.uniform(100, 500)
            discharge_energy = np.random.uniform(80, 500)
            voltage_drop = np.random.uniform(0.01, 0.2)
            current = np.random.uniform(10, 100)
            initial_voltage = np.random.uniform(230, 240)
            final_voltage = np.random.uniform(180, 240)
            time_days = np.random.randint(1, 30)
            soh = np.random.uniform(50, 100)
            soc = np.random.uniform(0, 100)
            ce = np.random.uniform(85, 100)
            ir = np.random.uniform(0.01, 0.5)
            sdr = np.random.uniform(0.01, 5)

            bhi_inputs = pd.DataFrame(
                [[current_capacity, rated_capacity, current_voltage, min_voltage, max_voltage,
                  charge_energy, discharge_energy, voltage_drop, current, initial_voltage,
                  final_voltage, time_days, soh, soc, ce, ir, sdr]],
                columns=["Current_Capacity", "Rated_Capacity", "Current_Voltage", "Min_Voltage", "Max_Voltage",
                         "Charge_Energy", "Discharge_Energy", "Voltage_Drop", "Current", "Initial_Voltage",
                         "Final_Voltage", "Time_Days", "SoH", "SoC", "CE", "IR", "SDR"]
            )

            # Predict BHI
            bhi = bhi_model.predict(bhi_inputs)[0]

            # Store prediction in session
            session['bhi'] = float(bhi)

    return render_template(
        'wear_tear_bhi.html',
        battery_health=session.get('battery_health'),  # Retrieve from session
        brake_health=session.get('brake_health'),
        tyre_health=session.get('tyre_health'),
        bhi=session.get('bhi')
    )

# Route for Summary Report page
@app.route('/summary_report')
def summary_report():
    # Retrieve predictions from session (default to 'Not available' if not set)
    speed_limit = session.get('speed_limit', 'Not available')
    battery_health = session.get('battery_health', 'Not available')
    brake_health = session.get('brake_health', 'Not available')
    tyre_health = session.get('tyre_health', 'Not available')
    bhi = session.get('bhi', 'Not available')

    # Create prompt for Llama model via Groq Cloud
    prompt = f"""
    Given the following vehicle health and performance metrics:
    - Predicted speed limit: {speed_limit if isinstance(speed_limit, float) else 'Not available'} km/h
    - Battery health: {battery_health if isinstance(battery_health, float) else 'Not available'}%
    - Brake health: {brake_health if isinstance(brake_health, float) else 'Not available'}%
    - Tyre health: {tyre_health if isinstance(tyre_health, float) else 'Not available'}%
    - Battery Health Index (BHI): {bhi if isinstance(bhi, float) else 'Not available'}%
    Please provide recommendations for maintenance and driving behavior to ensure safety and efficiency.
    """

    # Call Groq API to get Llama model response
    try:
        completion = groq_client.chat.completions.create(
            model="llama3-8b-8192",  # Replace with the exact Llama model name if different
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300,
            top_p=0.9
        )
        recommendations = completion.choices[0].message.content
    except Exception as e:
        recommendations = f"Error generating recommendations: {str(e)}"

    # Prepare data for visualization
    health_data = {
        'Battery Health': battery_health if isinstance(battery_health, float) else 0,
        'Brake Health': brake_health if isinstance(brake_health, float) else 0,
        'Tyre Health': tyre_health if isinstance(tyre_health, float) else 0,
        'BHI': bhi if isinstance(bhi, float) else 0
    }

    return render_template(
        'summary_report.html',
        recommendations=recommendations,
        health_data=health_data,
        speed_limit=speed_limit
    )

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)