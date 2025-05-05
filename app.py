from flask import Flask, render_template, request
from flask import jsonify, render_template_string
from flask import redirect
import pandas as pd
import numpy as np
import folium
from folium import PolyLine
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import time

import os, ast

app = Flask(__name__)

# Load model and scalers
model = load_model("tcn_model.h5", compile=False)
model.compile(optimizer=Adam(learning_rate=0.00030508), loss=MeanSquaredError())
x_scaler = joblib.load("x_scaler.save")
y_scaler = joblib.load("y_scaler.save")

# Load dataset
df = pd.read_csv("Final_Dataset_2013_2022.csv")
df['index'] = df['index'] + 1

# Input features for model
feature_cols = ['AADT_VN', 'BEGIN_POIN', 'COUNTY_COD', 'END_POINT', 'IS_IMPROVED',
                'SPEED_LIMI', 'THROUGH_LA', 'YEAR_RECOR', 'curval', 'tmiles', 'tons', 'value']

# Sequence builder
def create_sequences_from_df(data, input_features, window_size=8):
    sequences = []
    for i in range(len(data) - window_size):
        seq = data[input_features].iloc[i:i+window_size].values
        sequences.append(seq)
    return np.array(sequences)

@app.route('/predict', methods=['GET', 'POST'])
def index():
    route_ids = sorted(df['ROUTE_ID'].unique())
    begins, map_html, table_rows = [], None, []
    graph_path = None
    selected_begins = []  
    selected_route = ''

    if request.method == 'POST':
        selected_route = request.form.get('route_id')
        selected_begins = request.form.getlist('begin_poin')
        filtered = df[df['ROUTE_ID'] == selected_route]
        begins = sorted(filtered['BEGIN_POIN'].unique())

        if selected_begins:
            selected = filtered[filtered['BEGIN_POIN'].isin([float(b) for b in selected_begins])].reset_index(drop=True)
            seq_input = create_sequences_from_df(selected, feature_cols)

            if len(seq_input) == 0:
                return render_template('index.html', route_ids=route_ids, begins=begins, table=[], map_html=None, graph=None)

            # Scale and predict
            seq_scaled = x_scaler.transform(seq_input.reshape(-1, seq_input.shape[-1])).reshape(seq_input.shape)
            preds_scaled = model.predict(seq_scaled).reshape(-1, 1)
            preds = y_scaler.inverse_transform(preds_scaled).flatten()

            # Extract and deduplicate
            target_rows = selected.iloc[8:].copy().reset_index(drop=True)
            target_rows['Predicted_IRI'] = preds
            unique_targets = target_rows.drop_duplicates(subset=['BEGIN_POIN']).reset_index(drop=True)

            # Table
            table_rows = unique_targets[['ROUTE_ID', 'BEGIN_POIN', 'END_POINT', 'Predicted_IRI']].values.tolist()

            # Folium map
            try:
                all_coords = []
                fmap = folium.Map(location=[20, 80], zoom_start=5)  # initial safe default

                for _, row in unique_targets.iterrows():
                    try:
                        paths = ast.literal_eval(row['geometry_paths'])
                        color = 'green' if row['Predicted_IRI'] < 90 else 'yellow' if row['Predicted_IRI'] <= 150 else 'red'
                        for sub_path in paths:
                            coords = [(lat, lon) for lon, lat in sub_path]
                            all_coords.extend(coords)
                            PolyLine(coords, color=color, weight=3).add_to(fmap)
                    except:
                        continue

                if all_coords:
                    fmap.fit_bounds(all_coords)

                os.makedirs("static", exist_ok=True)
                fmap.save("static/map.html")
                map_html = "map.html"
            except:
                map_html = None

            # Graph
            if len(selected_begins) == 1:  # Only generate graph if one section is selected
                try:
                    plt.figure(figsize=(10, 5))
                    palette = sns.color_palette("husl", len(selected_begins))
                    for i, b in enumerate(selected_begins):
                        b = float(b)
                        section_data = df[(df['ROUTE_ID'] == selected_route) & (df['BEGIN_POIN'] == b)].copy()

                        # Replace IRI_VN for 2022 with predicted value
                        if not section_data.empty and 2022 in section_data['YEAR_RECOR'].values:
                            predicted_value = unique_targets.loc[unique_targets['BEGIN_POIN'] == b, 'Predicted_IRI']
                            if not predicted_value.empty:
                                section_data.loc[section_data['YEAR_RECOR'] == 2022, 'IRI_VN'] = predicted_value.values[0]

                        sns.lineplot(data=section_data, x='YEAR_RECOR', y='IRI_VN', label=f"Section {b}", color=palette[i])

                    plt.title(f"IRI Trend - Route {selected_route}")
                    plt.xlabel("Year")
                    plt.ylabel("IRI Value")
                    plt.legend()
                    graph_path = "static/iri_plot.png"
                    plt.tight_layout()
                    plt.savefig(graph_path)
                    plt.close()
                except Exception as e:
                    graph_path = None

    return render_template(
        'index.html',
        route_ids=route_ids,
        begins=begins,
        selected_begins=selected_begins,
        selected_route=selected_route,
        table=table_rows,
        map_html=map_html,
        graph=graph_path
    )

@app.route('/update_graph', methods=['POST'])
def update_graph():
    route_id = request.form.get('route_id')
    begin_poin = request.form.get('begin_poin')

    if not route_id or not begin_poin:
        return "<p>Please select a section.</p>"

    b = float(begin_poin)
    filtered = df[(df['ROUTE_ID'] == route_id) & (df['BEGIN_POIN'] == b)].copy()

    # Predict IRI if needed (simplified version)
    section_data = filtered.copy()
    predicted_value = None
    if not section_data.empty and 2022 in section_data['YEAR_RECOR'].values:
        seq_input = create_sequences_from_df(section_data, feature_cols)
        if len(seq_input) > 0:
            seq_scaled = x_scaler.transform(seq_input.reshape(-1, seq_input.shape[-1])).reshape(seq_input.shape)
            preds_scaled = model.predict(seq_scaled).reshape(-1, 1)
            preds = y_scaler.inverse_transform(preds_scaled).flatten()
            predicted_value = preds[-1]  # last value
            section_data.loc[section_data['YEAR_RECOR'] == 2022, 'IRI_VN'] = predicted_value

    # Create graph
    try:
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=section_data, x='YEAR_RECOR', y='IRI_VN', label=f"Section {b}")
        plt.title(f"IRI Trend - Route {route_id}")
        plt.xlabel("Year")
        plt.ylabel("IRI Value")
        plt.legend()
        graph_path = "static/iri_plot.png"
        plt.tight_layout()
        plt.savefig(graph_path)
        plt.close()

        # Return updated graph HTML
        return render_template_string("""
            <h3>Graph</h3>
            <img src="{{ url_for('static', filename='iri_plot.png') }}?t={{ timestamp }}" style="width: 100%; height: auto;" />
        """, timestamp=int(time.time()))
    except:
        return "<p>Error generating graph.</p>"


@app.route('/update_map', methods=['POST'])
def update_map():
    route_id = request.form.get('route_id')
    begin_poin = request.form.get('begin_poin')

    if not route_id or not begin_poin:
        return jsonify(success=False)

    b = float(begin_poin)
    filtered = df[(df['ROUTE_ID'] == route_id) & (df['BEGIN_POIN'] == b)].copy()

    try:
        all_coords = []
        fmap = folium.Map(location=[20, 80], zoom_start=5)

        for _, row in filtered.iterrows():
            try:
                paths = ast.literal_eval(row['geometry_paths'])
                iri_val = row.get("IRI_VN", 0)
                color = 'green' if iri_val < 94 else 'yellow' if iri_val <= 119 else 'red'
                for sub_path in paths:
                    coords = [(lat, lon) for lon, lat in sub_path]
                    all_coords.extend(coords)
                    PolyLine(coords, color=color, weight=3).add_to(fmap)
            except:
                continue

        if all_coords:
            fmap.fit_bounds(all_coords)

        os.makedirs("static", exist_ok=True)
        fmap_path = "static/map.html"
        fmap.save(fmap_path)
        return jsonify(success=True, map_url=f"/static/map.html?t={int(time.time())}")
    except Exception as e:
        print(f"Map update error: {e}")
        return jsonify(success=False)

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/')
def root():
    return redirect('/home')

if __name__ == '__main__':
    app.run(debug=True)
