import streamlit as st
import logging
import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify, request, send_file
import pulp
from collections import defaultdict
from working_optimizer_nba_sep5 import (
    optimize_single_lineup, 
    SALARY_CAP, 
    POSITION_LIMITS,
    simulate_iteration,
    OptimizationWorker
)
import os

app = Flask(__name__)

# Azure configuration
if 'WEBSITE_HOSTNAME' in os.environ:
    # Running on Azure
    app.config['DEBUG'] = False
    app.config['UPLOAD_FOLDER'] = '/tmp'  # Use temp directory on Azure
else:
    # Local development
    app.config['DEBUG'] = True
    app.config['UPLOAD_FOLDER'] = 'uploads'

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables
df_players = None
optimized_lineups = []  

def serialize_df(df):
    """Safely serialize DataFrame to JSON-compatible format"""
    data = []
    for _, row in df.iterrows():
        row_dict = {}
        for column in df.columns:
            value = row[column]
            if isinstance(value, (np.integer, np.floating)):
                value = float(value)
            elif isinstance(value, (pd.Timestamp, np.datetime64)):
                value = value.isoformat()
            elif isinstance(value, (np.bool_)):
                value = bool(value)
            elif pd.isna(value):
                value = None
            row_dict[column] = value
        data.append(row_dict)
    return data

@app.route('/')
def index():
    return render_template('optimizer.html')

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    global df_players
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file provided'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'})

        if file:
            df_players = pd.read_csv(file)
            df_players = clean_dataframe(df_players)
            
            return jsonify({
                'success': True,
                'message': 'CSV loaded successfully',
                'data': serialize_df(df_players)
            })
            
    except Exception as e:
        logging.error(f"Error uploading CSV: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'message': f'Error loading CSV: {str(e)}'})

@app.route('/optimize', methods=['POST'])
def optimize():
    global df_players, optimized_lineups
    try:
        if df_players is None:
            return jsonify({'success': False, 'message': 'No data loaded'})
        
        params = request.json
        
        # Create optimization worker with exact parameters from working_optimizer
        worker = OptimizationWorker(
            df_players=df_players,
            salary_cap=SALARY_CAP,
            position_limits=POSITION_LIMITS,
            included_players=params.get('selected_players', []),
            stack_settings=[params.get('stack_type', 'No Stacks')],
            min_exposure={},  # Add if needed
            max_exposure={},  # Add if needed
            min_points=float(params.get('min_points', 0)),
            monte_carlo_iterations=500,  # Default from working_optimizer
            num_lineups=int(params.get('num_lineups', 1)),
            min_unique=int(params.get('min_unique', 0)),
            team_selections=params.get('team_selections', {})
        )

        # Run optimization using worker's methods
        results, team_exposure, stack_exposure = worker.optimize_lineups()
        
        if results:
            # Convert first result to proper format
            lineup_data = results[0]
            lineup = lineup_data['lineup']
            
            result = {
                'success': True,
                'lineup': serialize_df(lineup),
                'stack_type': params.get('stack_type', 'No Stacks'),
                'total_salary': float(lineup['Salary'].sum()),
                'total_points': float(lineup['My Proj'].sum()),
                'team_exposure': team_exposure,
                'stack_exposure': stack_exposure
            }
            # /Users/sineshawmesfintesfaye/newenv/app.py
            optimized_lineups.append(lineup)
            logging.debug(f"Optimization successful. Total points: {result['total_points']}")
        else:
            result = {
                'success': False,
                'message': 'No valid lineup found'
            }
        
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Optimization error: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'message': str(e)})

def clean_dataframe(df):
    """Clean the dataframe by handling NaN values and invalid data"""
    logging.debug("Cleaning dataframe")
    
    # Replace NaN/inf values with 0
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    # Ensure numeric columns are properly typed
    if 'My Proj' in df.columns:
        df['My Proj'] = pd.to_numeric(df['My Proj'], errors='coerce').fillna(0)
    if 'Salary' in df.columns:
        df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce').fillna(0)
    
    # Remove rows with invalid data
    df = df[df['My Proj'] >= 0]
    df = df[df['Salary'] > 0]
    
    logging.debug(f"Cleaned dataframe shape: {df.shape}")
    return df

if __name__ == '__main__':
    # Use environment port if available (Azure requirement)
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)
