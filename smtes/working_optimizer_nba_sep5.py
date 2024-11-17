import sys
import logging
import traceback
import psutil
import pulp
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import concurrent.futures
from itertools import combinations
import csv
from collections import defaultdict

# ... existing imports ...

#  ... existing imports ...

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
SALARY_CAP = 50000
POSITION_LIMITS = {
    'PG': 1,
    'SG': 1,
    'SF': 1,
    'PF': 1,
    'C': 1,
    'G': 1,
    'F': 1,
    'UTIL': 1
}
REQUIRED_TEAM_SIZE = 8


def optimize_single_lineup(args):
    simulated_df, stack_type, team_projected_runs, team_selections, min_unique, existing_results, min_points = args
    logging.debug(f"optimize_single_lineup: Starting with stack type {stack_type} and min_points {min_points}")
    logging.debug(f"Team selections: {team_selections}")
    
    if stack_type == "No Stacks":
        stack_sizes = []
    elif stack_type == "4|Any":
        stack_sizes = [4]
    else:
        stack_sizes = [int(size) for size in stack_type.split('|')]
    
    # Check if all required stack sizes have selected teams
    if stack_sizes and not all(size in team_selections for size in stack_sizes):
        logging.warning(f"Not all required stack sizes have selected teams for stack type {stack_type}")
        return pd.DataFrame(), stack_type
    
    # Get all selected teams for this stack type
    selected_teams = set()
    for size in stack_sizes:
        if size in team_selections:
            selected_teams.update(team_selections[size])
    
    logging.debug(f"Selected teams for stack type {stack_type}: {selected_teams}")
    
    problem = pulp.LpProblem("Stack_Optimization", pulp.LpMaximize)
    
    # Filter players to only include those from selected teams
    if selected_teams:
        filtered_players = simulated_df[simulated_df['Team'].isin(selected_teams)]
    else:
        filtered_players = simulated_df
        
    if filtered_players.empty:
        logging.warning(f"No players available after filtering for selected teams: {selected_teams}")
        return pd.DataFrame(), stack_type

    # Log available players per team
    for team in selected_teams:
        team_players = filtered_players[filtered_players['Team'] == team]
        logging.debug(f"Team {team} has {len(team_players)} players available")
        for pos in POSITION_LIMITS.keys():
            pos_players = team_players[team_players['Pos'].str.contains(pos)]
            logging.debug(f"  {pos}: {len(pos_players)} players")

    # Create variables and constraints
    player_vars = {idx: pulp.LpVariable(f"player_{idx}", cat='Binary') for idx in filtered_players.index}

    # Add stack constraints
    if stack_type == "4|Any":
        # Ensure exactly 4 players are selected from one team
        for team in selected_teams:
            team_players = filtered_players[filtered_players['Team'] == team].index
            problem += pulp.lpSum([player_vars[idx] for idx in team_players]) == 4
    else:
        for i, size in enumerate(stack_sizes):
            available_teams = team_selections.get(size, [])
            if not available_teams:
                continue

            # Create binary variables for team selection
            team_vars = {team: pulp.LpVariable(f"team_{team}_{i}", cat='Binary') for team in available_teams}
            
            # Exactly one team must be selected for each stack size
            problem += pulp.lpSum(team_vars.values()) == 1
            
            # For the selected team, exactly 'size' number of players must be selected
            for team in available_teams:
                team_players = filtered_players[filtered_players['Team'] == team].index
                problem += pulp.lpSum([player_vars[idx] for idx in team_players]) >= size * team_vars[team]
                problem += pulp.lpSum([player_vars[idx] for idx in team_players]) <= size * team_vars[team] + (REQUIRED_TEAM_SIZE - size)

    # Multi-Objective Components
    
    # Objective 1: Maximize projected points
    points_expr = pulp.lpSum([filtered_players.at[idx, 'My Proj'] * player_vars[idx] 
                            for idx in filtered_players.index])
    
    # Objective 2: Minimize salary variance (normalized)
    avg_salary = filtered_players['Salary'].mean()
    salary_variance_expr = pulp.lpSum([
        ((filtered_players.at[idx, 'Salary'] - avg_salary) ** 2) * player_vars[idx] 
        for idx in filtered_players.index
    ]) / (SALARY_CAP ** 2)  # Normalize by max possible variance
    
    # Objective 3: Maximize lineup diversity (if there are existing lineups)
    if existing_results:
        diversity_expr = pulp.lpSum([
            player_vars[idx] 
            for idx in filtered_players.index 
            if filtered_players.loc[idx, 'Name'] not in [
                name 
                for lineup in existing_results.values() 
                for name in lineup['lineup']['Name']
            ]
        ]) / REQUIRED_TEAM_SIZE
    else:
        diversity_expr = 0

    # Combine objectives with weights
    # You can adjust these weights based on importance
    w1, w2, w3 = 0.6, 0.2, 0.2  # Weights should sum to 1
    problem += w1 * points_expr - w2 * salary_variance_expr + w3 * diversity_expr

    # Add constraints
    # ... existing constraints ...

    # Add the salary cap constraint
    problem += pulp.lpSum([filtered_players.at[idx, 'Salary'] * player_vars[idx] 
                          for idx in filtered_players.index]) <= SALARY_CAP
    
    # Add the team size constraint
    problem += pulp.lpSum(player_vars.values()) == REQUIRED_TEAM_SIZE

    # Position constraints
    pos_vars = {
        'PG': {},
        'SG': {},
        'SF': {},
        'PF': {},
        'C': {},
        'G': {},
        'F': {},
        'UTIL': {}
    }

    # Create binary variables for each player in each position slot
    for idx in filtered_players.index:
        player_positions = filtered_players.at[idx, 'Pos'].split('/')
        
        # Primary position slots
        if 'PG' in player_positions:
            pos_vars['PG'][idx] = pulp.LpVariable(f"PG_slot_{idx}", cat='Binary')
        if 'SG' in player_positions:
            pos_vars['SG'][idx] = pulp.LpVariable(f"SG_slot_{idx}", cat='Binary')
        if 'SF' in player_positions:
            pos_vars['SF'][idx] = pulp.LpVariable(f"SF_slot_{idx}", cat='Binary')
        if 'PF' in player_positions:
            pos_vars['PF'][idx] = pulp.LpVariable(f"PF_slot_{idx}", cat='Binary')
        if 'C' in player_positions:
            pos_vars['C'][idx] = pulp.LpVariable(f"C_slot_{idx}", cat='Binary')
        
        # G slot (PG/SG)
        if 'PG' in player_positions or 'SG' in player_positions:
            pos_vars['G'][idx] = pulp.LpVariable(f"G_slot_{idx}", cat='Binary')
        
        # F slot (SF/PF)
        if 'SF' in player_positions or 'PF' in player_positions:
            pos_vars['F'][idx] = pulp.LpVariable(f"F_slot_{idx}", cat='Binary')
        
        # UTIL slot (any position)
        pos_vars['UTIL'][idx] = pulp.LpVariable(f"UTIL_slot_{idx}", cat='Binary')

    # Ensure each position slot is filled exactly once
    for pos in POSITION_LIMITS:
        problem += pulp.lpSum(pos_vars[pos].values()) == POSITION_LIMITS[pos]

    # Link position variables to player selection
    for idx in filtered_players.index:
        # A player can only be selected once across all positions
        problem += pulp.lpSum([pos_vars[pos][idx] for pos in pos_vars if idx in pos_vars[pos]]) <= 1
        
        # Link player selection to position selection
        problem += player_vars[idx] == pulp.lpSum([pos_vars[pos][idx] for pos in pos_vars if idx in pos_vars[pos]])

    # Modified minimum points constraint for individual players
    for idx in filtered_players.index:
        proj_points = filtered_players.at[idx, 'My Proj']
        if proj_points < min_points:
            problem += player_vars[idx] == 0
            logging.debug(f"Player {filtered_players.at[idx, 'Name']} excluded (projected points: {proj_points} < {min_points})")
        else:
            logging.debug(f"Player {filtered_players.at[idx, 'Name']} eligible (projected points: {proj_points} >= {min_points})")

    # Add logging for salary constraint
    total_salary = pulp.lpSum([filtered_players.at[idx, 'Salary'] * player_vars[idx] 
                              for idx in filtered_players.index])
    logging.debug(f"Adding salary cap constraint: total_salary <= {SALARY_CAP}")

    # Solve the problem
    solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=60)
    status = problem.solve(solver)

    if pulp.LpStatus[status] == 'Optimal':
        lineup = filtered_players.loc[[idx for idx in filtered_players.index if player_vars[idx].varValue > 0.5]]
        total_points = lineup['My Proj'].sum()
        total_salary = lineup['Salary'].sum()
        
        logging.debug(f"Lineup Metrics:")
        logging.debug(f"Total Points: {total_points}")
        logging.debug(f"Total Salary: {total_salary}")
        
        # Verify salary cap constraint
        if total_salary > SALARY_CAP:
            logging.error(f"Salary cap violated: {total_salary} > {SALARY_CAP}")
            return pd.DataFrame(), stack_type
            
        return lineup, stack_type
    else:
        logging.debug(f"optimize_single_lineup: No optimal solution found. Status: {pulp.LpStatus[status]}")
        return pd.DataFrame(), stack_type

def simulate_iteration(df):
    """Enhanced Monte Carlo simulation for player projections"""
    df = df.copy()
    
    # Player-specific volatility based on historical performance
    # Higher projected players tend to be more consistent
    volatility = 0.1 + (50 - df['My Proj']) * 0.002  # Scale volatility inversely with projection
    volatility = volatility.clip(0.05, 0.25)  # Keep volatility between 5% and 25%
    
    # Generate correlated random factors for players on the same team
    teams = df['Team'].unique()
    team_factors = {team: np.random.normal(1, 0.05) for team in teams}  # Team-level adjustment
    
    # Apply both team and player-level randomness
    for team in teams:
        team_mask = df['Team'] == team
        team_factor = team_factors[team]
        player_factors = np.random.normal(1, volatility[team_mask])
        
        # Combine team and player factors
        combined_factors = team_factor * player_factors
        df.loc[team_mask, 'My Proj'] = df.loc[team_mask, 'My Proj'] * combined_factors
    
    # Add game environment factors
    pace_factor = np.random.normal(1, 0.03)  # Game pace variation
    df['My Proj'] = df['My Proj'] * pace_factor
    
    # Ensure projections stay positive and reasonable
    df['My Proj'] = df['My Proj'].clip(lower=0)
    
    # Add ceiling/floor calculations
    df['Ceiling'] = df['My Proj'] * 1.5
    df['Floor'] = df['My Proj'] * 0.5
    
    return df

class OptimizationWorker(QThread):
    optimization_done = pyqtSignal(dict, dict, dict)

    def __init__(self, df_players, salary_cap, position_limits, included_players, 
                 stack_settings, min_exposure, max_exposure, min_points, 
                 monte_carlo_iterations, num_lineups, min_unique, team_selections):
        super().__init__()
        self.df_players = df_players
        self.num_lineups = num_lineups
        self.salary_cap = salary_cap
        self.position_limits = position_limits
        self.included_players = included_players
        self.stack_settings = stack_settings
        self.min_exposure = min_exposure
        self.max_exposure = max_exposure
        self.team_projected_runs = self.calculate_team_projected_runs(df_players)
        
        self.max_workers = multiprocessing.cpu_count()  # Or set a specific number
        self.min_points = min_points
        self.monte_carlo_iterations = monte_carlo_iterations
        self.team_selections = team_selections
        self.min_unique = min_unique
        logging.debug(f"OptimizationWorker initialized with min_points: {min_points}")

        # New attributes for Monte Carlo analysis
        self.simulation_results = []
        self.player_stats = defaultdict(list)
        
    def run(self):
        logging.debug("OptimizationWorker: Starting optimization")
        results, team_exposure, stack_exposure = self.optimize_lineups()
        logging.debug(f"OptimizationWorker: Optimization complete. Results: {len(results)}")
        self.optimization_done.emit(results, team_exposure, stack_exposure)

    def optimize_lineups(self):
        df_filtered = self.preprocess_data()
        logging.debug(f"optimize_lineups: Starting with {len(df_filtered)} players")
        logging.debug(f"Team selections: {self.team_selections}")
        logging.debug(f"Stack settings: {self.stack_settings}")

        results = {}
        team_exposure = defaultdict(int)
        stack_exposure = defaultdict(int)
        
        # Run Monte Carlo simulations first
        logging.debug(f"Starting Monte Carlo simulations with {self.monte_carlo_iterations} iterations")
        all_simulated_lineups = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Store futures in a dictionary with their metadata
            futures_dict = {}
            
            for i in range(self.monte_carlo_iterations):
                # Generate simulated data for this iteration
                simulated_df = simulate_iteration(df_filtered)
                
                # Run optimization for each stack type with simulated data
                for stack_type in self.stack_settings:
                    future = executor.submit(
                        optimize_single_lineup, 
                        (simulated_df, stack_type, self.team_projected_runs, 
                         self.team_selections, self.min_unique, results, self.min_points)
                    )
                    # Store the future with its metadata
                    futures_dict[future] = (i, stack_type)

            # Process completed futures
            for future in concurrent.futures.as_completed(futures_dict.keys()):
                try:
                    lineup, stack_type = future.result()
                    sim_num = futures_dict[future][0]
                    if not lineup.empty:
                        all_simulated_lineups.append({
                            'lineup': lineup,
                            'stack_type': stack_type,
                            'sim_num': sim_num,
                            'total_points': lineup['My Proj'].sum()
                        })
                        logging.debug(f"Simulation {sim_num} produced lineup with {lineup['My Proj'].sum()} points")
                except Exception as e:
                    logging.error(f"Error in simulation {futures_dict[future][0]}: {str(e)}")

        # Analyze simulation results
        if all_simulated_lineups:
            # Sort lineups by total points
            all_simulated_lineups.sort(key=lambda x: x['total_points'], reverse=True)
            
            # Take the top N unique lineups
            seen_lineups = set()
            for lineup_data in all_simulated_lineups:
                lineup_key = frozenset(lineup_data['lineup']['Name'])
                if lineup_key not in seen_lineups and len(results) < self.num_lineups:
                    results[len(results)] = {
                        'total_points': lineup_data['total_points'],
                        'lineup': lineup_data['lineup']
                    }
                    seen_lineups.add(lineup_key)
                    
                    # Update exposure counts
                    for team in lineup_data['lineup']['Team'].unique():
                        team_exposure[team] += 1
                    stack_exposure[lineup_data['stack_type']] += 1

        logging.debug(f"optimize_lineups: Completed. Found {len(results)} valid lineups")
        logging.debug(f"Team exposure: {dict(team_exposure)}")
        logging.debug(f"Stack exposure: {dict(stack_exposure)}")
        
        return results, team_exposure, stack_exposure
    def preprocess_data(self):
        logging.debug("preprocess_data: Starting")
        df_filtered = self.df_players[self.df_players['My Proj'] > 0]  # Filter out players with 0 or negative projections
        df_filtered = df_filtered[df_filtered['Salary'] > 0]  # Filter out players with 0 or negative salary
        
        if self.included_players:
            df_filtered = df_filtered[df_filtered['Name'].isin(self.included_players)]
        
        # Use the already provided team selections instead of resetting them
        for stack_size in list(self.team_selections.keys()):
            self.team_selections[stack_size] = [
                team for team in self.team_selections[stack_size] 
                if team in df_filtered['Team'].unique()
            ]
            if not self.team_selections[stack_size]:
                del self.team_selections[stack_size]  # Remove stack sizes with no valid teams
        
        logging.debug(f"preprocess_data: Filtered data shape: {df_filtered.shape}")
        logging.debug(f"preprocess_data: Team selections: {self.team_selections}")
        return df_filtered
    def calculate_team_projected_runs(self, df):
        return {team: self.calculate_projected_runs(group) 
                for team, group in df.groupby('Team')}

    def calculate_projected_runs(self, team_players):
        if 'Saber Total' in team_players.columns:
            return team_players['Saber Total'].mean()
        elif 'My Proj' in team_players.columns:
            return team_players['My Proj'].sum() * 0.5
        else:
            logging.warning(f"No projection data available for team {team_players['Team'].iloc[0]}")
            return 0

    def run_monte_carlo_analysis(self):
        """Run multiple simulations and analyze the results"""
        logging.debug(f"Starting Monte Carlo analysis with {self.monte_carlo_iterations} iterations")
        
        for i in range(self.monte_carlo_iterations):
            simulated_df = simulate_iteration(self.df_players)
            results = self.optimize_single_simulation(simulated_df)
            self.simulation_results.append(results)
            
            # Track player performance across simulations
            for lineup in results.values():
                for _, player in lineup['lineup'].iterrows():
                    self.player_stats[player['Name']].append(player['My Proj'])
        
        # Analyze simulation results
        self.analyze_monte_carlo_results()
    
    def analyze_monte_carlo_results(self):
        """Analyze the results of all Monte Carlo simulations"""
        player_analysis = {}
        
        for player, projections in self.player_stats.items():
            stats = {
                'mean': np.mean(projections),
                'std': np.std(projections),
                'median': np.median(projections),
                'ceiling': np.percentile(projections, 90),
                'floor': np.percentile(projections, 10),
                'appearance_rate': len(projections) / self.monte_carlo_iterations
            }
            player_analysis[player] = stats
            
            logging.debug(f"Player {player} analysis:")
            logging.debug(f"  Mean projection: {stats['mean']:.2f}")
            logging.debug(f"  Ceiling (90th): {stats['ceiling']:.2f}")
            logging.debug(f"  Floor (10th): {stats['floor']:.2f}")
            logging.debug(f"  Appearance rate: {stats['appearance_rate']*100:.1f}%")
        
        # Use analysis to adjust player weights in optimization
        self.adjust_player_weights(player_analysis)
    
    def adjust_player_weights(self, player_analysis):
        """Adjust player weights based on Monte Carlo analysis"""
        for idx in self.df_players.index:
            player_name = self.df_players.at[idx, 'Name']
            if player_name in player_analysis:
                stats = player_analysis[player_name]
                
                # Calculate consistency score (lower std dev is better)
                consistency = 1 / (1 + stats['std'])
                
                # Calculate upside potential
                upside = stats['ceiling'] - stats['mean']
                
                # Calculate a composite score
                score = (0.4 * consistency + 0.6 * upside) * stats['appearance_rate']
                
                # Adjust player projection based on analysis
                self.df_players.at[idx, 'Adjusted_Proj'] = (
                    self.df_players.at[idx, 'My Proj'] * (1 + score * 0.1)
                )
    
    def optimize_single_simulation(self, simulated_df):
        """Run optimization for a single simulation"""
        # Your existing optimization logic here, but using simulated_df
        pass

class FantasyBaseballApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced MLB DFS Optimizer")
        self.setGeometry(100, 100, 1600, 1000)
        self.setup_ui()
        self.included_players = []
        self.stack_settings = {}
        self.min_exposure = {}
        self.max_exposure = {}
        self.min_points = 26
        self.monte_carlo_iterations = 500

    def setup_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.splitter)

        self.tabs = QTabWidget()
        self.splitter.addWidget(self.tabs)

        self.df_players = None
        self.df_entries = None
        self.player_exposure = {}
        self.optimized_lineups = []

        self.create_players_tab()
        self.create_team_stack_tab()
        self.create_stack_exposure_tab()
        self.create_positional_constraints_tab()
        self.create_control_panel()
        self.create_1_stack_tab()

    def create_players_tab(self):
        players_tab = QWidget()
        self.tabs.addTab(players_tab, "Players")

        players_layout = QVBoxLayout(players_tab)

        position_tabs = QTabWidget()
        players_layout.addWidget(position_tabs)

        self.player_tables = {}

        positions = ["All Players", "PG", "SG", "SF", "PF", "C"]
        for position in positions:
            sub_tab = QWidget()
            position_tabs.addTab(sub_tab, position)
            layout = QVBoxLayout(sub_tab)

            select_all_button = QPushButton("Select All")
            deselect_all_button = QPushButton("Deselect All")
            select_all_button.clicked.connect(lambda _, p=position: self.select_all(p))
            deselect_all_button.clicked.connect(lambda _, p=position: self.deselect_all(p))
            button_layout = QHBoxLayout()
            button_layout.addWidget(select_all_button)
            button_layout.addWidget(deselect_all_button)
            layout.addLayout(button_layout)

            table = QTableWidget(0, 11)
            table.setHorizontalHeaderLabels(["Select", "Name", "Team", "Pos", "Salary", "My Proj", "Own", "Min Exp", "Max Exp", "Actual Exp (%)", "My Proj"])
            layout.addWidget(table)

            self.player_tables[position] = table

    def create_team_stack_tab(self):
        team_stack_tab = QWidget()
        self.tabs.addTab(team_stack_tab, "Team Stacks")

        layout = QVBoxLayout(team_stack_tab)

        stack_size_tabs = QTabWidget()
        layout.addWidget(stack_size_tabs)

        stack_sizes = ["All Stacks", "1 Stack", "2 Stack", "3 Stack", "4 Stack", "5 Stack"]
        self.team_stack_tables = {}

        for stack_size in stack_sizes:
            sub_tab = QWidget()
            stack_size_tabs.addTab(sub_tab, stack_size)
            sub_layout = QVBoxLayout(sub_tab)

            # Add select/deselect all buttons
            select_all_button = QPushButton("Select All")
            deselect_all_button = QPushButton("Deselect All")
            select_all_button.clicked.connect(lambda _, s=stack_size: self.select_all_teams(s))
            deselect_all_button.clicked.connect(lambda _, s=stack_size: self.deselect_all_teams(s))
            button_layout = QHBoxLayout()
            button_layout.addWidget(select_all_button)
            button_layout.addWidget(deselect_all_button)
            sub_layout.addLayout(button_layout)

            table = QTableWidget(0, 8)
            table.setHorizontalHeaderLabels(["Select", "Teams", "Status", "Time", "Proj Runs", "Min Exp", "Max Exp", "Actual Exp (%)"])
            sub_layout.addWidget(table)

            self.team_stack_tables[stack_size] = table

        self.team_stack_table = self.team_stack_tables["All Stacks"]

        refresh_button = QPushButton("Refresh Team Stacks")
        refresh_button.clicked.connect(self.refresh_team_stacks)
        layout.addWidget(refresh_button)

    def refresh_team_stacks(self):
        self.populate_team_stack_table()

    def create_stack_exposure_tab(self):
        stack_exposure_tab = QWidget()
        self.tabs.addTab(stack_exposure_tab, "Stack Exposure")
    
        layout = QVBoxLayout(stack_exposure_tab)
    
        self.stack_exposure_table = QTableWidget(0, 7)
        self.stack_exposure_table.setHorizontalHeaderLabels(["Select", "Stack Type", "Min Exp", "Max Exp", "Lineup Exp", "Pool Exp", "Entry Exp"])
        layout.addWidget(self.stack_exposure_table)
    
        stack_types = ["4|2|2", "4|2", "3|3|2", "3|2|2", "2|2|2", "5|3", "5|2", "No Stacks", "3|2", "3|2|1|1", "3|2|1|1|1", "3|2|2|1"]
        for stack_type in stack_types:
            row_position = self.stack_exposure_table.rowCount()
            self.stack_exposure_table.insertRow(row_position)
    
            checkbox = QCheckBox()
            checkbox_widget = QWidget()
            layout_checkbox = QHBoxLayout(checkbox_widget)
            layout_checkbox.addWidget(checkbox)
            layout_checkbox.setAlignment(Qt.AlignCenter)
            layout_checkbox.setContentsMargins(0, 0, 0, 0)
            self.stack_exposure_table.setCellWidget(row_position, 0, checkbox_widget)
    
            self.stack_exposure_table.setItem(row_position, 1, QTableWidgetItem(stack_type))
            min_exp_item = QTableWidgetItem("0")
            min_exp_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled)
            self.stack_exposure_table.setItem(row_position, 2, min_exp_item)
    
            max_exp_item = QTableWidgetItem("100")
            max_exp_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled)
            self.stack_exposure_table.setItem(row_position, 3, max_exp_item)
    
            self.stack_exposure_table.setItem(row_position, 4, QTableWidgetItem("0.0%"))
            self.stack_exposure_table.setItem(row_position, 5, QTableWidgetItem("0.0%"))
            self.stack_exposure_table.setItem(row_position, 6, QTableWidgetItem("0.0%"))

    def create_positional_constraints_tab(self):
        positional_constraints_tab = QWidget()
        self.tabs.addTab(positional_constraints_tab, "Positional Constraints")

        layout = QVBoxLayout(positional_constraints_tab)

        self.position_constraints_table = QTableWidget(len(POSITION_LIMITS), 2)
        self.position_constraints_table.setHorizontalHeaderLabels(["Position", "Limit"])
        layout.addWidget(self.position_constraints_table)

        for row, (position, limit) in enumerate(POSITION_LIMITS.items()):
            self.position_constraints_table.setItem(row, 0, QTableWidgetItem(position))
            limit_spinbox = QSpinBox()
            limit_spinbox.setRange(0, 10)  # Adjust range as needed
            limit_spinbox.setValue(limit)
            self.position_constraints_table.setCellWidget(row, 1, limit_spinbox)

    def create_control_panel(self):
        control_panel = QFrame()
        control_panel.setFrameShape(QFrame.StyledPanel)
        control_layout = QVBoxLayout(control_panel)

        self.splitter.addWidget(control_panel)

        load_button = QPushButton('Load CSV')
        load_button.clicked.connect(self.load_file)
        control_layout.addWidget(load_button)

        load_entries_button = QPushButton('Load Entries CSV')
        load_entries_button.clicked.connect(self.load_entries_csv)
        control_layout.addWidget(load_entries_button)

        self.min_unique_label = QLabel('Min Unique:')
        self.min_unique_input = QLineEdit()
        control_layout.addWidget(self.min_unique_label)
        control_layout.addWidget(self.min_unique_input)

        self.min_points_label = QLabel('Min Points:')
        self.min_points_input = QLineEdit()
        control_layout.addWidget(self.min_points_label)
        control_layout.addWidget(self.min_points_input)

        self.sorting_label = QLabel('Sorting Method:')
        self.sorting_combo = QComboBox()
        self.sorting_combo.addItems(["Points", "Value", "Salary"])
        control_layout.addWidget(self.sorting_label)
        control_layout.addWidget(self.sorting_combo)

        run_button = QPushButton('Run Contest Sim')
        run_button.clicked.connect(self.run_optimization)
        control_layout.addWidget(run_button)

        save_button = QPushButton('Save CSV for DraftKings')
        save_button.clicked.connect(self.save_csv)
        control_layout.addWidget(save_button)

        self.results_table = QTableWidget(0, 9)
        self.results_table.setHorizontalHeaderLabels(["Player", "Team", "Pos", "Salary", "My Proj", "Total Salary", "Total Points", "Exposure (%)", "Max Exp (%)"])
        control_layout.addWidget(self.results_table)

        self.status_label = QLabel('')
        control_layout.addWidget(self.status_label)

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open CSV', '', 'CSV Files (*.csv)')
        if file_path:
            self.df_players = self.load_players(file_path)
            if self.df_players is not None:
                logging.debug(f"load_file: Loaded {len(self.df_players)} players")
                self.populate_player_tables()
            else:
                logging.error("load_file: Failed to load players")

    def load_entries_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open Entries CSV', '', 'CSV Files (*.csv)')
        if file_path:
            self.df_entries = self.load_and_standardize_csv(file_path)
            if self.df_entries is not None:
                self.status_label.setText('Entries CSV loaded and standardized successfully.')
            else:
                self.status_label.setText('Failed to standardize Entries CSV.')

    def load_players(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
            required_columns = ['Name', 'Team', 'Opp', 'Pos', 'My Proj', 'Salary']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = np.nan
            df['My Proj'] = pd.to_numeric(df['My Proj'], errors='coerce')
            df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')
            df['Positions'] = df['Pos'].apply(lambda x: x.split('/') if pd.notna(x) else [])
            logging.debug(f"load_players: DataFrame head:\n{df.head()}")
            return df
        except Exception as e:
            logging.error(f"load_players: Error loading file: {e}")
            return None

    def load_and_standardize_csv(self, file_path):
        try:
            df = pd.read_csv(file_path, skiprows=6, on_bad_lines='skip')
            df.columns = ['ID', 'Name', 'Other Columns...'] + df.columns[3:].tolist()
            return df
        except Exception as e:
            logging.error(f"Error loading or processing Can file: {e}")
            return None

    def populate_player_tables(self):
        positions = ["All Players", "PG", "SG", "SF", "PF", "C"]
        
        if self.df_players is None:
            logging.error("populate_player_tables: df_players is None")
            return

        # Debug: Check the contents of df_players
        logging.debug(f"populate_player_tables: df_players head:\n{self.df_players.head()}")

        # Ensure 'Pos' column is treated as string
        self.df_players['Pos'] = self.df_players['Pos'].astype(str)

        for position in positions:
            table = self.player_tables[position]
            table.setRowCount(0)

            if position == "All Players":
                df_filtered = self.df_players
            else:
                df_filtered = self.df_players[self.df_players['Positions'].apply(lambda x: position in x)]

            # Debug: Check the filtered DataFrame
            logging.debug(f"populate_player_tables: {position} filtered count: {len(df_filtered)}")

            for _, row in df_filtered.iterrows():
                row_position = table.rowCount()
                table.insertRow(row_position)

                checkbox = QCheckBox()
                checkbox_widget = QWidget()
                layout_checkbox = QHBoxLayout(checkbox_widget)
                layout_checkbox.addWidget(checkbox)
                layout_checkbox.setAlignment(Qt.AlignCenter)
                layout_checkbox.setContentsMargins(0, 0, 0, 0)
                table.setCellWidget(row_position, 0, checkbox_widget)

                table.setItem(row_position, 1, QTableWidgetItem(str(row['Name'])))
                table.setItem(row_position, 2, QTableWidgetItem(str(row['Team'])))
                table.setItem(row_position, 3, QTableWidgetItem(str(row['Pos'])))
                table.setItem(row_position, 4, QTableWidgetItem(str(row['Salary'])))
                table.setItem(row_position, 5, QTableWidgetItem(str(row['My Proj'])))

                min_exp_spinbox = QSpinBox()
                min_exp_spinbox.setRange(0, 100)
                min_exp_spinbox.setValue(0)
                table.setCellWidget(row_position, 7, min_exp_spinbox)

                max_exp_spinbox = QSpinBox()
                max_exp_spinbox.setRange(0, 100)
                max_exp_spinbox.setValue(100)
                table.setCellWidget(row_position, 8, max_exp_spinbox)

                actual_exp_label = QLabel("")
                table.setCellWidget(row_position, 9, actual_exp_label)

                if row['Name'] not in self.player_exposure:
                    self.player_exposure[row['Name']] = 0

        self.populate_team_stack_table()
        
    def populate_team_stack_table(self):
        team_runs = self.calculate_team_projected_runs()
        selected_teams = self.get_selected_teams()
        logging.debug(f"Populating team stack table with selected teams: {selected_teams}")

        for stack_size, table in self.team_stack_tables.items():
            table.setRowCount(0)
            for team in selected_teams:
                self.add_team_to_stack_table(table, team, team_runs.get(team, 0))
                logging.debug(f"Added team {team} to {stack_size} stack table")

    def get_selected_teams(self):
        selected_teams = set()
        for position, table in self.player_tables.items():
            for row in range(table.rowCount()):
                checkbox_widget = table.cellWidget(row, 0)
                if checkbox_widget is not None:
                    checkbox = checkbox_widget.findChild(QCheckBox)
                    if checkbox is not None and checkbox.isChecked():
                        team = table.item(row, 2).text()
                        selected_teams.add(team)
                        logging.debug(f"Selected team: {team}")
        logging.debug(f"Total selected teams: {len(selected_teams)}")
        return selected_teams
    def calculate_team_projected_runs(self):
        if self.df_players is None:
            return {}
        return {team: self.calculate_projected_runs(group) 
                for team, group in self.df_players.groupby('Team')}

    def calculate_projected_runs(self, team_players):
        if 'Saber Total' in team_players.columns:
            return team_players['Saber Total'].mean()
        elif 'My Proj' in team_players.columns:
            return team_players['My Proj'].sum() * 0.5
        else:
            logging.warning(f"No projection data available for team {team_players['Team'].iloc[0]}")
            return 0


    def add_team_to_stack_table(self, table, team, proj_runs):
        logging.debug(f"Adding team {team} to stack table with projected runs: {proj_runs}")
        row_position = table.rowCount()
        table.insertRow(row_position)

        checkbox = QCheckBox()
        checkbox_widget = QWidget()
        layout_checkbox = QHBoxLayout(checkbox_widget)
        layout_checkbox.addWidget(checkbox)
        layout_checkbox.setAlignment(Qt.AlignCenter)
        layout_checkbox.setContentsMargins(0, 0, 0, 0)
        table.setCellWidget(row_position, 0, checkbox_widget)

        table.setItem(row_position, 1, QTableWidgetItem(team))
        table.setItem(row_position, 2, QTableWidgetItem("Playing"))
        table.setItem(row_position, 3, QTableWidgetItem("7:00 PM"))
        table.setItem(row_position, 4, QTableWidgetItem(f"{proj_runs:.2f}"))

        min_exp_spinbox = QSpinBox()
        min_exp_spinbox.setRange(0, 100)
        min_exp_spinbox.setValue(0)
        table.setCellWidget(row_position, 5, min_exp_spinbox)

        max_exp_spinbox = QSpinBox()
        max_exp_spinbox.setRange(0, 100)
        max_exp_spinbox.setValue(100)
        table.setCellWidget(row_position, 6, max_exp_spinbox)

        actual_exp_label = QLabel("")
        table.setCellWidget(row_position, 7, actual_exp_label)
    def select_all(self, position):
            table = self.player_tables[position]
            for row in range(table.rowCount()):
                checkbox_widget = table.cellWidget(row, 0)
                if checkbox_widget is not None:
                    checkbox = checkbox_widget.findChild(QCheckBox)
                    if checkbox is not None:
                        checkbox.setChecked(True)
            self.refresh_team_stacks()

    def deselect_all(self, position):
            table = self.player_tables[position]
            for row in range(table.rowCount()):
                checkbox_widget = table.cellWidget(row, 0)
                if checkbox_widget is not None:
                    checkbox = checkbox_widget.findChild(QCheckBox)
                    if checkbox is not None:
                        checkbox.setChecked(False)
            self.refresh_team_stacks()

    def run_optimization(self):
        logging.debug("Starting run_optimization method")
        
        # Update position limits from the UI
        self.update_position_limits()
        
        if self.df_players is None or self.df_players.empty:
            self.status_label.setText("No player data loaded. Please load a CSV first.")
            logging.debug("No player data loaded")
            return
        
        logging.debug(f"df_players shape: {self.df_players.shape}")
        logging.debug(f"df_players columns: {self.df_players.columns}")
        logging.debug(f"df_players sample:\n{self.df_players.head()}")
        
        self.included_players = self.get_included_players()
        self.stack_settings = self.collect_stack_settings()
        logging.debug(f"Stack settings: {self.stack_settings}")
        self.min_exposure, self.max_exposure = self.collect_exposure_settings()
        
        logging.debug(f"Included players: {len(self.included_players)}")
        
        try:
            min_points = float(self.min_points_input.text()) if self.min_points_input.text().strip() else 0
            logging.debug(f"Using min_points value: {min_points}")
            if min_points <= 0:
                logging.warning("Min points is zero or negative, constraint may not be effective")
        except ValueError as e:
            logging.error(f"Invalid min_points value: {e}")
            self.status_label.setText("Invalid minimum points value")
            return

        # Retrieve min_unique from the UI input
        min_unique = int(self.min_unique_input.text()) if self.min_unique_input.text() else 0  # Ensure min_unique is defined
        logging.debug(f"Using min_unique value: {min_unique}")

        team_selections = self.collect_team_selections()
        logging.debug(f"Team selections for optimization: {team_selections}")
        
        self.optimization_thread = OptimizationWorker(
            df_players=self.df_players,
            salary_cap=SALARY_CAP,
            position_limits=POSITION_LIMITS,
            included_players=self.included_players,
            stack_settings=self.stack_settings,
            min_exposure=self.min_exposure,
            max_exposure=self.max_exposure,
            min_points=min_points,
            monte_carlo_iterations=self.monte_carlo_iterations,
            num_lineups=100,
            min_unique=min_unique,  # Pass the defined min_unique value
            team_selections=team_selections
        )
        self.optimization_thread.optimization_done.connect(self.display_results)
        logging.debug("Starting optimization thread")
        self.optimization_thread.start()
        
        self.status_label.setText("Running optimization... Please wait.")

    def display_results(self, results, team_exposure, stack_exposure):
        logging.debug(f"display_results: Received {len(results)} results")
        self.results_table.setRowCount(0)
        total_lineups = len(results)

        # Sort results by total points
        sorted_results = sorted(results.items(), key=lambda x: x[1]['total_points'], reverse=True)
        
        # Log all lineup total points for debugging
        for lineup_id, lineup_data in sorted_results:
            total_points = lineup_data['total_points']
            logging.debug(f"Lineup {lineup_id} total points: {total_points}")

        self.optimized_lineups = []
        for _, lineup_data in sorted_results:
            self.add_lineup_to_results(lineup_data, total_lineups)
            self.optimized_lineups.append(lineup_data['lineup'])

        self.update_exposure_in_all_tabs(total_lineups, team_exposure, stack_exposure)
        self.refresh_team_stacks()
        self.status_label.setText(f"Optimization complete. Generated {total_lineups} lineups.")

    def add_lineup_to_results(self, lineup_data, total_lineups):
        total_points = lineup_data['total_points']
        lineup = lineup_data['lineup']
        total_salary = lineup['Salary'].sum()

        for _, player in lineup.iterrows():
            row_position = self.results_table.rowCount()
            self.results_table.insertRow(row_position)
            self.results_table.setItem(row_position, 0, QTableWidgetItem(str(player['Name'])))
            self.results_table.setItem(row_position, 1, QTableWidgetItem(str(player['Team'])))
            self.results_table.setItem(row_position, 2, QTableWidgetItem(str(player['Pos'])))
            self.results_table.setItem(row_position, 3, QTableWidgetItem(str(player['Salary'])))
            self.results_table.setItem(row_position, 4, QTableWidgetItem(f"{player['My Proj']:.2f}"))
            self.results_table.setItem(row_position, 5, QTableWidgetItem(str(total_salary)))
            self.results_table.setItem(row_position, 6, QTableWidgetItem(f"{total_points:.2f}"))

            player_name = player['Name']
            if player_name in self.player_exposure:
                self.player_exposure[player_name] += 1
            else:
                self.player_exposure[player_name] = 1

            exposure = self.player_exposure.get(player_name, 0) / total_lineups * 100
            self.results_table.setItem(row_position, 7, QTableWidgetItem(f"{exposure:.2f}%"))
            self.results_table.setItem(row_position, 8, QTableWidgetItem(f"{self.max_exposure.get(player_name, 100):.2f}%"))

    def update_exposure_in_all_tabs(self, total_lineups, team_exposure, stack_exposure):
        if total_lineups > 0:
            for position in self.player_tables:
                table = self.player_tables[position]
                for row in range(table.rowCount()):
                    player_name = table.item(row, 1).text()
                    actual_exposure = (self.player_exposure.get(player_name, 0) / total_lineups) * 100
                    actual_exposure_label = table.cellWidget(row, 9)
                    if isinstance(actual_exposure_label, QLabel):
                        actual_exposure_label.setText(f"{actual_exposure:.2f}%")

            for stack_size, table in self.team_stack_tables.items():
                for row in range(table.rowCount()):
                    team_name = table.item(row, 1).text()
                    actual_exposure = min(100, (team_exposure.get(team_name, 0) / total_lineups) * 100)
                    table.setItem(row, 7, QTableWidgetItem(f"{actual_exposure:.2f}%"))

            for row in range(self.stack_exposure_table.rowCount()):
                stack_type = self.stack_exposure_table.item(row, 1).text()
                actual_exposure = min(100, (stack_exposure.get(stack_type, 0) / total_lineups) * 100)
                self.stack_exposure_table.setItem(row, 4, QTableWidgetItem(f"{actual_exposure:.2f}%"))

    def save_csv(self):
        if not hasattr(self, 'optimized_lineups') or not self.optimized_lineups:
            self.status_label.setText('No optimized lineups to save. Please run optimization first.')
            return

        output_path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")

        if not output_path:
            self.status_label.setText('Save operation canceled.')
            return

        try:
            with open(output_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['PG', 'SG', 'SF', 'PF', 'C'])
                
                for lineup in self.optimized_lineups:
                    row = []
                    for _, player in lineup.iterrows():
                        row.append(player['Name'])
                    writer.writerow(row)
            
            self.status_label.setText(f'Optimized lineups saved successfully to {output_path}')
        except Exception as e:
            self.status_label.setText(f'Error saving CSV: {str(e)}')

    def generate_output(self, entries_df, players_df, output_path):
        optimized_output = players_df[["Name", "Team", "Pos", "Salary", "My Proj"]]
        optimized_output.to_csv(output_path, index=False)

    def get_included_players(self):
        included_players = []
        for position in self.player_tables:
            table = self.player_tables[position]
            for row in range(table.rowCount()):
                checkbox_widget = table.cellWidget(row, 0)
                if checkbox_widget is not None:
                    checkbox = checkbox_widget.findChild(QCheckBox)
                    if checkbox is not None and checkbox.isChecked():
                        included_players.append(table.item(row, 1).text())
        return included_players

    def collect_stack_settings(self):
        stack_settings = []
        for row in range(self.stack_exposure_table.rowCount()):
            checkbox_widget = self.stack_exposure_table.cellWidget(row, 0)
            if checkbox_widget is not None:
                checkbox = checkbox_widget.findChild(QCheckBox)
                if checkbox is not None and checkbox.isChecked():
                    stack_type = self.stack_exposure_table.item(row, 1).text()
                    stack_settings.append(stack_type)
        if not stack_settings:
            stack_settings.append("No Stacks")  # Default to "No Stacks" if nothing is selected
        return stack_settings

    def collect_exposure_settings(self):
        min_exposure = {}
        self.max_exposure = {}
        for position in self.player_tables:
            table = self.player_tables[position]
            for row in range(table.rowCount()):
                player_name = table.item(row, 1).text()
                min_exp_widget = table.cellWidget(row, 7)
                max_exp_widget = table.cellWidget(row, 8)
                if isinstance(min_exp_widget, QSpinBox) and isinstance(max_exp_widget, QSpinBox):
                    min_exposure[player_name] = min_exp_widget.value() / 100
                    self.max_exposure[player_name] = max_exp_widget.value() / 100
        return min_exposure, self.max_exposure

    def collect_team_selections(self):
        team_selections = {}
        for stack_size, table in self.team_stack_tables.items():
            if stack_size != "All Stacks":
                size = int(stack_size.split()[0])  # Convert "2 Stack" to 2
                selected_teams = []
                for row in range(table.rowCount()):
                    checkbox_widget = table.cellWidget(row, 0)
                    if checkbox_widget is not None:
                        checkbox = checkbox_widget.findChild(QCheckBox)
                        if checkbox is not None and checkbox.isChecked():
                            team = table.item(row, 1).text()
                            selected_teams.append(team)
                            logging.debug(f"Selected team {team} for stack size {size}")
                
                if selected_teams:  # Only add if there are selected teams
                    team_selections[size] = selected_teams
        
        logging.debug(f"Final team selections: {team_selections}")
        return team_selections

    def select_all_teams(self, stack_size):
        table = self.team_stack_tables[stack_size]
        for row in range(table.rowCount()):
            checkbox_widget = table.cellWidget(row, 0)
            if checkbox_widget is not None:
                checkbox = checkbox_widget.findChild(QCheckBox)
                if checkbox is not None:
                    checkbox.setChecked(True)

    def deselect_all_teams(self, stack_size):
        table = self.team_stack_tables[stack_size]
        for row in range(table.rowCount()):
            checkbox_widget = table.cellWidget(row, 0)
            if checkbox_widget is not None:
                checkbox = checkbox_widget.findChild(QCheckBox)
                if checkbox is not None:
                    checkbox.setChecked(False)

    def update_position_limits(self):
        """Update POSITION_LIMITS based on the UI input."""
        for row in range(self.position_constraints_table.rowCount()):
            position = self.position_constraints_table.item(row, 0).text()
            limit_spinbox = self.position_constraints_table.cellWidget(row, 1)
            if isinstance(limit_spinbox, QSpinBox):
                POSITION_LIMITS[position] = limit_spinbox.value()
        logging.debug(f"Updated POSITION_LIMITS: {POSITION_LIMITS}")

    def create_1_stack_tab(self):
        one_stack_tab = QWidget()
        self.tabs.addTab(one_stack_tab, "1 Stack")

        layout = QVBoxLayout(one_stack_tab)

        # Add components specific to the "1 Stack" tab
        label = QLabel("Configure 1 Stack settings here.")
        layout.addWidget(label)

        # Example: Add a table for 1 Stack configurations
        self.one_stack_table = QTableWidget(0, 3)
        self.one_stack_table.setHorizontalHeaderLabels(["Select", "Team", "Exposure"])
        layout.addWidget(self.one_stack_table)

        # Add select/deselect all buttons
        select_all_button = QPushButton("Select All")
        deselect_all_button = QPushButton("Deselect All")
        select_all_button.clicked.connect(self.select_all_1_stack)
        deselect_all_button.clicked.connect(self.deselect_all_1_stack)
        button_layout = QHBoxLayout()
        button_layout.addWidget(select_all_button)
        button_layout.addWidget(deselect_all_button)
        layout.addLayout(button_layout)

    def select_all_1_stack(self):
        for row in range(self.one_stack_table.rowCount()):
            checkbox_widget = self.one_stack_table.cellWidget(row, 0)
            if checkbox_widget is not None:
                checkbox = checkbox_widget.findChild(QCheckBox)
                if checkbox is not None:
                    checkbox.setChecked(True)

    def deselect_all_1_stack(self):
        for row in range(self.one_stack_table.rowCount()):
            checkbox_widget = self.one_stack_table.cellWidget(row, 0)
            if checkbox_widget is not None:
                checkbox = checkbox_widget.findChild(QCheckBox)
                if checkbox is not None:
                    checkbox.setChecked(False)

if __name__ == "__main__":
    logging.debug(f"PuLP version: {pulp.__version__}")
    try:
        pulp.pulpTestAll()
        logging.debug("PuLP test results: All tests passed")
    except pulp.PulpError as e:
        logging.error(f"PuLP test results: Tests failed with error: {e}")
    
    app = QApplication(sys.argv)
    window = FantasyBaseballApp()
    window.show()
    sys.exit(app.exec_())
 
