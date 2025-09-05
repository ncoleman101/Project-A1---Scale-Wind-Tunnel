#!/usr/bin/env python3
"""
Aerodynamic Pressure Measurement and Analysis System
For drag, lift, and pressure distribution experiments
"""

import serial
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
import argparse
import json
import os
from scipy import signal
import threading
import queue

class PressureDataAcquisition:
    def __init__(self, port='COM3', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_connection = None
        self.is_connected = False
        self.is_recording = False
        self.data_buffer = queue.Queue()
        self.recorded_data = []
        self.live_data = {
            'timestamp': [],
            'static_pressure': [],
            'dynamic_pressure': [],
            'velocity_mph': [],
            'drag_pressures': [[], [], [], []],
            'lift_coefficient': [],
            'drag_coefficient': []
        }
        
        # Configuration
        self.config = {
            'air_density': 1.225,  # kg/m³
            'reference_area': 0.01,  # m²
            'chord_length': 0.1,   # m
            'sampling_rate': 100   # Hz
        }
        
    def connect(self):
        """Connect to Arduino pressure measurement system"""
        try:
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=2
            )
            time.sleep(3)  # Wait for Arduino to initialize
            
            # Test connection
            response = self.send_command("STATUS")
            if response and "Pressure Measurement Status" in response:
                self.is_connected = True
                print(f"✓ Connected to pressure measurement system on {self.port}")
                return True
            else:
                raise Exception("No response from Arduino")
                
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Arduino"""
        if self.is_recording:
            self.stop_recording()
            
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            self.is_connected = False
            print("Disconnected from pressure measurement system")
    
    def send_command(self, command):
        """Send command to Arduino"""
        if not self.is_connected:
            return None
            
        try:
            self.serial_connection.write((command + '\n').encode())
            
            # Read multi-line response
            response_lines = []
            start_time = time.time()
            
            while time.time() - start_time < 2.0:
                if self.serial_connection.in_waiting > 0:
                    line = self.serial_connection.readline().decode().strip()
                    if line:
                        response_lines.append(line)
                        if line.startswith("===") and len(response_lines) > 1:
                            break
                        if line in ["MEASUREMENT_STARTED", "MEASUREMENT_STOPPED", "CALIBRATION_COMPLETE", "SENSORS_ZEROED"]:
                            break
                time.sleep(0.01)
            
            return '\n'.join(response_lines)
            
        except Exception as e:
            print(f"Communication error: {e}")
            return None
    
    def configure_system(self, **kwargs):
        """Configure measurement parameters"""
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
                
                # Send to Arduino
                if key == 'air_density':
                    self.send_command(f"CONFIG:DENSITY:{value}")
                elif key == 'reference_area':
                    self.send_command(f"CONFIG:AREA:{value}")
                elif key == 'chord_length':
                    self.send_command(f"CONFIG:CHORD:{value}")
                elif key == 'sampling_rate':
                    self.send_command(f"CONFIG:RATE:{value}")
                    
                print(f"Set {key} to {value}")
    
    def calibrate_sensors(self):
        """Perform sensor calibration"""
        print("Calibrating sensors (ensure zero pressure conditions)...")
        response = self.send_command("CALIBRATE")
        print(response)
        return "CALIBRATION_COMPLETE" in response if response else False
    
    def zero_sensors(self):
        """Zero all sensor readings"""
        response = self.send_command("ZERO")
        return "SENSORS_ZEROED" in response if response else False
    
    def take_single_measurement(self):
        """Take a single measurement"""
        response = self.send_command("SINGLE")
        
        if response:
            # Parse the data line
            lines = response.split('\n')
            for line in lines:
                if ',' in line and not line.startswith('timestamp'):
                    return self.parse_data_line(line)
        return None
    
    def start_recording(self, duration=None):
        """Start continuous data recording"""
        if self.is_recording:
            print("Already recording")
            return False
            
        response = self.send_command("START")
        if not response or "MEASUREMENT_STARTED" not in response:
            print("Failed to start recording")
            return False
        
        self.is_recording = True
        self.recorded_data = []
        
        # Start data reading thread
        self.read_thread = threading.Thread(target=self._read_data_loop, daemon=True)
        self.read_thread.start()
        
        print("Recording started" + (f" for {duration}s" if duration else ""))
        
        # If duration specified, stop after that time
        if duration:
            def stop_after_duration():
                time.sleep(duration)
                if self.is_recording:
                    self.stop_recording()
            threading.Thread(target=stop_after_duration, daemon=True).start()
        
        return True
    
    def stop_recording(self):
        """Stop data recording"""
        if not self.is_recording:
            return
            
        self.is_recording = False
        response = self.send_command("STOP")
        
        # Wait for thread to finish
        if hasattr(self, 'read_thread'):
            self.read_thread.join(timeout=2)
        
        print(f"Recording stopped. Collected {len(self.recorded_data)} samples")
    
    def _read_data_loop(self):
        """Background thread for reading data"""
        while self.is_recording and self.is_connected:
            try:
                if self.serial_connection.in_waiting > 0:
                    line = self.serial_connection.readline().decode().strip()
                    
                    if line and ',' in line and not line.startswith('timestamp'):
                        data_point = self.parse_data_line(line)
                        if data_point:
                            self.recorded_data.append(data_point)
                            self.data_buffer.put(data_point)
                            
                            # Update live data for plotting
                            self.update_live_data(data_point)
                            
            except Exception as e:
                print(f"Data reading error: {e}")
                break
                
            time.sleep(0.001)  # Small delay
    
    def parse_data_line(self, line):
        """Parse a CSV data line from Arduino"""
        try:
            parts = line.split(',')
            if len(parts) >= 10:
                return {
                    'timestamp': int(parts[0]),
                    'static_pressure': float(parts[1]),
                    'dynamic_pressure': float(parts[2]),
                    'velocity_mph': float(parts[3]),
                    'drag_pressure_1': float(parts[4]),
                    'drag_pressure_2': float(parts[5]),
                    'drag_pressure_3': float(parts[6]),
                    'drag_pressure_4': float(parts[7]),
                    'lift_coefficient': float(parts[8]),
                    'drag_coefficient': float(parts[9])
                }
        except (ValueError, IndexError) as e:
            print(f"Data parsing error: {e}")
            return None
        
        return None
    
    def update_live_data(self, data_point):
        """Update live data buffers for plotting"""
        max_points = 1000  # Keep last 1000 points
        
        self.live_data['timestamp'].append(data_point['timestamp'])
        self.live_data['static_pressure'].append(data_point['static_pressure'])
        self.live_data['dynamic_pressure'].append(data_point['dynamic_pressure'])
        self.live_data['velocity_mph'].append(data_point['velocity_mph'])
        self.live_data['lift_coefficient'].append(data_point['lift_coefficient'])
        self.live_data['drag_coefficient'].append(data_point['drag_coefficient'])
        
        for i in range(4):
            self.live_data['drag_pressures'][i].append(data_point[f'drag_pressure_{i+1}'])
        
        # Trim to max_points
        for key in self.live_data:
            if isinstance(self.live_data[key], list):
                if len(self.live_data[key]) > max_points:
                    self.live_data[key] = self.live_data[key][-max_points:]
            elif isinstance(self.live_data[key][0], list):
                for sublist in self.live_data[key]:
                    if len(sublist) > max_points:
                        sublist[:] = sublist[-max_points:]
    
    def save_data(self, filename=None, format='csv'):
        """Save recorded data to file"""
        if not self.recorded_data:
            print("No data to save")
            return False
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if not filename:
            filename = f"pressure_data_{timestamp}"
        
        try:
            if format.lower() == 'csv':
                df = pd.DataFrame(self.recorded_data)
                df.to_csv(f"{filename}.csv", index=False)
                print(f"Data saved to {filename}.csv ({len(df)} samples)")
                
            elif format.lower() == 'json':
                with open(f"{filename}.json", 'w') as f:
                    json.dump({
                        'config': self.config,
                        'data': self.recorded_data,
                        'timestamp': timestamp
                    }, f, indent=2)
                print(f"Data saved to {filename}.json")
                
            return True
            
        except Exception as e:
            print(f"Failed to save data: {e}")
            return False
    
    def load_data(self, filename):
        """Load data from file for analysis"""
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(filename)
                return df.to_dict('records')
            elif filename.endswith('.json'):
                with open(filename, 'r') as f:
                    data = json.load(f)
                return data['data']
        except Exception as e:
            print(f"Failed to load data: {e}")
            return None
    
    def analyze_data(self, data=None):
        """Perform basic analysis on recorded data"""
        if data is None:
            data = self.recorded_data
            
        if not data:
            print("No data to analyze")
            return
        
        df = pd.DataFrame(data)
        
        print("\n=== Data Analysis Summary ===")
        print(f"Total samples: {len(df)}")
        print(f"Duration: {(df['timestamp'].iloc[-1] - df['timestamp'].iloc[0])/1000:.1f} seconds")
        
        print(f"\nVelocity (mph):")
        print(f"  Mean: {df['velocity_mph'].mean():.2f}")
        print(f"  Std:  {df['velocity_mph'].std():.2f}")
        print(f"  Min:  {df['velocity_mph'].min():.2f}")
        print(f"  Max:  {df['velocity_mph'].max():.2f}")
        
        print(f"\nDynamic Pressure (Pa):")
        print(f"  Mean: {df['dynamic_pressure'].mean():.2f}")
        print(f"  Std:  {df['dynamic_pressure'].std():.2f}")
        
        print(f"\nLift Coefficient:")
        print(f"  Mean: {df['lift_coefficient'].mean():.4f}")
        print(f"  Std:  {df['lift_coefficient'].std():.4f}")
        
        print(f"\nDrag Coefficient:")
        print(f"  Mean: {df['drag_coefficient'].mean():.4f}")
        print(f"  Std:  {df['drag_coefficient'].std():.4f}")
        
        return df
    
    def plot_data(self, data=None, live=False):
        """Create plots of the data"""
        if live:
            self.plot_live_data()
        else:
            if data is None:
                data = self.recorded_data
                
            if not data:
                print("No data to plot")
                return
                
            df = pd.DataFrame(data)
            self.plot_static_data(df)
    
    def plot_static_data(self, df):
        """Create static plots of recorded data"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Aerodynamic Pressure Measurements', fontsize=16)
        
        # Time axis (convert to seconds)
        time_s = (df['timestamp'] - df['timestamp'].iloc[0]) / 1000.0
        
        # Velocity
        axes[0, 0].plot(time_s, df['velocity_mph'])
        axes[0, 0].set_title('Velocity')
        axes[0, 0].set_ylabel('Velocity (mph)')
        axes[0, 0].grid(True)
        
        # Pressure
        axes[0, 1].plot(time_s, df['dynamic_pressure'], label='Dynamic')
        axes[0, 1].plot(time_s, df['static_pressure'], label='Static')
        axes[0, 1].set_title('Pressures')
        axes[0, 1].set_ylabel('Pressure (Pa)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Drag pressures
        for i in range(4):
            axes[0, 2].plot(time_s, df[f'drag_pressure_{i+1}'], label=f'Port {i+1}')
        axes[0, 2].set_title('Drag Measurement Pressures')
        axes[0, 2].set_ylabel('Pressure (Pa)')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Coefficients
        axes[1, 0].plot(time_s, df['lift_coefficient'])
        axes[1, 0].set_title('Lift Coefficient')
        axes[1, 0].set_ylabel('Cl')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(time_s, df['drag_coefficient'])
        axes[1, 1].set_title('Drag Coefficient')
        axes[1, 1].set_ylabel('Cd')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].grid(True)
        
        # Cl vs Cd (drag polar)
        axes[1, 2].scatter(df['drag_coefficient'], df['lift_coefficient'], alpha=0.6)
        axes[1, 2].set_title('Drag Polar')
        axes[1, 2].set_xlabel('Cd')
        axes[1, 2].set_ylabel('Cl')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_live_data(self):
        """Create live updating plots"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Live Pressure Data', fontsize=14)
        
        def animate(frame):
            if not self.live_data['timestamp']:
                return
            
            # Clear axes
            for ax in axes.flat:
                ax.clear()
            
            time_data = np.array(self.live_data['timestamp']) / 1000.0  # Convert to seconds
            time_data = time_data - time_data[0] if len(time_data) > 0 else time_data
            
            # Velocity
            axes[0, 0].plot(time_data, self.live_data['velocity_mph'])
            axes[0, 0].set_title('Velocity')
            axes[0, 0].set_ylabel('mph')
            axes[0, 0].grid(True)
            
            # Pressures
            axes[0, 1].plot(time_data, self.live_data['dynamic_pressure'], label='Dynamic')
            axes[0, 1].plot(time_data, self.live_data['static_pressure'], label='Static')
            axes[0, 1].set_title('Pressures')
            axes[0, 1].set_ylabel('Pa')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Coefficients
            axes[1, 0].plot(time_data, self.live_data['lift_coefficient'])
            axes[1, 0].set_title('Lift Coefficient')
            axes[1, 0].set_ylabel('Cl')
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].grid(True)
            
            axes[1, 1].plot(time_data, self.live_data['drag_coefficient'])
            axes[1, 1].set_title('Drag Coefficient')
            axes[1, 1].set_ylabel('Cd')
            axes[1, 1].set_xlabel('Time (s)')
            axes[1, 1].grid(True)
        
        ani = animation.FuncAnimation(fig, animate, interval=100, cache_frame_data=False)
        plt.show()
        return ani
    
    def calculate_pressure_distribution(self, data=None):
        """Calculate pressure coefficient distribution"""
        if data is None:
            data = self.recorded_data
            
        if not data:
            return None
        
        df = pd.DataFrame(data)
        
        # Calculate pressure coefficients for each measurement point
        # Cp = (P_local - P_static) / (0.5 * rho * V^2)
        cp_data = []
        
        for _, row in df.iterrows():
            if row['dynamic_pressure'] > 0:
                cp_values = []
                for i in range(1, 5):  # drag_pressure_1 through drag_pressure_4
                    cp = (row[f'drag_pressure_{i}'] - row['static_pressure']) / row['dynamic_pressure']
                    cp_values.append(cp)
                
                cp_data.append({
                    'timestamp': row['timestamp'],
                    'velocity_mph': row['velocity_mph'],
                    'cp_1': cp_values[0],
                    'cp_2': cp_values[1],
                    'cp_3': cp_values[2],
                    'cp_4': cp_values[3]
                })
        
        return pd.DataFrame(cp_data)
    
    def plot_pressure_distribution(self, data=None):
        """Plot pressure coefficient distribution"""
        cp_df = self.calculate_pressure_distribution(data)
        if cp_df is None or len(cp_df) == 0:
            print("No data available for pressure distribution")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Time history of Cp values
        time_s = (cp_df['timestamp'] - cp_df['timestamp'].iloc[0]) / 1000.0
        
        for i in range(1, 5):
            ax1.plot(time_s, cp_df[f'cp_{i}'], label=f'Port {i}')
        
        ax1.set_title('Pressure Coefficient vs Time')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Cp')
        ax1.legend()
        ax1.grid(True)
        
        # Average Cp distribution
        port_positions = [0.25, 0.5, 0.75, 1.0]  # Normalized positions along chord
        avg_cp = [cp_df[f'cp_{i}'].mean() for i in range(1, 5)]
        std_cp = [cp_df[f'cp_{i}'].std() for i in range(1, 5)]
        
        ax2.errorbar(port_positions, avg_cp, yerr=std_cp, marker='o', capsize=5)
        ax2.set_title('Average Pressure Distribution')
        ax2.set_xlabel('x/c (Normalized Position)')
        ax2.set_ylabel('Cp')
        ax2.grid(True)
        ax2.invert_yaxis()  # Typical convention for Cp plots
        
        plt.tight_layout()
        plt.show()

class AerodynamicExperiment:
    """High-level class for running complete aerodynamic experiments"""
    
    def __init__(self, port='COM3'):
        self.daq = PressureDataAcquisition(port=port)
        self.experiment_data = {}
        
    def setup_experiment(self, model_name, test_conditions):
        """Setup experiment parameters"""
        self.model_name = model_name
        self.test_conditions = test_conditions
        
        # Configure DAQ system
        self.daq.configure_system(**test_conditions)
        
        print(f"Experiment setup for model: {model_name}")
        print(f"Test conditions: {test_conditions}")
    
    def run_velocity_sweep(self, velocities, duration_per_point=30):
        """Run experiment at multiple velocities"""
        if not self.daq.connect():
            return False
        
        # Calibrate sensors
        input("Ensure zero pressure conditions and press Enter to calibrate...")
        self.daq.calibrate_sensors()
        
        results = {}
        
        for velocity in velocities:
            print(f"\nTesting at {velocity} mph...")
            input(f"Set wind tunnel to {velocity} mph and press Enter to start measurement...")
            
            # Record data for specified duration
            if self.daq.start_recording(duration=duration_per_point):
                time.sleep(duration_per_point + 2)  # Wait for completion
                
                # Analyze this velocity point
                data = self.daq.recorded_data.copy()
                df = pd.DataFrame(data)
                
                if len(df) > 0:
                    results[velocity] = {
                        'mean_velocity': df['velocity_mph'].mean(),
                        'mean_lift_coeff': df['lift_coefficient'].mean(),
                        'mean_drag_coeff': df['drag_coefficient'].mean(),
                        'std_lift_coeff': df['lift_coefficient'].std(),
                        'std_drag_coeff': df['drag_coefficient'].std(),
                        'dynamic_pressure': df['dynamic_pressure'].mean(),
                        'raw_data': data
                    }
                    
                    print(f"  Cl = {results[velocity]['mean_lift_coeff']:.4f} ± {results[velocity]['std_lift_coeff']:.4f}")
                    print(f"  Cd = {results[velocity]['mean_drag_coeff']:.4f} ± {results[velocity]['std_drag_coeff']:.4f}")
        
        self.experiment_data['velocity_sweep'] = results
        self.daq.disconnect()
        
        # Save experiment data
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.model_name}_velocity_sweep_{timestamp}"
        
        with open(f"{filename}.json", 'w') as f:
            # Convert data to JSON serializable format
            json_data = {}
            for vel, data in results.items():
                json_data[str(vel)] = {
                    'mean_velocity': data['mean_velocity'],
                    'mean_lift_coeff': data['mean_lift_coeff'],
                    'mean_drag_coeff': data['mean_drag_coeff'],
                    'std_lift_coeff': data['std_lift_coeff'],
                    'std_drag_coeff': data['std_drag_coeff'],
                    'dynamic_pressure': data['dynamic_pressure']
                }
            
            json.dump({
                'model_name': self.model_name,
                'test_conditions': self.test_conditions,
                'results': json_data,
                'timestamp': timestamp
            }, f, indent=2)
        
        print(f"\nExperiment data saved to {filename}.json")
        return results
    
    def plot_experiment_results(self):
        """Plot results from velocity sweep experiment"""
        if 'velocity_sweep' not in self.experiment_data:
            print("No velocity sweep data available")
            return
        
        data = self.experiment_data['velocity_sweep']
        velocities = list(data.keys())
        
        cl_means = [data[v]['mean_lift_coeff'] for v in velocities]
        cl_stds = [data[v]['std_lift_coeff'] for v in velocities]
        cd_means = [data[v]['mean_drag_coeff'] for v in velocities]
        cd_stds = [data[v]['std_drag_coeff'] for v in velocities]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Cl vs Velocity
        ax1.errorbar(velocities, cl_means, yerr=cl_stds, marker='o', capsize=5)
        ax1.set_title('Lift Coefficient vs Velocity')
        ax1.set_xlabel('Velocity (mph)')
        ax1.set_ylabel('Cl')
        ax1.grid(True)
        
        # Cd vs Velocity
        ax2.errorbar(velocities, cd_means, yerr=cd_stds, marker='s', capsize=5)
        ax2.set_title('Drag Coefficient vs Velocity')
        ax2.set_xlabel('Velocity (mph)')
        ax2.set_ylabel('Cd')
        ax2.grid(True)
        
        # Drag Polar
        ax3.errorbar(cd_means, cl_means, xerr=cd_stds, yerr=cl_stds, marker='D', capsize=3)
        ax3.set_title('Drag Polar')
        ax3.set_xlabel('Cd')
        ax3.set_ylabel('Cl')
        ax3.grid(True)
        
        plt.suptitle(f'Aerodynamic Characteristics - {self.model_name}')
        plt.tight_layout()
        plt.show()

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Aerodynamic Pressure Measurement System')
    parser.add_argument('--port', default='COM3', help='Serial port')
    parser.add_argument('--mode', choices=['single', 'record', 'live', 'experiment'], 
                       default='single', help='Operation mode')
    parser.add_argument('--duration', type=int, default=30, help='Recording duration (seconds)')
    parser.add_argument('--file', help='Data file to load and analyze')
    
    args = parser.parse_args()
    
    if args.file:
        # Load and analyze existing data
        daq = PressureDataAcquisition()
        data = daq.load_data(args.file)
        if data:
            daq.analyze_data(data)
            daq.plot_data(data)
    else:
        # Live operation
        daq = PressureDataAcquisition(port=args.port)
        
        if not daq.connect():
            return
        
        if args.mode == 'single':
            result = daq.take_single_measurement()
            if result:
                print("Single measurement:")
                for key, value in result.items():
                    print(f"  {key}: {value}")
        
        elif args.mode == 'record':
            if daq.start_recording(duration=args.duration):
                print("Recording... Press Ctrl+C to stop early")
                try:
                    time.sleep(args.duration + 1)
                except KeyboardInterrupt:
                    pass
                
                daq.stop_recording()
                daq.save_data()
                daq.analyze_data()
                daq.plot_data()
        
        elif args.mode == 'live':
            print("Starting live plotting... Close plot window to stop")
            daq.start_recording()
            ani = daq.plot_live_data()
            plt.show()
            daq.stop_recording()
        
        elif args.mode == 'experiment':
            # Interactive experiment mode
            print("Aerodynamic Experiment Mode")
            model_name = input("Enter model name: ")
            
            # Get test conditions
            air_density = float(input("Air density (kg/m³) [1.225]: ") or 1.225)
            ref_area = float(input("Reference area (m²) [0.01]: ") or 0.01)
            chord_length = float(input("Chord length (m) [0.1]: ") or 0.1)
            
            velocities_str = input("Enter test velocities (mph, comma-separated) [10,15,20,25]: ") or "10,15,20,25"
            velocities = [float(v.strip()) for v in velocities_str.split(',')]
            
            # Setup and run experiment
            experiment = AerodynamicExperiment(port=args.port)
            experiment.setup_experiment(model_name, {
                'air_density': air_density,
                'reference_area': ref_area,
                'chord_length': chord_length
            })
            
            results = experiment.run_velocity_sweep(velocities)
            if results:
                experiment.plot_experiment_results()
        
        daq.disconnect()

if __name__ == "__main__":
    main()