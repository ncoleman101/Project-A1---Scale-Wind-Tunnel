#!/usr/bin/env python3
"""
Wind Tunnel Control System
Python interface for Arduino-controlled wind tunnel
"""

import serial
import time
import threading
import json
from datetime import datetime
import argparse

class WindTunnelController:
    def __init__(self, port='COM3', baudrate=9600):
        """
        Initialize wind tunnel controller
        
        Args:
            port (str): Serial port (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux)
            baudrate (int): Serial communication speed
        """
        self.port = port
        self.baudrate = baudrate
        self.serial_connection = None
        self.is_connected = False
        self.monitoring = False
        self.status_data = {
            'system_enabled': False,
            'current_velocity': 0.0,
            'target_velocity': 0.0,
            'motor_pwm': 0,
            'timestamp': None
        }
        self.data_log = []
        
    def connect(self):
        """Establish serial connection to Arduino"""
        try:
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=2,
                write_timeout=2
            )
            time.sleep(2)  # Wait for Arduino to initialize
            
            # Test connection
            response = self.send_command("STATUS")
            if response:
                self.is_connected = True
                print(f"✓ Connected to wind tunnel controller on {self.port}")
                return True
            else:
                raise Exception("No response from Arduino")
                
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self):
        """Close serial connection"""
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            self.is_connected = False
            print("Disconnected from wind tunnel controller")
    
    def send_command(self, command):
        """
        Send command to Arduino and return response
        
        Args:
            command (str): Command to send
            
        Returns:
            str: Response from Arduino or None if failed
        """
        if not self.is_connected or not self.serial_connection:
            print("Error: Not connected to Arduino")
            return None
            
        try:
            # Clear input buffer
            self.serial_connection.reset_input_buffer()
            
            # Send command
            self.serial_connection.write((command + '\n').encode())
            
            # Read response (wait up to 1 second)
            response_lines = []
            start_time = time.time()
            
            while time.time() - start_time < 1.0:
                if self.serial_connection.in_waiting > 0:
                    line = self.serial_connection.readline().decode().strip()
                    if line:
                        response_lines.append(line)
                        # Stop reading if we get a status line or confirmation
                        if line.startswith("STATUS:") or line in ["System ENABLED", "System DISABLED"] or "set to:" in line:
                            break
                time.sleep(0.01)
            
            return '\n'.join(response_lines) if response_lines else None
            
        except Exception as e:
            print(f"Communication error: {e}")
            return None
    
    def turn_on(self):
        """Enable the wind tunnel system"""
        response = self.send_command("ON")
        if response and "ENABLED" in response:
            print("Wind tunnel system ENABLED")
            return True
        else:
            print("Failed to enable system")
            return False
    
    def turn_off(self):
        """Disable the wind tunnel system"""
        response = self.send_command("OFF")
        if response and "DISABLED" in response:
            print("Wind tunnel system DISABLED")
            return True
        else:
            print("Failed to disable system")
            return False
    
    def set_velocity(self, velocity_mph):
        """
        Set target velocity in mph
        
        Args:
            velocity_mph (float): Target velocity in miles per hour
        """
        if velocity_mph < 0:
            print("Error: Velocity cannot be negative")
            return False
            
        command = f"SET:{velocity_mph}"
        response = self.send_command(command)
        
        if response and "set to:" in response:
            print(f"Target velocity set to {velocity_mph} mph")
            return True
        else:
            print(f"Failed to set velocity to {velocity_mph} mph")
            return False
    
    def get_status(self):
        """Get current system status"""
        response = self.send_command("STATUS")
        
        if response and response.startswith("STATUS:"):
            try:
                # Parse status: STATUS:ON/OFF,current_vel,target_vel,motor_pwm
                status_parts = response.replace("STATUS:", "").split(",")
                
                self.status_data = {
                    'system_enabled': status_parts[0] == "ON",
                    'current_velocity': float(status_parts[1]),
                    'target_velocity': float(status_parts[2]),
                    'motor_pwm': int(status_parts[3]),
                    'timestamp': datetime.now()
                }
                
                return self.status_data.copy()
                
            except (IndexError, ValueError) as e:
                print(f"Error parsing status: {e}")
                return None
        
        return None
    
    def print_status(self):
        """Print formatted status information"""
        status = self.get_status()
        if status:
            print("\n=== Wind Tunnel Status ===")
            print(f"System: {'ENABLED' if status['system_enabled'] else 'DISABLED'}")
            print(f"Current Velocity: {status['current_velocity']:.2f} mph")
            print(f"Target Velocity: {status['target_velocity']:.2f} mph")
            print(f"Motor PWM: {status['motor_pwm']}/255 ({status['motor_pwm']/255*100:.1f}%)")
            print(f"Last Update: {status['timestamp'].strftime('%H:%M:%S')}")
            print("==========================\n")
        else:
            print("Failed to get status")
    
    def start_monitoring(self, interval=1.0):
        """
        Start continuous monitoring of wind tunnel status
        
        Args:
            interval (float): Update interval in seconds
        """
        if self.monitoring:
            print("Monitoring already active")
            return
            
        self.monitoring = True
        
        def monitor_loop():
            while self.monitoring:
                status = self.get_status()
                if status:
                    self.data_log.append(status)
                    # Keep only last 1000 entries
                    if len(self.data_log) > 1000:
                        self.data_log.pop(0)
                time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        print(f"Started monitoring (interval: {interval}s)")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring = False
        print("Stopped monitoring")
    
    def save_log(self, filename=None):
        """Save data log to JSON file"""
        if not filename:
            filename = f"wind_tunnel_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert datetime objects to strings for JSON serialization
        log_data = []
        for entry in self.data_log:
            entry_copy = entry.copy()
            entry_copy['timestamp'] = entry['timestamp'].isoformat()
            log_data.append(entry_copy)
        
        try:
            with open(filename, 'w') as f:
                json.dump(log_data, f, indent=2)
            print(f"Data log saved to {filename}")
            return True
        except Exception as e:
            print(f"Failed to save log: {e}")
            return False

def main():
    """Command line interface for wind tunnel control"""
    parser = argparse.ArgumentParser(description='Wind Tunnel Controller')
    parser.add_argument('--port', default='COM3', help='Serial port (default: COM3)')
    parser.add_argument('--baudrate', type=int, default=9600, help='Baud rate (default: 9600)')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    # Create controller instance
    controller = WindTunnelController(port=args.port, baudrate=args.baudrate)
    
    # Connect to Arduino
    if not controller.connect():
        print("Failed to connect. Check your Arduino connection and port.")
        return
    
    if args.interactive:
        # Interactive mode
        print("\nWind Tunnel Controller - Interactive Mode")
        print("Commands: on, off, set <velocity>, status, monitor, stop, save, quit")
        
        controller.start_monitoring(interval=2.0)
        
        try:
            while True:
                command = input("\n> ").strip().lower()
                
                if command == 'quit' or command == 'exit':
                    break
                elif command == 'on':
                    controller.turn_on()
                elif command == 'off':
                    controller.turn_off()
                elif command.startswith('set '):
                    try:
                        velocity = float(command.split()[1])
                        controller.set_velocity(velocity)
                    except (IndexError, ValueError):
                        print("Usage: set <velocity_mph>")
                elif command == 'status':
                    controller.print_status()
                elif command == 'monitor':
                    controller.print_status()
                elif command == 'stop':
                    controller.turn_off()
                elif command == 'save':
                    controller.save_log()
                else:
                    print("Unknown command. Try: on, off, set <velocity>, status, monitor, stop, save, quit")
                    
        except KeyboardInterrupt:
            print("\nShutting down...")
        
        controller.stop_monitoring()
    else:
        # Just show status
        controller.print_status()
    
    # Cleanup
    controller.disconnect()

if __name__ == "__main__":
    main()