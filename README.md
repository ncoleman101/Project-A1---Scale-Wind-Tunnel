# Open Loop Scale Wind Tunnel

## What This Is?
Developing an open loop wind tunnel design for fluid dynamic testing of model cars and airplanes. This project is designed to be repeatable and inexpensive for home experiments. The inspiration is the SDSU closed loop wind tunnel with the goal being the ability to run experiments as close to professional procedure as possible.

### Design Goals
1. **Create Wind Tunnel Infrastructure**: Design and build converging/ diverging wind tunnel shell using plywood sanded to minimum 500 grit and sealed to reduce friction.
2. **Install Wind Tunnel Fan and Vanes**: Install a box fan in the entrance of wind tunnel to force ambient air into the test section. Install Vanes along converging section of tunnel to keep airflow as laminar as possible
    - Fan diameter must be a minimum of 20 inches across
    - Minimum airflow rate must be 1,500 CFM 
3. **Wiring Fan**: Wire a 3-pin 12V DC Motor Speed Controller (PWM dimmer) into the fans preexisting hardware for variable CFM control. 
4. **Develop DIY Arduino Controlled Pitot Tube System**: Using MLXV7002DP Pressure Sensor, Arduino Uno, Pitot Tube and Tubing to first convert voltage measurement to pressure differential reading.
5. **Create Relay to PC to Read Velocity Measurements**: Program Arduino to convert pressure reading to velocity using scripted code.
6. **Repeat Steps 4 and 5**: Repeat to build enough pitot tubes to be able to create a full pressure and velocity map across the confines of the wind tunnel as well as static pressure of the freestream
7. **Conduct Aerodynamic Experiments on Model Vehicles**: Place various model vehicles in the wind tunnel test section and position pitot tube array to capture data. 
