import numpy as np
import time
import traceback
import os
import sys

class SystemStats:
    """Class to handle system statistics printing and tracking for simulation"""
    
    def __init__(self, sim):
        """Initialize statistics tracker with reference to main simulation"""
        self.sim = sim
        self.last_cycle_time = time.time()
        self.last_values = {}
        self.print_stats_enabled = True  # Enable by default
        self.force_next_update = True  # Force first update
        self.last_update_step = 0  # Track when last stats were calculated
        self.update_frequency = 1  # Update stats every step by default
        
        # Check if terminal supports ANSI escape codes for clear screen
        self.supports_ansi = False
        if sys.platform != 'win32' or 'ANSICON' in os.environ:
            self.supports_ansi = True
            
        # Performance tracking
        self.fps_history = []
        self.last_step_timing = {
            'total': 0,
            'temperature': 0,
            'pressure': 0,
            'wind': 0,
            'precipitation': 0
        }
        
        # Add simulation log tracking (recent messages)
        self.simulation_logs = []
        self.max_log_entries = 5  # Keep last 5 log entries
        
        # Add debugging info
        print(f"SystemStats initialized with print_stats_enabled={self.print_stats_enabled}")
        
    def _get_cloud_bias(self, day_factor):
        """Calculate cloud bias with safety check for cloud_cover attribute"""
        try:
            if hasattr(self.sim, 'cloud_cover') and self.sim.cloud_cover is not None:
                cloud_mean = np.mean(self.sim.cloud_cover) * 100
                cloud_bias = cloud_mean - (self.sim.cloud_cover * (0.7 - day_factor)).mean() * 100
                return cloud_bias
            else:
                return 0.0  # Default value when no cloud data
        except Exception:
            return 0.0  # Safe fallback on error
    
    def log_message(self, message):
        """Add a message to the simulation log and optionally print it immediately"""
        # Add timestamp to message
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        # Add to logs
        self.simulation_logs.append(log_entry)
        
        # Trim logs if they exceed the maximum
        if len(self.simulation_logs) > self.max_log_entries:
            self.simulation_logs = self.simulation_logs[-self.max_log_entries:]
            
        # Only force an immediate update if it's been a while since the last update
        # This prevents too many updates when multiple log messages come in quickly
        current_time = time.time()
        last_update_time = getattr(self, 'last_log_update_time', 0)
        
        # Only print directly if stats are disabled, otherwise queue for stats update
        if not self.print_stats_enabled:
            # If stats aren't being displayed, print log messages directly
            print(log_entry)
        elif current_time - last_update_time > 0.5:  # Limit to one update per 0.5 second for logs
            # Store time of update and force a stats refresh that will include the log
            self.last_log_update_time = current_time
            self.force_next_update = True
            self.print_stats()
    
    def print_stats(self):
        """Print detailed statistics about the simulation"""
        try:
            if not self.print_stats_enabled and not self.force_next_update:
                return
                
            # Only update every few steps to reduce console spam
            if self.sim.time_step <= self.last_update_step and not self.force_next_update:
                return
            
            self.last_update_step = self.sim.time_step
            self.force_next_update = False
            
            # If not running, just skip stats update
            if not hasattr(self.sim, 'simulation_running') or not self.sim.simulation_running:
                return
            
            # Clear the previous output if supported
            if self.supports_ansi:
                print("\033[H\033[J", end="")  # Clear screen and home cursor
            else:
                if sys.platform == 'win32':
                    # Windows-specific clear (but only if detected)
                    try:
                        os.system('cls')
                    except:
                        # If that fails, print separator
                        print("\n" + "="*80 + "\n")
                else:
                    # Just print a separator on other platforms
                    print("\n" + "="*80 + "\n")
                    
            # --- SIMULATION IDENTIFICATION ---
            # Print header text in cyan if color supported
            if self.supports_ansi:
                print("\033[96m===== PLANET SIMULATION STATISTICS =====\033[0m")
            else:
                print("===== PLANET SIMULATION STATISTICS =====")
            
            # --- SIMULATION TIME INFO ---
            # Add simulation time information
            time_info = []
            time_info.append(f"Time Step: {self.sim.time_step}")
            
            # If the simulation has day/hour tracking
            if hasattr(self.sim, 'simulation_day') and hasattr(self.sim, 'hour_of_day'):
                time_info.append(f"Simulation Day: {self.sim.simulation_day}")
                time_info.append(f"Hour of Day: {self.sim.hour_of_day}")
                
                # Print solar factor if available
                if hasattr(self.sim, 'calculate_solar_factor'):
                    solar_factor = self.sim.calculate_solar_factor()
                    time_info.append(f"Solar Factor: {solar_factor:.2f}")
                    
                # Total hours simulated
                if hasattr(self.sim, 'total_simulated_hours'):
                    time_info.append(f"Total Hours: {self.sim.total_simulated_hours}")
            
            print(" | ".join(time_info))
            
            # --- PERFORMANCE METRICS ---
            # Calculate cycle time
            current_time = time.time()
            cycle_time = max(current_time - self.last_cycle_time, 0.000001)  # Prevent division by zero
            self.last_cycle_time = current_time
            
            # Calculate multiple statistics in a single pass where possible
            
            # Temperature stats
            temp_stats = np.array([
                np.mean(self.sim.temperature_celsius),
                np.min(self.sim.temperature_celsius),
                np.max(self.sim.temperature_celsius)
            ])
            avg_temp, min_temp, max_temp = temp_stats
            
            # Pressure stats
            pressure_stats = np.array([
                np.mean(self.sim.pressure),
                np.min(self.sim.pressure),
                np.max(self.sim.pressure)
            ])
            avg_pressure, min_pressure, max_pressure = pressure_stats
            
            # Wind stats (calculate wind_speed once and reuse)
            wind_speed = np.sqrt(self.sim.u**2 + self.sim.v**2)
            wind_stats = np.array([
                np.mean(wind_speed),
                np.min(wind_speed),
                np.max(wind_speed)
            ])
            avg_wind, min_wind, max_wind = wind_stats
            
            # Get energy budget values
            solar_in = self.sim.energy_budget.get('solar_in', 0)
            greenhouse = self.sim.energy_budget.get('greenhouse_effect', 0)
            longwave_out = self.sim.energy_budget.get('longwave_out', 0)
            net_flux = self.sim.energy_budget.get('net_flux', 0)
            
            # Calculate net changes
            if not self.last_values:
                self.last_values = {
                    'temp': avg_temp,
                    'pressure': avg_pressure,
                    'solar': solar_in,
                    'greenhouse': greenhouse,
                    'net_flux': net_flux,
                    'wind': avg_wind
                }
            
            # Calculate all deltas
            temp_change = avg_temp - self.last_values['temp']
            pressure_change = avg_pressure - self.last_values['pressure']
            solar_change = solar_in - self.last_values['solar']
            greenhouse_change = greenhouse - self.last_values['greenhouse']
            net_flux_change = net_flux - self.last_values['net_flux']
            wind_change = avg_wind - self.last_values['wind']

            # Update last values
            self.last_values.update({
                'temp': avg_temp,
                'pressure': avg_pressure,
                'solar': solar_in,
                'greenhouse': greenhouse,
                'net_flux': net_flux,
                'wind': avg_wind
            })
            
            # --- NEW DIAGNOSTIC METRICS ---
            
            # 1. Humidity and Cloud Metrics (if available)
            humidity_stats = ""
            cloud_stats = ""
            
            if hasattr(self.sim, 'humidity'):
                avg_humidity = np.mean(self.sim.humidity) * 100  # Convert to percentage
                min_humidity = np.min(self.sim.humidity) * 100
                max_humidity = np.max(self.sim.humidity) * 100
                
                # Store and calculate change
                if 'humidity' not in self.last_values:
                    self.last_values['humidity'] = avg_humidity
                humidity_change = avg_humidity - self.last_values['humidity']
                self.last_values['humidity'] = avg_humidity
                
                humidity_stats = f"Humidity          | Avg: {avg_humidity:6.1f}% | Min: {min_humidity:6.1f}% | Max: {max_humidity:6.1f}% | Δ: {humidity_change:+6.2f}%"
            
            if hasattr(self.sim, 'cloud_cover'):
                avg_cloud = np.mean(self.sim.cloud_cover) * 100  # Convert to percentage
                max_cloud = np.max(self.sim.cloud_cover) * 100
                cloud_area = np.sum(self.sim.cloud_cover > 0.1) / self.sim.cloud_cover.size * 100  # % area with clouds
                
                # Store and calculate change
                if 'cloud' not in self.last_values:
                    self.last_values['cloud'] = avg_cloud
                cloud_change = avg_cloud - self.last_values.get('cloud', avg_cloud)
                self.last_values['cloud'] = avg_cloud
                
                cloud_stats = f"Cloud Cover       | Avg: {avg_cloud:6.1f}% | Max: {max_cloud:6.1f}% | Area: {cloud_area:6.1f}% | Δ: {cloud_change:+6.2f}%"
            
            # 2. Land vs Ocean Temperature Difference (important for circulation)
            is_land = self.sim.elevation > 0
            is_ocean = ~is_land
            
            # --- SELECTIVE STAT CALCULATION BASED ON SIMULATION SPEED ---
            # If in high-speed mode, calculate less frequently the more complex stats
            high_speed = getattr(self.sim, 'high_speed_mode', False)
            sim_speed = 0
            if hasattr(self.sim, 'simulation_speed'):
                # Handle both regular value and DoubleVar
                if hasattr(self.sim.simulation_speed, 'get'):
                    sim_speed = self.sim.simulation_speed.get()  # Get value from DoubleVar
                else:
                    sim_speed = self.sim.simulation_speed  # Direct value
            
            # For very high speed, calculate complex stats only occasionally
            calculate_complex_stats = not (high_speed and sim_speed >= 12) or (self.sim.time_step % 10 == 0)
            
            if calculate_complex_stats:
                # Calculate land vs ocean temperature difference
                if np.any(is_land) and np.any(is_ocean):
                    land_temp = np.mean(self.sim.temperature_celsius[is_land])
                    ocean_temp = np.mean(self.sim.temperature_celsius[is_ocean])
                    temp_diff = land_temp - ocean_temp
                    
                    # Store and calculate change in differential
                    if 'temp_diff' not in self.last_values:
                        self.last_values['temp_diff'] = temp_diff
                    diff_change = temp_diff - self.last_values['temp_diff']
                    self.last_values['temp_diff'] = temp_diff
                    
                    land_ocean_stats = f"Land-Ocean Temp   | Land: {land_temp:6.1f}°C | Ocean: {ocean_temp:6.1f}°C | Diff: {temp_diff:+6.1f}°C | Δ: {diff_change:+6.2f}°C"
                else:
                    land_ocean_stats = "Land-Ocean Temp   | No valid data"
                
                # Calculate wind patterns - directional bias and vorticity
                u_mean = np.mean(self.sim.u)
                v_mean = np.mean(self.sim.v)
                mean_direction = self.sim.wind_system.calculate_direction(u_mean, v_mean)
                
                # Calculate wind vorticity (curl) - indicator of cyclonic/anticyclonic behavior
                dy = self.sim.grid_spacing_y
                dx = self.sim.grid_spacing_x
                dvdx = np.gradient(self.sim.v, axis=1) / dx
                dudy = np.gradient(self.sim.u, axis=0) / dy
                vorticity = dvdx - dudy
                mean_vorticity = np.mean(vorticity)
                
                # Store and calculate change
                if 'vorticity' not in self.last_values:
                    self.last_values['vorticity'] = mean_vorticity
                vorticity_change = mean_vorticity - self.last_values['vorticity']
                self.last_values['vorticity'] = mean_vorticity
                
                # Cache the computed values for use when not calculating complex stats
                self._cached_land_ocean_stats = f"Land-Ocean Temp   | Land: {land_temp:6.1f}°C | Ocean: {ocean_temp:6.1f}°C | Diff: {temp_diff:+6.1f}°C | Δ: {diff_change:+6.2f}°C"
                self._cached_wind_pattern_stats = f"Wind Patterns     | Dir: {mean_direction:5.1f}° | Vorticity: {mean_vorticity:+7.2e} | Δ: {vorticity_change:+7.2e}"
                
                # Assign the wind_pattern_stats variable to use in output
                wind_pattern_stats = self._cached_wind_pattern_stats
            else:
                # Use cached values for complex stats when in high-speed mode
                if hasattr(self, '_cached_land_ocean_stats'):
                    land_ocean_stats = self._cached_land_ocean_stats
                else:
                    land_ocean_stats = "Land-Ocean Temp   | [Calculating...]"
                    
                if hasattr(self, '_cached_wind_pattern_stats'):
                    wind_pattern_stats = self._cached_wind_pattern_stats
                else:
                    wind_pattern_stats = "Wind Patterns     | [Calculating...]"
            
            # 4. Energy balance check (incoming vs outgoing)
            energy_in = solar_in + greenhouse
            energy_out = self.sim.energy_budget.get('longwave_out', 0)
            energy_imbalance = energy_in - energy_out
            imbalance_percent = np.abs(energy_imbalance) / energy_in * 100 if energy_in > 0 else 0

            # Calculate day/night effect for context
            cos_phi = np.cos(self.sim.latitudes_rad)
            day_factor = np.clip(np.mean(cos_phi), 0, 1)  # 0=full night, 1=full day

            # Only show warning if imbalance is high AND it's not due to normal day/night cycle
            # Higher threshold at night (15% vs 10% during day)
            night_threshold = 15.0  # Higher threshold at night when imbalance is expected
            day_threshold = 10.0    # Lower threshold during day

            # Dynamic threshold based on day/night cycle
            threshold = night_threshold * (1.0 - day_factor) + day_threshold * day_factor

            warning = "⚠️ " if imbalance_percent > threshold else ""

            energy_balance = f"Energy Balance    | In: {energy_in:6.1f} W/m² | Out: {energy_out:6.1f} W/m² | Imbalance: {energy_imbalance:+6.1f} W/m² | {warning}{imbalance_percent:4.1f}%"
            
            # 5. Physical Stability Indicators
            temp_variability = np.std(self.sim.temperature_celsius)
            pressure_variability = np.std(self.sim.pressure) / avg_pressure * 100  # Percentage variability
            
            # Calculate rate of change - looking for explosive instabilities
            temp_rate = abs(temp_change / cycle_time) if cycle_time > 0 else 0
            pressure_rate = abs(pressure_change / cycle_time) if cycle_time > 0 else 0
            
            # Warning flags
            temp_warning = "⚠️ " if temp_rate > 0.1 else ""  # Warning if temp changing >0.1°C per second
            pressure_warning = "⚠️ " if pressure_rate > 10 else ""  # Warning if pressure changing >10 Pa per second
            
            stability_stats = f"Stability Check   | Temp Δ/s: {temp_warning}{temp_rate:5.3f}°C | Pres Δ/s: {pressure_warning}{pressure_rate:5.1f}Pa | T-var: {temp_variability:5.2f}°C | P-var: {pressure_variability:5.2f}%"
            
            # 6. Hemisphere asymmetry - sanity check for a realistic climate
            northern_temp = np.mean(self.sim.temperature_celsius[self.sim.latitude > 0])
            southern_temp = np.mean(self.sim.temperature_celsius[self.sim.latitude < 0])
            hemisphere_diff = northern_temp - southern_temp
            
            # Store and calculate change
            if 'hemisphere_diff' not in self.last_values:
                self.last_values['hemisphere_diff'] = hemisphere_diff
            hemi_change = hemisphere_diff - self.last_values['hemisphere_diff']
            self.last_values['hemisphere_diff'] = hemisphere_diff
            
            hemisphere_stats = f"Hemisphere Check  | North: {northern_temp:6.1f}°C | South: {southern_temp:6.1f}°C | Diff: {hemisphere_diff:+6.1f}°C | Δ: {hemi_change:+6.2f}°C"
            
            # Calculate FPS with safety check
            fps = min(1/cycle_time, 999.9)  # Cap FPS display at 999.9
            
            # Get simulation speed (hours per update)
            sim_speed = 0
            if hasattr(self.sim, 'simulation_speed'):
                # Handle both regular value and DoubleVar
                if hasattr(self.sim.simulation_speed, 'get'):
                    sim_speed = self.sim.simulation_speed.get()  # Get value from DoubleVar
                else:
                    sim_speed = self.sim.simulation_speed  # Direct value
            
            # Check if high-speed mode is active
            high_speed = getattr(self.sim, 'high_speed_mode', False)
            high_speed_text = " (Approx)" if high_speed and sim_speed >= 12 else ""
            
            # Create the output strings with fixed width
            cycle_str = f"Simulation Cycle: {self.sim.time_step:6d} | Cycle Time: {cycle_time:6.3f}s | FPS: {fps:5.1f} | Sim Speed: {sim_speed:4.1f}h/update{high_speed_text}"
            temp_str = f"Temperature (°C)   | Avg: {avg_temp:6.1f}  | Min: {min_temp:6.1f} | Max: {max_temp:6.1f} | Δ: {temp_change:+6.2f}"
            pres_str = f"Pressure (Pa)      | Avg: {avg_pressure:8.0f} | Min: {min_pressure:8.0f} | Max: {max_pressure:8.0f} | Δ: {pressure_change:+6.0f}"
            wind_str = f"Wind Speed (m/s)   | Avg: {avg_wind:6.1f}  | Min: {min_wind:6.1f} | Max: {max_wind:6.1f} | Δ: {wind_change:+6.2f}"
            
            # Energy budget strings with net changes
            energy_in_str = f"Energy In (W/m²)   | Solar: {solar_in:6.1f} | Δ: {solar_change:+6.2f} | Greenhouse: {greenhouse:6.1f} | Δ: {greenhouse_change:+6.2f}"
            
            # Calculate the number of lines to print (base + new metrics)
            base_lines = 6  # Original number of lines
            extra_lines = 0
            
            extra_metrics = []
            if humidity_stats:
                extra_metrics.append(humidity_stats)
                extra_lines += 1
            if cloud_stats:
                extra_metrics.append(cloud_stats)
                extra_lines += 1
            
            # Always add these metrics
            extra_metrics.extend([
                land_ocean_stats,
                wind_pattern_stats,
                stability_stats,
                
                # 1. DIURNAL CYCLE TRACKING
                # Calculate day/night phase more precisely
                f"Day-Night Status   | Phase: {'Day' if day_factor > 0.5 else 'Night'} | Sol. Factor: {day_factor:0.2f} | Cloud Bias: {self._get_cloud_bias(day_factor):+.1f}%",
                
                # 2. ENERGY FLUX BREAKDOWN
                # Calculate percentage contribution of each energy component
                f"Energy Components  | Solar: {solar_in/max(energy_in, 1e-5)*100:4.1f}% | GHG: {greenhouse/max(energy_in, 1e-5)*100:4.1f}% | Cloud: {self.sim.energy_budget.get('cloud_effect', 0)/max(energy_in, 1e-5)*100:4.1f}% | Imbalance: {energy_imbalance/max(energy_in, 1e-5)*100:+4.1f}%",
                
                # Add the energy balance line here (moved from above)
                energy_balance,
                
                # 3. HUMIDITY TRANSPORT METRICS
                # Add moisture budget if humidity available
                f"Moisture Budget   | Evap: {self.sim.energy_budget.get('evaporation', 0):5.2f} | Precip: {self.sim.energy_budget.get('precipitation', 0):5.2f} | Transport: {self.sim.energy_budget.get('humidity_transport', 0):+5.2f} | Balance: {(self.sim.energy_budget.get('evaporation', 0) - self.sim.energy_budget.get('precipitation', 0)):+5.2f}" if hasattr(self.sim, 'humidity') else "Moisture Budget   | No humidity data available",
                
                # 4. OCEAN-ATMOSPHERE HEAT EXCHANGE
                # Track ocean heat content and exchange
                f"Ocean Heat        | Flux: {self.sim.energy_budget.get('ocean_flux', 0):+6.1f} W/m² | Storage: {self.sim.energy_budget.get('ocean_heat_content', 0):.1e} J/m² | Change: {self.sim.energy_budget.get('ocean_heat_change', 0):+5.2f}%" if hasattr(self.sim, 'ocean_data_available') and self.sim.ocean_data_available else "Ocean Heat        | No ocean data available",
                
                hemisphere_stats
            ])
            extra_lines += 9  # Added 9 new diagnostic lines
            
            total_lines = base_lines + extra_lines
            
            # Print each line without cursor repositioning
            all_stats_lines = [
                cycle_str,
                temp_str,
                pres_str,
                wind_str,
                energy_in_str,
            ] + extra_metrics
            
            # Print each line normally - no repositioning
            for line in all_stats_lines:
                print(line)
            
            # Include other energy exchanges if available
            if 'land_atmosphere_exchange' in self.sim.energy_budget:
                land_atm = self.sim.energy_budget['land_atmosphere_exchange']
                other_str = f"Land-Atm Flux: {land_atm:+.2f} W/m²"
                print(other_str)
            
            # --- SIMULATION LOGS ---
            if hasattr(self, 'simulation_logs') and self.simulation_logs:
                if self.supports_ansi:
                    print("\033[96m----- SIMULATION LOGS -----\033[0m")
                else:
                    print("----- SIMULATION LOGS -----")
                
                # Print each log entry
                for log in self.simulation_logs:
                    print(log)
            
            # Force flush stdout to ensure display updates immediately
            sys.stdout.flush()
            
        except Exception as e:
            print(f"Error in print_system_stats: {e}")
            traceback.print_exc() 