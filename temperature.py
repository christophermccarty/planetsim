import numpy as np
from scipy.ndimage import gaussian_filter, zoom
import traceback
from scipy.ndimage import binary_dilation
from map_generation import MapGenerator

class Temperature:
    def __init__(self, sim):
        """Initialize temperature module with reference to main simulation"""
        self.sim = sim
        
        # Optimization: Precompute and cache temperature-dependent values
        self._temp_cache = {}
        self._saturation_vapor_cache = None
        self._temp_range_for_cache = None
        
        # Optimization: Arrays for temperature calculations
        self._base_albedo = None
        self._solar_in = None
        self._longwave_out = None
        self._net_flux = None
        self._heat_capacity = None
        self._delta_T = None
        
        # Adaptive resolution settings
        self._adaptive_mode = False
        self._downsampling_factor = 2  # Default reduction factor
        self._downsampled_fields = None
        
        # Two-Layer Model settings (Solution 1)
        self._update_counter = 0
        self._fast_component_update_freq = 1  # Update every step
        self._slow_component_update_freq = 3  # Update every 3 steps
        
        # Cache for slow-changing components
        self._greenhouse_cache = None
        self._albedo_cache = None
        self._heat_exchange_cache = None

    def initialize(self):
        """Initialize temperature field with improved baseline for real-world values, including greenhouse effect."""
        # Constants
        S0 = 1361  # Solar constant in W/m²
        albedo = 0.3  # Earth's average albedo
        sigma = 5.670374419e-8  # Stefan-Boltzmann constant
        lapse_rate = 6.5  # K per km
        humidity_effect_coefficient = 0.1  # K per unit humidity

        # Calculate effective temperature
        S_avg = S0 / 4
        cos_phi = np.clip(np.cos(self.sim.latitudes_rad), 0, None)
        S_lat = S_avg * cos_phi
        T_eff = (((1 - albedo) * S_lat) / sigma) ** 0.25

        # Calculate greenhouse temperature adjustment
        climate_sensitivity = 2.0  # K per W/m²
        greenhouse_temperature_adjustment = climate_sensitivity * self.sim.total_radiative_forcing

        # Calculate humidity adjustment
        humidity_effect_adjustment = humidity_effect_coefficient * self.sim.humidity

        # Total surface temperature
        T_surface = T_eff + greenhouse_temperature_adjustment + humidity_effect_adjustment

        # Convert to Celsius
        self.sim.temperature_celsius = T_surface - 273.15  # Convert Kelvin to Celsius

        # Create more realistic temperature distribution based on latitude
        # Start with a smooth latitude-based temperature gradient for the entire globe
        equator_temp = 30.0  # Maximum temperature at equator (°C)
        pole_temp = -25.0   # Minimum temperature at poles (°C)
        
        # Create smooth latitude gradient using cosine function
        base_temp_gradient = equator_temp * np.cos(np.radians(self.sim.latitude)) + pole_temp * (1 - np.cos(np.radians(np.abs(self.sim.latitude))))
        
        # Identify land and ocean cells
        is_land = self.sim.elevation > 0
        is_ocean = ~is_land
        
        # Land temperatures with altitude adjustment
        altitude_km = np.maximum(self.sim.elevation, 0) / 1000.0
        altitude_adjustment = -lapse_rate * altitude_km
        
        # Apply land/ocean adjustments
        land_temp = base_temp_gradient + altitude_adjustment
        ocean_temp = base_temp_gradient * 0.8  # Oceans have less extreme temperatures
        
        # Apply constraints for realism
        land_temp = np.clip(land_temp, -50.0, 45.0)
        ocean_temp = np.clip(ocean_temp, -2.0, 30.0)  # Ocean doesn't get below freezing easily
        
        # Combine land and ocean temperatures
        self.sim.temperature_celsius = np.where(is_land, land_temp, ocean_temp)
        
        # Simulate seasonal effects
        # Summer in Northern hemisphere (adds asymmetry)
        hemisphere_bias = 5.0  # Degrees C
        seasonal_adjustment = hemisphere_bias * np.sin(np.radians(self.sim.latitude))
        self.sim.temperature_celsius += seasonal_adjustment
        
        # Apply final smoothing for natural transitions
        self.sim.temperature_celsius = gaussian_filter(self.sim.temperature_celsius, sigma=2.0, mode='wrap')
        
        # Final clipping to ensure realistic bounds
        self.sim.temperature_celsius = np.clip(self.sim.temperature_celsius, -50.0, 45.0)

        # Calculate normalized temperature for visualization (0-1 range)
        self.sim.temperature_normalized = MapGenerator.normalize_data(self.sim.temperature_celsius)
        
        print("Temperature field initialized with realistic gradients.")

    def update_land_ocean(self):
        """
        Update temperature fields with improved greenhouse effect,
        including cloud effects and humidity coupling.
        Optimized using two-layer thermal model approach.
        """
        try:
            # Check if high-speed mode is active
            high_speed = getattr(self.sim, 'high_speed_mode', False)
            
            # Update adaptive mode status
            if high_speed and high_speed != self._adaptive_mode:
                self._adaptive_mode = True
                # Clear cached fields when mode changes
                self._downsampled_fields = None
                
            elif not high_speed and self._adaptive_mode:
                self._adaptive_mode = False
                self._downsampled_fields = None
                
            # Use adaptive resolution in high-speed mode
            if self._adaptive_mode:
                self._update_land_ocean_adaptive()
                return
                
            # Regular full-resolution update continues below
            
            # Initialize arrays if they don't exist
            if not hasattr(self, '_base_albedo') or self._base_albedo is None:
                self._base_albedo = np.zeros_like(self.sim.elevation, dtype=np.float32)
                
            if not hasattr(self, '_solar_in') or self._solar_in is None:
                self._solar_in = np.zeros_like(self.sim.temperature_celsius, dtype=np.float32)
                
            if not hasattr(self, '_longwave_out') or self._longwave_out is None:
                self._longwave_out = np.zeros_like(self.sim.temperature_celsius, dtype=np.float32)
                
            if not hasattr(self, '_net_flux') or self._net_flux is None:
                self._net_flux = np.zeros_like(self.sim.temperature_celsius, dtype=np.float32)
                
            if not hasattr(self, '_heat_capacity') or self._heat_capacity is None:
                self._heat_capacity = np.zeros_like(self.sim.elevation, dtype=np.float32)
                
            if not hasattr(self, '_delta_T') or self._delta_T is None:
                self._delta_T = np.zeros_like(self.sim.temperature_celsius, dtype=np.float32)
            
            # Increment update counter
            self._update_counter += 1
                
            # STABILITY CHECK: Handle NaN values in temperature at the beginning
            if np.any(np.isnan(self.sim.temperature_celsius)):
                print("WARNING: NaN values detected in temperature, resetting to reasonable defaults")
                is_land = self.sim.elevation > 0
                nan_mask = np.isnan(self.sim.temperature_celsius)
                self.sim.temperature_celsius[nan_mask & is_land] = 15.0  # Default land temp
                self.sim.temperature_celsius[nan_mask & ~is_land] = 5.0  # Default ocean temp
            
            # STABILITY CHECK: Clip extreme temperatures before calculations
            self.sim.temperature_celsius = np.clip(self.sim.temperature_celsius, -100.0, 100.0)
            
            params = self.sim.climate_params
            
            # Add default values for missing parameters
            if 'snow_temp_threshold' not in params:
                params['snow_temp_threshold'] = 0.0  # Default snow temperature threshold (°C)
            if 'ice_temp_threshold' not in params:
                params['ice_temp_threshold'] = -2.0  # Default sea ice temperature threshold (°C)
            if 'albedo_snow' not in params:
                params['albedo_snow'] = 0.8  # Default snow albedo
            if 'albedo_ice' not in params:
                params['albedo_ice'] = 0.6  # Default ice albedo
            if 'albedo_cloud_factor' not in params:
                params['albedo_cloud_factor'] = 0.2  # Default cloud albedo effect
            
            # Type conversion for better performance
            if self.sim.temperature_celsius.dtype != np.float32:
                self.sim.temperature_celsius = self.sim.temperature_celsius.astype(np.float32)
            
            # --- BASIC LAND-OCEAN MASKS ---
            is_land = self.sim.elevation > 0
            is_ocean = ~is_land
            
            # --- FAST COMPONENTS (UPDATED EVERY STEP) ---
            # These components change rapidly and affect visual appearance
            
            # 1. SOLAR INPUT CALCULATION - Fast component (daily cycle)
            S0 = np.float32(params['solar_constant'])  # Solar constant
            
            # Use the simulation's current hour of day to get solar factor
            solar_factor = 1.0
            if hasattr(self.sim, 'calculate_solar_factor'):
                # Get solar factor based on the current hour of day
                solar_factor = self.sim.calculate_solar_factor()
                
            # Calculate solar zenith angle effect (vectorized)
            cos_phi = np.cos(self.sim.latitudes_rad).astype(np.float32)
            
            # Combine latitude effect with time-of-day solar factor
            day_length_factor = np.clip(cos_phi * solar_factor, 0, 1)  # Day/night cycle
            
            # Calculate average insolation (vectorized)
            S_avg = S0 / 4
            S_lat = S_avg * day_length_factor  # Modified to use combined day/night factor
                   
            # 2. CALCULATE ALBEDO - Use cached values when possible
            should_update_albedo = (self._update_counter % self._slow_component_update_freq == 0) or (self._albedo_cache is None)
            
            if should_update_albedo:
                # Full albedo calculation including cloud effects
                if self._base_albedo.shape != is_land.shape:
                    self._base_albedo = np.zeros_like(is_land, dtype=np.float32)
                
                # Set default albedo values based on land/ocean (vectorized)
                np.place(self._base_albedo, is_land, params['albedo_land'])
                np.place(self._base_albedo, is_ocean, params['albedo_ocean'])
                
                # Apply snow/ice albedo effect based on temperature (vectorized)
                cold_land = is_land & (self.sim.temperature_celsius < params['snow_temp_threshold'])
                cold_ocean = is_ocean & (self.sim.temperature_celsius < params['ice_temp_threshold'])
                
                # Partial snow/ice cover based on temperature
                land_snow_factor = np.clip(1.0 - (self.sim.temperature_celsius - params['snow_temp_threshold']) / 5.0, 0, 1)
                ocean_ice_factor = np.clip(1.0 - (self.sim.temperature_celsius - params['ice_temp_threshold']) / 2.0, 0, 1)
                
                # Apply snow albedo to cold land with transition
                if np.any(cold_land):
                    snow_albedo_effect = (params['albedo_snow'] - params['albedo_land']) * land_snow_factor[cold_land]
                    self._base_albedo[cold_land] += snow_albedo_effect
                
                # Apply ice albedo to cold ocean with transition
                if np.any(cold_ocean):
                    ice_albedo_effect = (params['albedo_ice'] - params['albedo_ocean']) * ocean_ice_factor[cold_ocean]
                    self._base_albedo[cold_ocean] += ice_albedo_effect
                
                # Add cloud albedo if available (vectorized)
                if hasattr(self.sim, 'cloud_cover') and self.sim.cloud_cover is not None:
                    # Cloud reflectivity varies by type, use a simple parameterization
                    cloud_albedo_effect = params['albedo_cloud_factor'] * self.sim.cloud_cover
                    self._base_albedo += cloud_albedo_effect
                    self.sim.energy_budget['cloud_effect'] = float(np.mean(cloud_albedo_effect * S_lat))
                
                # STABILITY CHECK: Ensure albedo is within physical limits
                self._base_albedo = np.clip(self._base_albedo, 0.05, 0.95)
                
                # Save calculated values
                self._albedo_cache = np.copy(self._base_albedo)
            else:
                # Reuse cached values
                self._base_albedo = self._albedo_cache
            
            # 3. CALCULATE SOLAR INPUT - Fast component
            albedo = self._base_albedo
            self._solar_in = S_lat * (1 - albedo)
            
            # Store average for budget tracking
            self.sim.energy_budget['solar_in'] = float(np.mean(self._solar_in))
            
            # 4. OUTGOING LONGWAVE RADIATION - Fast component (depends on current temperature)
            # Convert to Kelvin for Stefan-Boltzmann
            temperature_K = self.sim.temperature_celsius + 273.15
            
            # Calculate outgoing radiation (vectorized)
            sigma = np.float32(5.670374419e-8)  # Stefan-Boltzmann constant
            self._longwave_out = sigma * temperature_K**4
            self.sim.energy_budget['longwave_out'] = float(np.mean(self._longwave_out))
            
            # --- SLOW COMPONENTS (UPDATED LESS FREQUENTLY) ---
            should_update_greenhouse = (self._update_counter % self._slow_component_update_freq == 0) or (self._greenhouse_cache is None)
            
            if should_update_greenhouse:
                # 5. GREENHOUSE EFFECT - Slow component
                greenhouse_forcing = self._calculate_greenhouse_effect() * params['greenhouse_strength']
                self._greenhouse_cache = greenhouse_forcing
            else:
                # Use cached values
                greenhouse_forcing = self._greenhouse_cache
            
            # 6. NET ENERGY FLUX - Combine fast and slow components
            if not hasattr(self, '_net_flux') or self._net_flux.shape != self._solar_in.shape:
                self._net_flux = np.zeros_like(self._solar_in, dtype=np.float32)
            
            np.add(self._solar_in, greenhouse_forcing, out=self._net_flux)
            np.subtract(self._net_flux, self._longwave_out, out=self._net_flux)
            
            # STABILITY CHECK: Remove extreme flux values
            self._net_flux = np.clip(self._net_flux, -1000.0, 1000.0)
            self.sim.energy_budget['net_flux'] = float(np.mean(self._net_flux))
            
            # 7. HEAT ADVECTION - Fast component (depends on current winds and temperature)
            dT_dy, dT_dx = np.gradient(self.sim.temperature_celsius, self.sim.grid_spacing_y, self.sim.grid_spacing_x)

            # Calculate wind speed for scaling advection strength
            wind_speed = np.sqrt(self.sim.u**2 + self.sim.v**2)
            max_wind = np.max(wind_speed)
            if max_wind > 0:
                # Normalize wind speed to 0-1 range for scaling
                wind_strength = wind_speed / max_wind
            else:
                wind_strength = np.zeros_like(wind_speed)

            # Scale factor for realistic advection (stronger at higher wind speeds)
            # Real atmospheric advection can cause temperature changes of 10-15°C in 24 hours
            # during strong frontal passages
            advection_scale = 3.0 + 7.0 * wind_strength  # 3-10°C range based on wind strength

            # Calculate advection with enhanced scale factor
            temperature_advection = -(self.sim.u * dT_dx + self.sim.v * dT_dy) * advection_scale

            # More generous limits for advection to allow realistic frontal passages
            # Allow up to 15°C/day for strong winds, which is realistic for major weather systems
            temperature_advection = np.clip(temperature_advection, -15.0, 15.0)

            # STABILITY CHECK: Limit advection to reasonable values
            # But allow larger changes than before to properly represent heat transport
            # Only clip truly extreme values that would be physically unrealistic
            
            # 8. HEAT CAPACITY - Slow component
            should_update_heat_capacity = (self._update_counter % self._slow_component_update_freq == 0) or (self._heat_capacity is None)
            
            if should_update_heat_capacity:
                # Calculate heat capacity based on surface type
                if not hasattr(self, '_heat_capacity') or self._heat_capacity.shape != is_land.shape:
                    self._heat_capacity = np.zeros_like(is_land, dtype=np.float32)
                
                # Set land/ocean heat capacities (vectorized)
                np.place(self._heat_capacity, is_land, params['heat_capacity_land'])
                np.place(self._heat_capacity, ~is_land, params['heat_capacity_ocean'])
            
            # 9. TEMPERATURE CHANGE - Fast component
            if not hasattr(self, '_delta_T') or self._delta_T.shape != self._net_flux.shape:
                self._delta_T = np.zeros_like(self._net_flux, dtype=np.float32)
            
            # Ensure heat capacity is never zero to avoid division by zero
            heat_capacity_safe = np.maximum(self._heat_capacity, 1.0)  # Minimum heat capacity of 1.0
            
            # Calculate raw temperature change based on energy flux
            np.divide(self._net_flux, heat_capacity_safe, out=self._delta_T)
            np.multiply(self._delta_T, self.sim.time_step_seconds, out=self._delta_T)
            np.add(self._delta_T, temperature_advection * self.sim.time_step_seconds, out=self._delta_T)
            
            # STABILITY CHECK: Limit temperature change per step (more strict in early steps)
            # Use more conservative limits for first few updates to prevent initial distortion
            if self._update_counter <= 5:
                # Very strict limits for the first few steps
                max_change = 0.5  # Maximum 0.5°C change per step initially
                self._delta_T = np.clip(self._delta_T, -max_change, max_change)
            else:
                # Normal limits after initial stabilization
                self._delta_T = np.clip(self._delta_T, -5.0, 5.0)
            
            # Apply altitude effects on land temperature (lapse rate)
            if is_land.any():
                # Standard atmospheric lapse rate is around 6.5°C per km
                lapse_rate = 6.5  # °C/km
                altitude_km = np.maximum(self.sim.elevation, 0) / 1000.0
                altitude_adjustment = -lapse_rate * altitude_km
                
                # Apply altitude adjustment to delta_T for land areas only
                altitude_effect = altitude_adjustment - np.where(is_land, self.sim.temperature_celsius, 0)
                # Apply a gradual correction (5% per step) to avoid sudden jumps
                altitude_correction = altitude_effect * 0.05
                # Only apply to land areas
                self._delta_T[is_land] += altitude_correction[is_land]
            
            # Apply a much weaker latitude-based constraint to avoid completely unrealistic temperatures
            # Only apply 1% correction to extreme latitudes (>70°) and only when temperatures are very far from expected
            if self._update_counter % 5 == 0:  # Only apply occasionally
                abs_lat = np.abs(self.sim.latitude)
                extreme_lat_mask = abs_lat > 70
                
                if np.any(extreme_lat_mask):
                    # Very broad temperature boundaries based on latitude
                    high_lat_max_temp = 25.0  # Maximum reasonable temperature at high latitudes
                    high_lat_min_temp = -70.0  # Minimum reasonable temperature at high latitudes
                    
                    # Only apply corrections for truly extreme values
                    too_hot_mask = extreme_lat_mask & (self.sim.temperature_celsius > high_lat_max_temp)
                    too_cold_mask = extreme_lat_mask & (self.sim.temperature_celsius < high_lat_min_temp)
                    
                    # Apply very gentle corrections (1% per step)
                    if np.any(too_hot_mask):
                        hot_correction = (high_lat_max_temp - self.sim.temperature_celsius[too_hot_mask]) * 0.01
                        self._delta_T[too_hot_mask] += hot_correction
                        
                    if np.any(too_cold_mask):
                        cold_correction = (high_lat_min_temp - self.sim.temperature_celsius[too_cold_mask]) * 0.01
                        self._delta_T[too_cold_mask] += cold_correction
            
            # 10. APPLY TEMPERATURE CHANGE - Fast component
            np.add(self.sim.temperature_celsius, self._delta_T, out=self.sim.temperature_celsius)
            
            # STABILITY CHECK: Handle any NaN values that may have been created
            if np.any(np.isnan(self.sim.temperature_celsius)):
                print("WARNING: NaN values generated during temperature update, fixing")
                nan_mask = np.isnan(self.sim.temperature_celsius)
                self.sim.temperature_celsius[nan_mask & is_land] = 15.0  # Default land temp
                self.sim.temperature_celsius[nan_mask & ~is_land] = 5.0  # Default ocean temp
            
            # 11. APPLY DIFFUSION - Medium frequency component (can be optimized)
            # Apply diffusion every update rather than occasionally to maintain gradients
            # Treat land and ocean diffusion separately
            temperature_land = np.copy(self.sim.temperature_celsius)
            temperature_ocean = np.copy(self.sim.temperature_celsius)
            
            # Apply different diffusion amounts to land and ocean
            # Increase sigma values for more aggressive smoothing
            temperature_land = gaussian_filter(temperature_land, sigma=params['atmospheric_heat_transfer'] * 1.5, mode='wrap')
            temperature_ocean = gaussian_filter(temperature_ocean, sigma=params['atmospheric_heat_transfer'] * 2.5, mode='wrap')
            
            # Recombine land and ocean temperatures
            self.sim.temperature_celsius = np.where(is_land, temperature_land, temperature_ocean).astype(np.float32)
            
            # Apply coastal mixing for better land-ocean transitions
            coastal_mask = np.zeros_like(self.sim.elevation, dtype=bool)
            
            # Define a simple kernel for finding coastal regions
            kernel = np.ones((3, 3), dtype=bool)
            kernel[1, 1] = False
            
            # Find cells that are land but adjacent to ocean
            coastal_land = is_land & binary_dilation(is_ocean, structure=kernel)
            
            # Find cells that are ocean but adjacent to land
            coastal_ocean = is_ocean & binary_dilation(is_land, structure=kernel)
            
            # Combine both to get all coastal regions
            coastal_mask = coastal_land | coastal_ocean
            
            # Apply a targeted smoothing only to coastal regions
            if np.any(coastal_mask):
                smoothed_coastal = gaussian_filter(self.sim.temperature_celsius, sigma=3.0, mode='wrap')
                # Blend original with smoothed along the coast (50% blend)
                self.sim.temperature_celsius[coastal_mask] = (0.5 * self.sim.temperature_celsius[coastal_mask] + 
                                                     0.5 * smoothed_coastal[coastal_mask])
            
            # Apply stronger global smoothing as a final step to ensure gradients are natural
            # Use stronger smoothing in early steps
            if self._update_counter <= 10:
                sigma = 2.0  # Strong initial smoothing
            else:
                sigma = 1.5  # Normal smoothing after stabilization
            
            self.sim.temperature_celsius = gaussian_filter(
                self.sim.temperature_celsius, 
                sigma=sigma
            ).astype(np.float32)
            
            # 12. FINAL STABILITY CHECK - Fast component
            # Tighter temperature constraints to avoid unrealistic extremes
            # Even stricter for early updates
            if self._update_counter <= 5:
                # Very tight constraints initially
                self.sim.temperature_celsius = np.clip(self.sim.temperature_celsius, -60.0, 60.0)
            else:
                # Expanded constraints after initial stabilization
                self.sim.temperature_celsius = np.clip(self.sim.temperature_celsius, -90.0, 70.0)
                
            # Add a weaker coupling between land and ocean temperatures
            land_temp = self.sim.temperature_celsius[is_land]
            ocean_temp = self.sim.temperature_celsius[is_ocean]

            # Skip if either land or ocean is missing
            if not np.any(is_land) or not np.any(is_ocean):
                return

            # Calculate mean temperatures
            land_mean = np.mean(land_temp)
            ocean_mean = np.mean(ocean_temp)

            # Apply a smaller correction to reduce only extreme differences
            # Reduced from 2% to 0.5% to allow more natural variations
            correction_factor = 0.005  # 0.5% adjustment per step
            if abs(land_mean - ocean_mean) > 30:  # Only activate for extreme differences (30° vs 15°)
                if land_mean - ocean_mean > 30:  # If land much warmer than ocean
                    self.sim.temperature_celsius[is_land] -= (land_mean - ocean_mean - 30) * correction_factor
                    self.sim.temperature_celsius[is_ocean] += (land_mean - ocean_mean - 30) * correction_factor * 0.5
                elif ocean_mean - land_mean > 30:  # If ocean much warmer than land
                    self.sim.temperature_celsius[is_land] += (ocean_mean - land_mean - 30) * correction_factor
                    self.sim.temperature_celsius[is_ocean] -= (ocean_mean - land_mean - 30) * correction_factor * 0.5
            
            # Update the normalized temperature for visualization
            # This ensures the visualization stays current
            self.sim.temperature_normalized = MapGenerator.normalize_data(self.sim.temperature_celsius)
            
            # 13. TRACK TEMPERATURE HISTORY - Slow component
            if self._update_counter % self._slow_component_update_freq == 0:
                if 'temperature_history' not in self.sim.energy_budget:
                    self.sim.energy_budget['temperature_history'] = []
                
                # Append current average temperature to history
                avg_temp = float(np.mean(self.sim.temperature_celsius))
                # Only append if it's a valid value
                if not np.isnan(avg_temp) and not np.isinf(avg_temp):
                    self.sim.energy_budget['temperature_history'].append(avg_temp)
            
        except Exception as e:
            print(f"Error updating temperature: {e}")
            traceback.print_exc()

    def _calculate_greenhouse_effect(self):
        """
        Calculate greenhouse effect based on atmospheric composition and conditions,
        with realistic latitude-dependent variations.
        """
        params = self.sim.climate_params
        
        # Provide default value if parameter doesn't exist
        if 'base_greenhouse_effect' not in params:
            base_greenhouse = 150.0  # Default base greenhouse effect in W/m²
        else:
            base_greenhouse = params['base_greenhouse_effect']
        
        # Add default for cloud greenhouse factor if missing
        if 'cloud_greenhouse_factor' not in params:
            params['cloud_greenhouse_factor'] = 25.0  # Default cloud greenhouse effect
        
        # Add default for humidity greenhouse factor if missing
        if 'humidity_greenhouse_factor' not in params:
            params['humidity_greenhouse_factor'] = 0.15  # Default humidity greenhouse effect
        
        # Create latitude-dependent base greenhouse effect
        # In reality, greenhouse effect is stronger in tropics and weaker at poles
        abs_latitude = np.abs(self.sim.latitude)
        
        # Calculate latitude scaling factors for different greenhouse components
        # Base scaling: maximum at equator (100%), reduced at poles (70%)
        base_latitude_scaling = 0.7 + 0.3 * np.cos(np.radians(abs_latitude))
        
        # Water vapor scaling: more dramatic drop-off at high latitudes due to cold, dry air
        water_vapor_scaling = 0.3 + 0.7 * np.cos(np.radians(abs_latitude))**2
        
        # Apply scaling to base greenhouse effect - 40% from non-water vapor GHGs (CO2, CH4, etc)
        # and 60% from water vapor (typical Earth proportions)
        base_ghg_component = base_greenhouse * 0.4 * base_latitude_scaling  # Non-water vapor GHGs
        water_vapor_component = base_greenhouse * 0.6 * water_vapor_scaling  # Water vapor component
        
        # Combine components for base greenhouse effect
        base_greenhouse_latitude = base_ghg_component + water_vapor_component
        
        # Start with latitude-dependent base greenhouse effect
        greenhouse_forcing = base_greenhouse_latitude
        
        # Add cloud greenhouse effect if available
        if hasattr(self.sim, 'cloud_cover') and self.sim.cloud_cover is not None:
            # Cloud greenhouse effect varies with latitude (stronger at poles than tropics)
            # This is a simplification of the complex cloud radiative effects
            cloud_latitude_factor = 1.0 + 0.5 * np.cos(self.sim.latitudes_rad)
            cloud_greenhouse = params['cloud_greenhouse_factor'] * self.sim.cloud_cover * cloud_latitude_factor
            greenhouse_forcing += cloud_greenhouse
            self.sim.energy_budget['cloud_greenhouse'] = float(np.mean(cloud_greenhouse))
        
        # Add humidity feedback if available (supplemental to baseline water vapor component)
        if hasattr(self.sim, 'humidity') and self.sim.humidity is not None:
            # Apply latitude-dependent humidity effect
            humidity_values = np.clip(self.sim.humidity, 0.01, 1.0)
            humidity_effect = humidity_values * params['humidity_greenhouse_factor'] * water_vapor_scaling
            greenhouse_forcing += humidity_effect
            self.sim.energy_budget['humidity_greenhouse'] = float(np.mean(humidity_effect))
        
        # Add seasonal variation in greenhouse effect
        # Northern/Southern hemisphere asymmetry
        if hasattr(self.sim, 'current_day'):
            # Simplified seasonal cycle (stronger greenhouse in local summer hemisphere)
            day_of_year = self.sim.current_day % 365
            season_factor = np.sin(2 * np.pi * (day_of_year / 365))
            hemisphere_factor = np.sign(self.sim.latitude) * season_factor * 0.1
            greenhouse_forcing *= (1 + hemisphere_factor)
        
        # Track components in energy budget
        self.sim.energy_budget['greenhouse_effect'] = float(np.mean(greenhouse_forcing))
        self.sim.energy_budget['gh_equator_pole_ratio'] = float(
            np.mean(greenhouse_forcing[abs_latitude < 10]) / 
            np.mean(greenhouse_forcing[abs_latitude > 70])
        )
        
        return greenhouse_forcing

    def _update_land_ocean_adaptive(self):
        """
        Lower resolution version of temperature update for high-speed mode
        Uses downsampling to speed up calculations while preserving key dynamics
        """
        try:
            # Get parameters
            params = self.sim.climate_params
            
            # Determine downsampling factor based on grid size
            grid_height, grid_width = self.sim.temperature_celsius.shape
            
            # Adaptive downsampling factor (larger grids get more downsampling)
            if grid_height > 360:  # Very high-res
                self._downsampling_factor = 4
            elif grid_height > 180:  # High-res
                self._downsampling_factor = 3
            else:  # Standard res
                self._downsampling_factor = 2
                
            # Skip if grid is already small
            if grid_height < 90 or grid_width < 180:
                # Just use regular update for small grids
                self._update_counter += 1
                return self.update_land_ocean()
                
            # Initialize downsampled fields if needed
            if self._downsampled_fields is None:
                # Create dictionary to store all downsampled fields
                self._downsampled_fields = {}
                
                # Downsample key fields
                ds_factor = self._downsampling_factor
                
                # Essential fields to downsample
                self._downsampled_fields['temperature'] = zoom(self.sim.temperature_celsius, 1/ds_factor, order=1)
                self._downsampled_fields['elevation'] = zoom(self.sim.elevation, 1/ds_factor, order=0)
                
                # Wind fields
                if hasattr(self.sim, 'u') and hasattr(self.sim, 'v'):
                    self._downsampled_fields['u'] = zoom(self.sim.u, 1/ds_factor, order=1)
                    self._downsampled_fields['v'] = zoom(self.sim.v, 1/ds_factor, order=1)
                else:
                    # Default to zero if wind fields don't exist
                    ds_shape = self._downsampled_fields['temperature'].shape
                    self._downsampled_fields['u'] = np.zeros(ds_shape, dtype=np.float32)
                    self._downsampled_fields['v'] = np.zeros(ds_shape, dtype=np.float32)
                    
                # Generate downsampled latitude coordinates
                if hasattr(self.sim, 'latitudes_rad'):
                    ds_shape = self._downsampled_fields['temperature'].shape
                    lat_range = np.linspace(-np.pi/2, np.pi/2, ds_shape[0])
                    self._downsampled_fields['latitudes_rad'] = np.tile(lat_range[:, np.newaxis], (1, ds_shape[1]))
            
            # Increment update counter
            self._update_counter += 1
            
            # ---- Work with downsampled fields ----
            # Extract downsampled fields for easier access
            ds_temp = self._downsampled_fields['temperature']
            ds_elev = self._downsampled_fields['elevation']
            ds_u = self._downsampled_fields['u']
            ds_v = self._downsampled_fields['v']
            ds_lat = self._downsampled_fields['latitudes_rad']
            
            # Create masks
            is_land = ds_elev > 0
            is_ocean = ~is_land
            
            # STABILITY CHECK: Handle NaN values in downsampled temperature
            if np.any(np.isnan(ds_temp)):
                is_land = ds_elev > 0
                nan_mask = np.isnan(ds_temp)
                ds_temp[nan_mask & is_land] = 15.0
                ds_temp[nan_mask & ~is_land] = 5.0
            
            # --- FAST COMPONENTS ---
            # Calculate solar input
            S0 = np.float32(params['solar_constant'])
            
            # Use the simulation's current hour of day to get solar factor
            solar_factor = 1.0
            if hasattr(self.sim, 'calculate_solar_factor'):
                # Get solar factor based on the current hour of day
                solar_factor = self.sim.calculate_solar_factor()
                
            # Calculate solar zenith angle effect (vectorized)
            cos_phi = np.cos(ds_lat).astype(np.float32)
            
            # Combine latitude effect with time-of-day solar factor
            day_length_factor = np.clip(cos_phi * solar_factor, 0, 1)  # Day/night cycle
            
            # Calculate albedo (simplified)
            albedo = np.where(is_land, params['albedo_land'], params['albedo_ocean']).astype(np.float32)
            
            # Simplified snow/ice effect
            cold_land = is_land & (ds_temp < params['snow_temp_threshold'])
            cold_ocean = is_ocean & (ds_temp < params['ice_temp_threshold'])
            albedo[cold_land] = params['albedo_snow']
            albedo[cold_ocean] = params['albedo_ice']
            
            # Calculate average insolation (vectorized)
            S_avg = S0 / 4
            S_lat = S_avg * day_length_factor  # Modified to use combined day/night factor
            solar_in = S_lat * (1 - albedo)
            
            # Simplified greenhouse effect
            base_greenhouse = params['base_greenhouse_effect'] * params['greenhouse_strength']
            
            # Outgoing longwave radiation (Stefan-Boltzmann)
            temperature_K = ds_temp + 273.15
            sigma = np.float32(5.670374419e-8)
            longwave_out = sigma * temperature_K**4
            
            # Net energy flux
            net_flux = solar_in + base_greenhouse - longwave_out
            
            # Heat capacity
            heat_capacity = np.where(is_land, 
                                    params['heat_capacity_land'],
                                    params['heat_capacity_ocean']).astype(np.float32)
            
            # Temperature update
            delta_T = net_flux / heat_capacity * self.sim.time_step_seconds
            
            # Add enhanced advection (apply every step for more realism)
            dT_dy, dT_dx = np.gradient(ds_temp)
            
            # Calculate wind speed for scaling
            wind_speed = np.sqrt(ds_u**2 + ds_v**2)
            max_wind = np.max(wind_speed)
            if max_wind > 0:
                wind_strength = wind_speed / max_wind
            else:
                wind_strength = np.zeros_like(wind_speed)
                
            # Scale advection based on wind strength (similar to full-resolution mode but slightly reduced)
            advection_scale = 2.0 + 5.0 * wind_strength  # 2-7°C range for adaptive mode
            
            # Calculate scaled advection
            temperature_advection = -(ds_u * dT_dx + ds_v * dT_dy) * advection_scale
            
            # Allow more realistic temperature changes but still maintain stability
            temperature_advection = np.clip(temperature_advection, -10.0, 10.0)
            
            # Apply advection to temperature change
            delta_T += temperature_advection * self.sim.time_step_seconds
            
            # Apply altitude effects for land areas (simplified for adaptive mode)
            if np.any(is_land):
                # Standard atmospheric lapse rate effect
                lapse_rate = 6.5  # °C/km
                altitude_km = np.maximum(ds_elev, 0) / 1000.0
                altitude_adjustment = -lapse_rate * altitude_km
                
                # Calculate current deviation from expected altitude-adjusted temperature
                altitude_effect = np.zeros_like(ds_temp)
                altitude_effect[is_land] = (altitude_adjustment[is_land] - ds_temp[is_land])
                
                # Apply a gradual 5% correction toward altitude-appropriate temperature
                altitude_correction = altitude_effect * 0.05
                delta_T[is_land] += altitude_correction[is_land]
            
            # STABILITY CHECK: Limit change
            delta_T = np.clip(delta_T, -2.0, 2.0)
            
            # Apply temperature change
            ds_temp += delta_T
            
            # Simplified diffusion
            if self._update_counter % 2 == 0:
                ds_temp = gaussian_filter(ds_temp, sigma=1.0, mode='wrap')
            
            # STABILITY CHECK: Ensure temperature bounds
            ds_temp = np.clip(ds_temp, -100.0, 100.0)
            
            # Update downsampled array
            self._downsampled_fields['temperature'] = ds_temp
            
            # Upsample temperature back to full resolution
            ds_factor = self._downsampling_factor
            upsampled_temp = zoom(ds_temp, ds_factor, order=1)
            
            # Ensure dimensions match (zoom can sometimes produce off-by-one errors)
            if upsampled_temp.shape != self.sim.temperature_celsius.shape:
                # Crop or pad as needed
                target_shape = self.sim.temperature_celsius.shape
                result_temp = np.zeros(target_shape, dtype=np.float32)
                
                # Copy valid region
                min_h = min(target_shape[0], upsampled_temp.shape[0])
                min_w = min(target_shape[1], upsampled_temp.shape[1])
                result_temp[:min_h, :min_w] = upsampled_temp[:min_h, :min_w]
                
                # Use result as final temperature
                self.sim.temperature_celsius = result_temp
            else:
                # Use upsampled result directly
                self.sim.temperature_celsius = upsampled_temp
            
            # Update energy budget with downsampled metrics
            self.sim.energy_budget['solar_in'] = float(np.mean(solar_in))
            self.sim.energy_budget['longwave_out'] = float(np.mean(longwave_out))
            self.sim.energy_budget['net_flux'] = float(np.mean(net_flux))
            
            # Update temperature history less frequently in adaptive mode
            if self._update_counter % self._slow_component_update_freq == 0:
                if 'temperature_history' not in self.sim.energy_budget:
                    self.sim.energy_budget['temperature_history'] = []
                
                avg_temp = float(np.mean(ds_temp))
                if not np.isnan(avg_temp) and not np.isinf(avg_temp):
                    self.sim.energy_budget['temperature_history'].append(avg_temp)
            
            # --- CONSTRAINTS AND STABILITY ---
            # Ensure temperature bounds
            ds_temp = np.clip(ds_temp, -100.0, 100.0)

            # Apply a minimal latitude-based constraint only for extreme values
            if self._update_counter % 6 == 0:  # Apply even less frequently (every 6 steps)
                # Only apply to extreme latitudes (above 75 degrees) to avoid artificial banding
                abs_lat = np.abs(ds_lat)
                extreme_lat_mask = abs_lat > 1.3  # About 75 degrees latitude
                
                if np.any(extreme_lat_mask):
                    # Only constrain temperatures that are truly extreme for their latitude
                    extreme_hot = ds_temp[extreme_lat_mask] > 25.0  # Unrealistically hot for extreme latitudes
                    extreme_cold = ds_temp[extreme_lat_mask] < -70.0  # Unrealistically cold
                    
                    # Get indices of cells that need correction
                    extreme_indices = np.where(extreme_lat_mask)
                    
                    # Apply very gentle correction (1% per step) only to extremes
                    for i in range(len(extreme_indices[0])):
                        y, x = extreme_indices[0][i], extreme_indices[1][i]
                        
                        if ds_temp[y, x] > 25.0:
                            # Very hot at high latitude, apply gentle cooling
                            ds_temp[y, x] += (25.0 - ds_temp[y, x]) * 0.01
                        elif ds_temp[y, x] < -70.0:
                            # Very cold, apply gentle warming
                            ds_temp[y, x] += (-70.0 - ds_temp[y, x]) * 0.01
            
        except Exception as e:
            print(f"Error in adaptive temperature update: {e}")
            traceback.print_exc()
            # Fall back to standard update if adaptive mode fails
            self._adaptive_mode = False
            self.update_land_ocean()

    def update_ocean(self):
        """
        Update ocean temperatures with depth layers using stratified model approach
        Different depths are updated at different frequencies to match real-world dynamics
        """
        try:
            # Ocean mask (vectorized)
            is_ocean = self.sim.elevation <= 0
            
            # Skip if no ocean
            if not np.any(is_ocean):
                return
            
            # Increment update counter if not already done
            if not hasattr(self, '_update_counter'):
                self._update_counter = 0
            self._update_counter += 1
            
            # Stratified Ocean Model settings (Solution 1)
            surface_update_freq = 1    # Update surface layer every step (fast)
            mixed_update_freq = 2      # Update mixed layer every 2 steps (medium)
            deep_update_freq = 5       # Update deep layer every 5 steps (slow)
            abyssal_update_freq = 10   # Update abyssal layer very rarely (very slow)
            
            # Create or update ocean layer temperatures if needed
            if not hasattr(self.sim, 'ocean_layers') or self.sim.ocean_layers is None:
                # Initialize 4 ocean layers with specific characteristics:
                # 0: Surface layer (0-10m) - responds quickly to atmospheric forcing
                # 1: Mixed layer (10-200m) - moderate response, seasonal thermocline
                # 2: Deep layer (200-1000m) - slow response, permanent thermocline
                # 3: Abyssal layer (>1000m) - very slow response, nearly constant
                num_layers = 4
                self.sim.ocean_layers = np.zeros((num_layers, self.sim.map_height, self.sim.map_width), dtype=np.float32)
                
                # Initialize with decreasing temperature with depth
                # Surface layer starts at surface temperature
                self.sim.ocean_layers[0] = self.sim.temperature_celsius.copy()
                
                # Deeper layers start progressively cooler
                for i in range(1, num_layers):
                    # Each layer is colder based on typical ocean temperature profiles
                    if i == 1:  # Mixed layer (10-200m)
                        temp_offset = -2.0
                    elif i == 2:  # Deep layer (200-1000m)
                        temp_offset = -8.0
                    else:  # Abyssal layer (>1000m)
                        temp_offset = -15.0
                    
                    # Initialize layer with offset from surface, but only in ocean areas
                    layer_temp = self.sim.temperature_celsius.copy()
                    layer_temp[is_ocean] += temp_offset
                    # Clip to realistic ocean temperatures for each layer
                    layer_temp[is_ocean] = np.clip(layer_temp[is_ocean], -2.0, 25.0)
                    self.sim.ocean_layers[i] = layer_temp
                
                # Also initialize layer-specific properties
                if not hasattr(self.sim, 'ocean_layer_properties'):
                    self.sim.ocean_layer_properties = {
                        # Thermal diffusivity decreases with depth (m²/s)
                        'thermal_diffusivity': np.array([5e-5, 2e-5, 8e-6, 2e-6]),
                        # Heat capacity increases with depth (relative values)
                        'heat_capacity_factor': np.array([1.0, 5.0, 10.0, 20.0]),
                        # Layer thickness in meters
                        'thickness': np.array([10.0, 190.0, 800.0, 2000.0])
                    }
            
            # Get layer properties
            layer_props = self.sim.ocean_layer_properties
            diff_coefs = layer_props['thermal_diffusivity']
            heat_caps = layer_props['heat_capacity_factor']
            thicknesses = layer_props['thickness']
            
            # --- SURFACE LAYER UPDATES (EVERY STEP) ---
            # Surface layer interacts directly with atmosphere
            
            # Calculate temperature gradients for surface layer (vectorized)
            surf_temp = self.sim.ocean_layers[0]
            dT_dy, dT_dx = np.gradient(surf_temp, self.sim.grid_spacing_y, self.sim.grid_spacing_x)
            
            # Get ocean currents if available
            if hasattr(self.sim, 'ocean_u') and hasattr(self.sim, 'ocean_v'):
                u_curr = self.sim.ocean_u
                v_curr = self.sim.ocean_v
            else:
                # Default to zero if currents not initialized
                u_curr = np.zeros_like(self.sim.temperature_celsius)
                v_curr = np.zeros_like(self.sim.temperature_celsius)
            
            # Calculate horizontal advection for surface layer with enhanced scaling
            # Ocean currents can cause significant temperature transport (Gulf Stream, etc.)
            current_speed = np.sqrt(u_curr**2 + v_curr**2)
            max_current = np.max(current_speed)
            
            if max_current > 0:
                current_strength = current_speed / max_current
            else:
                current_strength = np.zeros_like(current_speed)
                
            # Scale ocean advection - stronger currents like Gulf Stream can transport ~8°C
            ocean_advection_scale = 2.0 + 6.0 * current_strength
            
            # Calculate scaled surface advection
            surf_advection = -(u_curr * dT_dx + v_curr * dT_dy) * ocean_advection_scale
            
            # Allow stronger changes but maintain stability for ocean
            surf_advection = np.clip(surf_advection, -8.0, 8.0)
            
            # Apply atmospheric forcing to surface layer (air-sea heat exchange)
            if self._update_counter % surface_update_freq == 0:
                # Calculate air-sea temperature difference
                air_temp = self.sim.temperature_celsius.copy()
                air_sea_diff = air_temp - surf_temp
                
                # Air-sea heat exchange coefficient (higher in windy conditions)
                if hasattr(self.sim, 'u') and hasattr(self.sim, 'v'):
                    wind_speed = np.sqrt(self.sim.u**2 + self.sim.v**2)
                    exchange_coef = 1e-5 * (1.0 + wind_speed * 0.1)
                else:
                    exchange_coef = 1e-5
                
                # Calculate heat exchange (faster in high wind, proportional to temperature difference)
                surf_forcing = exchange_coef * air_sea_diff
                
                # Add solar heating to surface layer (penetrates only the surface)
                if hasattr(self, '_solar_in'):
                    # Only a portion of solar radiation penetrates water surface
                    water_absorption = 0.7  # 70% of solar radiation absorbed by water
                    surf_solar_heating = self._solar_in * water_absorption * 1e-6
                    surf_forcing += surf_solar_heating
                
                # Update surface layer with all forcings
                dt = self.sim.time_step_seconds
                surf_temp[is_ocean] += (surf_forcing[is_ocean] + surf_advection[is_ocean]) * dt / heat_caps[0]
            
            # Add latitude-based temperature constraint for oceans to prevent unrealistic banding
            # This ensures ocean temperatures follow a realistic pole-to-equator gradient
            if self._update_counter % surface_update_freq == 0:
                # Calculate ideal ocean temperature profile based on latitude with quadratic cooling toward poles
                abs_lat = np.abs(self.sim.latitude)
                ideal_ocean_temp = 28.0 - 0.012 * abs_lat**2  # Quadratic cooling toward poles
                ideal_ocean_temp = np.clip(ideal_ocean_temp, -2.0, 28.0)  # Realistic ocean limits
                
                # Calculate difference between current ocean surface temperature and ideal profile
                ocean_temp_deviation = self.sim.ocean_layers[0] - ideal_ocean_temp
                
                # Apply a gentle correction force toward the ideal profile (weaker to allow more natural patterns)
                # Reduce corrections significantly (was 5%, now 2%)
                correction_strength = np.ones_like(self.sim.temperature_celsius) * 0.02  # Base 2% correction
                
                # Apply slightly stronger correction only at extreme latitudes (>70)
                high_lat_mask = abs_lat > 70
                for y in range(self.sim.map_height):
                    # Since latitude is constant for each row, check the first element in the row
                    if high_lat_mask[y, 0]:  # Check just the first element in the row
                        correction_strength[y, :] = 0.04  # 4% correction in extreme areas (was 12%)
                
                # Apply correction only to ocean areas
                correction = -ocean_temp_deviation * correction_strength
                self.sim.ocean_layers[0][is_ocean] += correction[is_ocean]
                
                # Also apply a weaker correction to the mixed layer for consistency
                self.sim.ocean_layers[1][is_ocean] += correction[is_ocean] * 0.25  # Reduced from 0.5
            
            # --- MIXED LAYER UPDATES (MEDIUM FREQUENCY) ---
            # Mixed layer has seasonal variations but slower than surface
            
            if self._update_counter % mixed_update_freq == 0:
                mixed_temp = self.sim.ocean_layers[1]
                
                # Calculate vertical heat exchange between surface and mixed layer
                surf_mixed_diff = surf_temp - mixed_temp
                vertical_flux_1 = diff_coefs[0] * surf_mixed_diff / ((thicknesses[0] + thicknesses[1]) / 2)
                
                # Apply to mixed layer (accounting for different heat capacities)
                dt = self.sim.time_step_seconds
                cap_ratio = heat_caps[0] / heat_caps[1]
                mixed_temp[is_ocean] += vertical_flux_1[is_ocean] * dt * cap_ratio
                
                # Also apply horizontal advection for mixed layer (weaker than surface)
                # Calculate gradients for mixed layer
                dT_dy_mixed, dT_dx_mixed = np.gradient(mixed_temp, self.sim.grid_spacing_y, self.sim.grid_spacing_x)
                
                # Scale mixed layer advection (reduced compared to surface but still significant)
                # The mixed layer is affected by major currents but with less intensity
                mixed_advection_scale = 1.5 + 4.0 * current_strength  # 1.5-5.5°C range
                
                # Calculate scaled mixed layer advection
                mixed_advection = -(u_curr * dT_dx_mixed + v_curr * dT_dy_mixed) * mixed_advection_scale
                
                # Apply appropriate limits
                mixed_advection = np.clip(mixed_advection, -6.0, 6.0)
                
                # Apply advection to mixed layer
                mixed_temp[is_ocean] += mixed_advection[is_ocean] * dt / heat_caps[1]
            
            # --- DEEP LAYER UPDATES (SLOW FREQUENCY) ---
            # Deep ocean changes very slowly with permanent thermocline
            
            if self._update_counter % deep_update_freq == 0:
                deep_temp = self.sim.ocean_layers[2]
                abyssal_temp = self.sim.ocean_layers[3]
                
                # Vertical exchange between mixed and deep layers
                mixed_deep_diff = mixed_temp - deep_temp
                vertical_flux_2 = diff_coefs[1] * mixed_deep_diff / ((thicknesses[1] + thicknesses[2]) / 2)
                
                # Apply to deep layer
                dt = self.sim.time_step_seconds
                cap_ratio_2 = heat_caps[1] / heat_caps[2]
                deep_temp[is_ocean] += vertical_flux_2[is_ocean] * dt * cap_ratio_2
                
                # For deep layer, horizontal advection is much weaker and slower
                # Use simplified approach with just diffusion
                if self._update_counter % (deep_update_freq * 2) == 0:
                    # Smooth temperature field to represent slow horizontal mixing
                    deep_ocean = np.zeros_like(deep_temp)
                    deep_ocean[is_ocean] = deep_temp[is_ocean]
                    deep_ocean_smoothed = gaussian_filter(deep_ocean, sigma=1.0, mode='wrap')
                    deep_temp[is_ocean] = deep_ocean_smoothed[is_ocean]
            
            # --- ABYSSAL LAYER UPDATES (VERY SLOW FREQUENCY) ---
            # Abyssal ocean changes extremely slowly, almost constant
            
            if self._update_counter % abyssal_update_freq == 0:
                # Vertical exchange between deep and abyssal layers
                deep_abyssal_diff = deep_temp - abyssal_temp
                vertical_flux_3 = diff_coefs[2] * deep_abyssal_diff / ((thicknesses[2] + thicknesses[3]) / 2)
                
                # Apply to abyssal layer (very small changes)
                dt = self.sim.time_step_seconds
                cap_ratio_3 = heat_caps[2] / heat_caps[3]
                abyssal_temp[is_ocean] += vertical_flux_3[is_ocean] * dt * cap_ratio_3 * 0.5  # Further reduced
            
            # --- CORRECTIVE FEEDBACK TO ENSURE REALISTIC PROFILES ---
            if self._update_counter % mixed_update_freq == 0:
                # Ensure temperature decreases with depth (or remains the same)
                num_layers = len(self.sim.ocean_layers) if hasattr(self.sim, 'ocean_layers') else 4
                for i in range(num_layers - 1):
                    # Calculate where deeper layer is warmer than layer above
                    inverted = self.sim.ocean_layers[i+1] > self.sim.ocean_layers[i]
                    if np.any(inverted & is_ocean):
                        # Reset to layer above temperature (mild cooling)
                        self.sim.ocean_layers[i+1][inverted & is_ocean] = self.sim.ocean_layers[i][inverted & is_ocean] - 0.1
            
            # --- CONSTRAINTS AND STABILITY ---
            # Apply temperature constraints for each layer
            layer_min_temps = [-2.0, -2.0, -2.0, -2.0]
            layer_max_temps = [30.0, 26.0, 20.0, 15.0]
            
            for i in range(num_layers):
                self.sim.ocean_layers[i][is_ocean] = np.clip(
                    self.sim.ocean_layers[i][is_ocean],
                    layer_min_temps[i],
                    layer_max_temps[i]
                )
            
            # Update surface temperature from first ocean layer (CRITICAL FIX)
            # Make sure ocean temperatures are properly copied to the main temperature array
            # This is likely where the 0.00 display issue comes from
            self.sim.temperature_celsius[is_ocean] = self.sim.ocean_layers[0][is_ocean]
            
            # Ensure ocean temperatures are valid (no NaN or zeros)
            if np.any(np.isnan(self.sim.temperature_celsius[is_ocean])):
                self.sim.temperature_celsius[is_ocean] = np.nan_to_num(self.sim.temperature_celsius[is_ocean], nan=5.0)
            
            # Check for any zero ocean temperatures and set to a valid value
            zero_ocean = (self.sim.temperature_celsius == 0.0) & is_ocean
            if np.any(zero_ocean):
                # Get the average temperature of non-zero ocean cells
                valid_ocean = self.sim.temperature_celsius[is_ocean & (self.sim.temperature_celsius != 0.0)]
                mean_ocean_temp = 5.0  # Default if no valid ocean cells
                if len(valid_ocean) > 0:
                    mean_ocean_temp = np.mean(valid_ocean)
                # Set zero ocean cells to the mean ocean temperature
                self.sim.temperature_celsius[zero_ocean] = mean_ocean_temp
            
            # --- DIAGNOSTIC CALCULATIONS ---
            # Track ocean heat content for diagnostics
            try:
                # Ensure these keys exist in energy_budget
                if 'ocean_heat_content' not in self.sim.energy_budget:
                    self.sim.energy_budget['ocean_heat_content'] = 0.0
                    self.sim.energy_budget['ocean_heat_change'] = 0.0
                    self.sim.energy_budget['ocean_flux'] = 0.0
                    self.sim.energy_budget['ocean_layer_temps'] = []
                
                # Calculate total ocean heat content across all layers
                heat_content = 0.0
                layer_avg_temps = []
                
                # Constants
                specific_heat_water = 4186  # J/(kg·K)
                density_water = 1000  # kg/m³
                
                # Ocean area
                ocean_area = np.sum(is_ocean) * self.sim.grid_spacing_x * self.sim.grid_spacing_y
                
                for i in range(num_layers):
                    # Convert to Kelvin for energy calculations
                    layer_temp_K = self.sim.ocean_layers[i][is_ocean] + 273.15
                    avg_temp_K = np.mean(layer_temp_K) if len(layer_temp_K) > 0 else 273.15
                    layer_avg_temps.append(float(avg_temp_K - 273.15))  # Store in Celsius
                    
                    # Calculate heat content for this layer
                    layer_volume = ocean_area * thicknesses[i]
                    layer_heat = avg_temp_K * layer_volume * density_water * specific_heat_water
                    heat_content += layer_heat
                
                # Calculate heat exchange with atmosphere
                if np.any(~is_ocean):
                    air_sea_temp_diff = np.mean(self.sim.temperature_celsius[is_ocean]) - np.mean(self.sim.temperature_celsius[~is_ocean])
                    heat_flux = air_sea_temp_diff * 20  # W/m² per degree difference (simplified)
                else:
                    heat_flux = 0.0
                
                # Calculate percent change safely
                prev_heat = self.sim.energy_budget['ocean_heat_content']
                if prev_heat != 0:
                    heat_change_pct = (heat_content - prev_heat) / abs(prev_heat) * 100
                else:
                    heat_change_pct = 0.0
                
                # Store updated values
                self.sim.energy_budget['ocean_heat_content'] = heat_content
                self.sim.energy_budget['ocean_heat_change'] = heat_change_pct
                self.sim.energy_budget['ocean_flux'] = heat_flux
                self.sim.energy_budget['ocean_layer_temps'] = layer_avg_temps
                
                # Mark ocean data as available for system stats
                self.sim.ocean_data_available = True
                
            except Exception as e:
                print(f"Error calculating ocean heat budget: {e}")
                traceback.print_exc()
        
        except Exception as e:
            print(f"Error updating ocean: {e}")
            traceback.print_exc()

    def calculate_greenhouse_effect(self, params):
        """
        Calculate greenhouse effect with more physically-based parameters.
        Returns forcing in W/m²
        """
        # Base greenhouse gases forcing (W/m²)
        # Values based on IPCC AR6 estimates
        base_forcings = {
            'co2': 5.35 * np.log(self.sim.co2_concentration / 278),  # Pre-industrial CO2 was 278 ppm
            'ch4': 0.036 * (np.sqrt(self.sim.ch4_concentration) - np.sqrt(722)),  # Pre-industrial CH4 was 722 ppb
            'n2o': 0.12 * (np.sqrt(self.sim.n2o_concentration) - np.sqrt(270))   # Pre-industrial N2O was 270 ppb
        }
        
        # Calculate water vapor contribution
        # Water vapor accounts for about 60% of Earth's greenhouse effect
        T_kelvin = self.sim.temperature_celsius + 273.15
        reference_temp = 288.15  # 15°C in Kelvin
        
        # Base water vapor effect (should be ~90 W/m² at 15°C)
        base_water_vapor = 90.0
        
        # Calculate water vapor forcing with temperature dependence
        # Avoid negative values by using exponential relationship
        temp_factor = np.exp(0.07 * (np.mean(T_kelvin) - reference_temp))
        water_vapor_forcing = base_water_vapor * temp_factor
        
        # Other greenhouse gases (ozone, CFCs, etc.)
        other_ghg = 25.0
        
        # Calculate ocean absorption modifier
        ocean_absorption = self.calculate_ocean_co2_absorption()
        
        # Total greenhouse forcing with ocean absorption
        ghg_forcing = sum(base_forcings.values())  # CO2, CH4, N2O
        total_forcing = ((ghg_forcing + water_vapor_forcing + other_ghg) * 
                        params['greenhouse_strength'] * 
                        (1.0 + ocean_absorption))  # Apply ocean absorption modifier
        
        # Store individual components for debugging
        self.sim.energy_budget.update({
            'co2_forcing': base_forcings['co2'],
            'ch4_forcing': base_forcings['ch4'],
            'n2o_forcing': base_forcings['n2o'],
            'water_vapor_forcing': water_vapor_forcing,
            'other_ghg_forcing': other_ghg,
            'ocean_absorption': ocean_absorption,
            'total_greenhouse_forcing': total_forcing
        })

        return total_forcing

    def calculate_ocean_co2_absorption(self):
        """Calculate the amount of CO2 absorbed by oceans and impact on greenhouse effect"""
        try:
            # Get the ocean mask
            ocean_mask = self.sim.elevation < 0
            
            # Get the ocean temperature in Celsius for calculations
            ocean_temp_celsius = self.sim.temperature_celsius[ocean_mask]
            
            # Skip if no ocean
            if not np.any(ocean_mask):
                return 0
            
            # Calculate absorption based on temperature 
            # Cold water absorbs more CO2 than warm water
            # Normalized temperature (0-1 scale, 1 being coldest)
            temp_norm = 1.0 - ocean_temp_celsius / 30  # Assuming ocean temps 0-30C
            temp_norm = np.clip(temp_norm, 0, 1)
            
            # Absorption coefficient (0-1)
            # This simplistic model assumes max absorption at coldest temps
            absorption_coef = temp_norm
            
            # Calculate mean absorption across all oceans
            mean_absorption = np.mean(absorption_coef)
            
            # Scale the greenhouse effect modification
            # At optimal absorption (1.0), reduce greenhouse effect by up to 25%
            greenhouse_modifier = -0.25 * mean_absorption
            
            return greenhouse_modifier
        
        except Exception as e:
            print(f"Error in calculate_ocean_co2_absorption: {e}")
            traceback.print_exc()
            return 0

    def calculate_water_vapor_saturation(self, T):
        """
        Calculate saturation vapor pressure (hPa) based on temperature (°C)
        Uses Bolton's formula for accuracy and efficiency with enhanced stability.
        """
        try:
            # Ensure T is a numpy array for vectorized operations
            if not isinstance(T, np.ndarray):
                T = np.array(T)
            
            # Check for NaN or inf values and replace with reasonable defaults
            bad_values = np.isnan(T) | np.isinf(T)
            if np.any(bad_values):
                print(f"WARNING: Found {np.sum(bad_values)} NaN/Inf values in temperature for vapor saturation")
                # Create a copy to avoid modifying the original array
                T = np.copy(T)
                # Replace bad values with reasonable defaults (15°C)
                T[bad_values] = 15.0
                
            # Hard clip temperature values to prevent numerical issues
            T_clipped = np.clip(T, -50, 50)  # Limit to physically reasonable range
            
            # Initialize cache if needed (do this once for performance)
            if not hasattr(self, '_saturation_vapor_cache') or self._saturation_vapor_cache is None:
                # Create a temperature range covering -50°C to 50°C in 0.1°C steps
                temp_range = np.linspace(-50, 50, 1001)
                
                # Apply Bolton's formula to the range: e_sat = 6.112 * exp(17.67 * T / (T + 243.5))
                self._saturation_vapor_cache = 6.112 * np.exp(17.67 * temp_range / (temp_range + 243.5))
                self._temp_range_for_cache = temp_range
            
            # Convert temperature to index in the cache (0 = -50°C, 1000 = 50°C)
            indices = np.clip(((T_clipped + 50) * 10).astype(np.int32), 0, 1000)
            
            # Look up values in the cache
            result = self._saturation_vapor_cache[indices]
            
            # Final validation: ensure result is within physical limits
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                print("WARNING: Saturation vapor calculation produced invalid results, fixing")
                result = np.nan_to_num(result, nan=6.112, posinf=100.0, neginf=0.1)
                
            return result
            
        except Exception as e:
            print(f"Error in calculate_water_vapor_saturation: {e}")
            # Return a safe default value if there's an error
            if isinstance(T, np.ndarray):
                return np.full_like(T, 6.112)
            else:
                return 6.112

    def calculate_water_density(self, temperature):
        """Calculate water density based on temperature"""
        reference_density = 1000  # kg/m³
        thermal_expansion = 0.0002  # per degree C
        temperature_difference = np.abs(temperature - 4)
        return reference_density * (1 - thermal_expansion * temperature_difference)

    def calculate_relative_humidity(self, vapor_pressure, T):
        """
        Calculate relative humidity from vapor pressure and temperature
        Returns value from 0-1
        """
        return vapor_pressure / self.calculate_water_vapor_saturation(T)
        
    def initialize_ocean_currents(self):
        """Initialize ocean current vectors"""
        # 1. Create ocean mask
        ocean_mask = self.sim.elevation <= 0
        
        # Skip if no ocean
        if not np.any(ocean_mask):
            return
            
        # Create arrays for u and v components if they don't exist or are None
        if not hasattr(self.sim, 'ocean_u') or self.sim.ocean_u is None:
            self.sim.ocean_u = np.zeros((self.sim.map_height, self.sim.map_width))
        else:
            self.sim.ocean_u.fill(0.0)
            
        if not hasattr(self.sim, 'ocean_v') or self.sim.ocean_v is None:
            self.sim.ocean_v = np.zeros((self.sim.map_height, self.sim.map_width))
        else:
            self.sim.ocean_v.fill(0.0)
        
        # 2. Calculate temperature gradients
        # Get temperature differences in x and y directions
        # We'll use these as initial forcing for currents
        temp_grad_y, temp_grad_x = np.gradient(self.sim.temperature_celsius)
        
        # 3. Initialize currents based on temperature gradients
        # Only set currents in ocean cells
        # u-component (east-west)
        self.sim.ocean_u[ocean_mask] = -temp_grad_y[ocean_mask] * 2.0
        
        # v-component (north-south) 
        self.sim.ocean_v[ocean_mask] = temp_grad_x[ocean_mask] * 2.0
        
        # 4. Apply a simple Coriolis-like effect based on latitude
        # Create a latitude effect array
        latitudes = np.linspace(-1, 1, self.sim.map_height)
        lat_effect = np.zeros_like(self.sim.ocean_u)
        
        # Apply latitude effect to each row
        for i, lat in enumerate(latitudes):
            lat_effect[i, :] = lat * 0.01
        
        # Apply the effect only to ocean cells
        ocean_indices = np.where(ocean_mask)
        for i in range(len(ocean_indices[0])):
            y, x = ocean_indices[0][i], ocean_indices[1][i]
            coriolis_factor = lat_effect[y, x] * 0.2
            
            # Store original u value
            temp_u = self.sim.ocean_u[y, x]
            
            # Apply Coriolis-like effect
            self.sim.ocean_u[y, x] -= coriolis_factor * self.sim.ocean_v[y, x]
            self.sim.ocean_v[y, x] += coriolis_factor * temp_u
        
        # 5. Normalize to reasonable values
        # Find the maximum speed
        max_speed = np.sqrt(np.maximum(self.sim.ocean_u**2 + self.sim.ocean_v**2, 1e-10))
        max_speed_val = np.max(max_speed)
        
        if max_speed_val > 0:
            # Scale to ensure max current is 1.0
            scale_factor = 1.0 / max_speed_val
            self.sim.ocean_u *= scale_factor
            self.sim.ocean_v *= scale_factor 