import numpy as np
from scipy.ndimage import gaussian_filter
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

        # Define baseline land temperature - HIGHER VALUE FOR LAND
        baseline_land_temperature = 18.0  # °C at sea level (was 15.0)

        # Apply lapse rate relative to baseline
        is_land = self.sim.elevation > 0
        altitude_km = np.maximum(self.sim.elevation, 0) / 1000.0
        delta_T_altitude = -lapse_rate * altitude_km
        self.sim.temperature_celsius[is_land] = baseline_land_temperature + delta_T_altitude[is_land]

        # Initialize ocean temperatures with latitude-based gradient
        is_ocean = self.sim.elevation <= 0
        ocean_base_temp = 13.0  # Cooler baseline for oceans (was implicitly 20)
        self.sim.temperature_celsius[is_ocean] = np.clip(
            ocean_base_temp - (np.abs(self.sim.latitude[is_ocean]) / 90) * 35,  # Temperature decreases with latitude
            -2,  # Minimum ocean temperature
            28   # Maximum ocean temperature (was 30, reduced slightly)
        )

        # Apply seasonal cycle if desired
        # If summer in Northern hemisphere, add temperature bias
        # This creates expected asymmetry in land vs ocean temperatures
        summer_north = True  # Whether it's summer in the North
        if summer_north:
            # Add temperature bonus to Northern hemisphere, penalty to Southern
            hemisphere_adjustment = np.cos(np.deg2rad(self.sim.latitude)) * 3.0  # 3°C adjustment
            self.sim.temperature_celsius += hemisphere_adjustment
        
        # Clip temperatures to realistic bounds
        self.sim.temperature_celsius = np.clip(self.sim.temperature_celsius, -50.0, 50.0)

        # Calculate normalized temperature for visualization (0-1 range)
        self.sim.temperature_normalized = MapGenerator.normalize_data(self.sim.temperature_celsius)

    def update_land_ocean(self):
        """
        Update temperature fields with improved greenhouse effect,
        including cloud effects and humidity coupling.
        Optimized for performance with vectorized operations and reduced memory allocations.
        """
        try:
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
            
            # Type conversion for better performance
            if self.sim.temperature_celsius.dtype != np.float32:
                self.sim.temperature_celsius = self.sim.temperature_celsius.astype(np.float32)
            
            # --- BASIC LAND-OCEAN MASKS ---
            is_land = self.sim.elevation > 0
            is_ocean = ~is_land
                
            # --- SOLAR INPUT CALCULATION ---
            # Constants - use single precision
            S0 = np.float32(params['solar_constant'])  # Solar constant
            
            # Calculate solar zenith angle effect (vectorized)
            cos_phi = np.cos(self.sim.latitudes_rad).astype(np.float32)
            day_length_factor = np.clip(cos_phi, 0, 1)  # Day/night cycle
            
            # Calculate average insolation (vectorized)
            S_avg = S0 / 4  # Spherical geometry factor
            S_lat = S_avg * day_length_factor
            
            # --- ALBEDO CALCULATION ---
            # Surface type masks
            is_land = self.sim.elevation > 0
            
            # Pre-allocate base albedo array to avoid new allocation
            if not hasattr(self, '_base_albedo') or self._base_albedo.shape != is_land.shape:
                self._base_albedo = np.zeros_like(is_land, dtype=np.float32)
            
            # Apply land/ocean albedo (vectorized)
            np.place(self._base_albedo, is_land, params['albedo_land'])
            np.place(self._base_albedo, ~is_land, params['albedo_ocean'])
            
            # Zenith angle dependent albedo (vectorized)
            zenith_factor = np.clip(1.0 / (cos_phi + 0.1), 1.0, 2.0)
            
            # --- CLOUD EFFECTS ---
            if hasattr(self.sim, 'cloud_cover'):
                # STABILITY CHECK: Handle NaN values in cloud cover
                if np.any(np.isnan(self.sim.cloud_cover)):
                    print("WARNING: NaN values detected in cloud cover, resetting to zero")
                    self.sim.cloud_cover = np.nan_to_num(self.sim.cloud_cover, nan=0.0)
                
                # STABILITY CHECK: Clip cloud cover values
                self.sim.cloud_cover = np.clip(self.sim.cloud_cover, 0.0, 1.0)
                
                # Cloud albedo calculation (vectorized)
                cloud_albedo = np.float32(0.6)
                effective_albedo = (1 - self.sim.cloud_cover) * self._base_albedo * zenith_factor + self.sim.cloud_cover * cloud_albedo
            else:
                effective_albedo = np.clip(self._base_albedo * zenith_factor, 0.0, 0.9)
            
            # --- ABSORBED SOLAR RADIATION ---
            # Reuse array if possible
            if not hasattr(self, '_solar_in') or self._solar_in.shape != S_lat.shape:
                self._solar_in = np.zeros_like(S_lat, dtype=np.float32)
            
            # Calculate absorbed radiation (vectorized)
            np.multiply(S_lat, 1 - effective_albedo, out=self._solar_in)
            self.sim.energy_budget['solar_in'] = float(np.mean(self._solar_in))
            
            # --- GREENHOUSE EFFECT ---
            greenhouse_forcing = self.calculate_greenhouse_effect(params)
            self.sim.energy_budget['greenhouse_effect'] = greenhouse_forcing
            
            # Add cloud greenhouse effect (vectorized)
            if hasattr(self.sim, 'cloud_cover'):
                day_night_factor = 1.0 + (1.0 - day_length_factor) * 1.5
                cloud_greenhouse = 35.0 * self.sim.cloud_cover * day_night_factor
                greenhouse_forcing += cloud_greenhouse
            
            # --- OUTGOING LONGWAVE RADIATION ---
            sigma = np.float32(5.670374419e-8)  # Stefan-Boltzmann constant
            T_kelvin = self.sim.temperature_celsius + 273.15
            
            if hasattr(self.sim, 'cloud_cover'):
                cloud_emissivity_factor = 0.2 + 0.1 * (1.0 - day_length_factor)
                effective_emissivity = params['emissivity'] * (1 - cloud_emissivity_factor * self.sim.cloud_cover)
            else:
                effective_emissivity = params['emissivity']
            
            night_cooling_factor = 0.85 + 0.15 * day_length_factor
            
            # Calculate longwave radiation (vectorized)
            if not hasattr(self, '_longwave_out') or self._longwave_out.shape != T_kelvin.shape:
                self._longwave_out = np.zeros_like(T_kelvin, dtype=np.float32)
            
            np.power(T_kelvin, 4, out=self._longwave_out)
            np.multiply(self._longwave_out, sigma, out=self._longwave_out)
            np.multiply(self._longwave_out, effective_emissivity, out=self._longwave_out)
            np.multiply(self._longwave_out, night_cooling_factor, out=self._longwave_out)
            
            self.sim.energy_budget['longwave_out'] = float(np.mean(self._longwave_out))
            
            # --- NET ENERGY FLUX ---
            # Calculate net flux (vectorized)
            if not hasattr(self, '_net_flux') or self._net_flux.shape != self._solar_in.shape:
                self._net_flux = np.zeros_like(self._solar_in, dtype=np.float32)
            
            np.add(self._solar_in, greenhouse_forcing, out=self._net_flux)
            np.subtract(self._net_flux, self._longwave_out, out=self._net_flux)
            
            # STABILITY CHECK: Remove extreme flux values
            self._net_flux = np.clip(self._net_flux, -1000.0, 1000.0)
            
            self.sim.energy_budget['net_flux'] = float(np.mean(self._net_flux))
            
            # --- HEAT ADVECTION ---
            # Calculate temperature gradients
            dT_dy, dT_dx = np.gradient(self.sim.temperature_celsius, self.sim.grid_spacing_y, self.sim.grid_spacing_x)
            temperature_advection = -(self.sim.u * dT_dx + self.sim.v * dT_dy)
            
            # STABILITY CHECK: Limit advection to reasonable values
            temperature_advection = np.clip(temperature_advection, -5.0, 5.0)
            
            # --- HEAT CAPACITY ---
            # Initialize heat capacity array (reuse if possible)
            if not hasattr(self, '_heat_capacity') or self._heat_capacity.shape != is_land.shape:
                self._heat_capacity = np.zeros_like(is_land, dtype=np.float32)
            
            # Set land/ocean heat capacities (vectorized)
            np.place(self._heat_capacity, is_land, params['heat_capacity_land'])
            np.place(self._heat_capacity, ~is_land, params['heat_capacity_ocean'])
            
            # Adjust for humidity (vectorized)
            if hasattr(self.sim, 'humidity'):
                # STABILITY CHECK: Handle NaN values in humidity
                if np.any(np.isnan(self.sim.humidity)):
                    print("WARNING: NaN values detected in humidity, resetting to reasonable defaults")
                    self.sim.humidity = np.nan_to_num(self.sim.humidity, nan=0.5)
                
                # STABILITY CHECK: Clip humidity values
                self.sim.humidity = np.clip(self.sim.humidity, 0.01, 1.0)
                
                moisture_factor = 1.0 + self.sim.humidity * 0.5
                self._heat_capacity[is_land] *= moisture_factor[is_land]
            
            # STABILITY CHECK: Ensure heat capacity is never zero
            self._heat_capacity = np.maximum(self._heat_capacity, 1.0e4)
            
            # --- TEMPERATURE CHANGE ---
            # Calculate delta T (vectorized)
            if not hasattr(self, '_delta_T') or self._delta_T.shape != self._net_flux.shape:
                self._delta_T = np.zeros_like(self._net_flux, dtype=np.float32)
            
            np.divide(self._net_flux, self._heat_capacity, out=self._delta_T)
            np.multiply(self._delta_T, self.sim.time_step_seconds, out=self._delta_T)
            np.add(self._delta_T, temperature_advection * self.sim.time_step_seconds, out=self._delta_T)
            
            # STABILITY CHECK: Limit temperature change per step
            self._delta_T = np.clip(self._delta_T, -5.0, 5.0)
            
            # Apply temperature change (vectorized)
            np.add(self.sim.temperature_celsius, self._delta_T, out=self.sim.temperature_celsius)
            
            # STABILITY CHECK: Handle any NaN values that may have been created
            if np.any(np.isnan(self.sim.temperature_celsius)):
                print("WARNING: NaN values generated during temperature update, fixing")
                nan_mask = np.isnan(self.sim.temperature_celsius)
                self.sim.temperature_celsius[nan_mask & is_land] = 15.0  # Default land temp
                self.sim.temperature_celsius[nan_mask & ~is_land] = 5.0  # Default ocean temp
            
            # --- APPLY DIFFUSION WITH SEPARATE LAND AND OCEAN TREATMENT ---
            # Treat land and ocean diffusion separately
            temperature_land = np.copy(self.sim.temperature_celsius)
            temperature_ocean = np.copy(self.sim.temperature_celsius)
            
            # Apply different diffusion amounts to land and ocean
            temperature_land = gaussian_filter(temperature_land, sigma=params['atmospheric_heat_transfer'], mode='wrap')
            temperature_ocean = gaussian_filter(temperature_ocean, sigma=params['atmospheric_heat_transfer']*1.5, mode='wrap')
            
            # Recombine land and ocean temperatures
            self.sim.temperature_celsius = np.where(is_land, temperature_land, temperature_ocean).astype(np.float32)
            
            # Apply an additional lateral mixing step along coast lines
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
            
            # Apply gentle global smoothing as a final step
            self.sim.temperature_celsius = gaussian_filter(
                self.sim.temperature_celsius, 
                sigma=0.5  # Reduced final smoothing - just enough to remove any remaining artifacts
            ).astype(np.float32)
            
            # FINAL STABILITY CHECK: Clip temperatures to physically reasonable bounds
            self.sim.temperature_celsius = np.clip(self.sim.temperature_celsius, -100.0, 100.0)
            
            # Track temperature history
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

    def update_ocean(self):
        """Update ocean temperatures with depth layers using vectorized operations"""
        # Ocean mask (vectorized)
        is_ocean = self.sim.elevation <= 0
        
        # Create or update ocean layer temperatures (vectorized approach)
        if not hasattr(self.sim, 'ocean_layers'):
            # Initialize ocean layers with copies of surface temperature
            # Using vectorized slice assignment
            num_layers = 3
            self.sim.ocean_layers = np.zeros((num_layers, self.sim.map_height, self.sim.map_width))
            for i in range(num_layers):
                self.sim.ocean_layers[i] = self.sim.temperature_celsius.copy()
        
        # Calculate temperature gradients (vectorized)
        dT_dy = np.gradient(self.sim.temperature_celsius, axis=0) / self.sim.grid_spacing_y
        dT_dx = np.gradient(self.sim.temperature_celsius, axis=1) / self.sim.grid_spacing_x
        
        # Temperature advection (vectorized)
        temperature_advection = -(self.sim.ocean_u * dT_dx + self.sim.ocean_v * dT_dy)
        
        # Vertical mixing between layers (vectorized)
        layer_diffs = np.diff(self.sim.ocean_layers, axis=0)
        vertical_mixing_rate = 1e-5  # m²/s
        heat_transfer = vertical_mixing_rate * layer_diffs
        
        # Update layer temperatures (vectorized)
        # First layer (surface) gets effects from atmosphere
        self.sim.ocean_layers[0][is_ocean] += (temperature_advection[is_ocean] * self.sim.time_step_seconds)
        
        # Update all layers with vertical mixing (vectorized)
        for i in range(len(self.sim.ocean_layers)-1):
            self.sim.ocean_layers[i] -= heat_transfer[i]
            self.sim.ocean_layers[i+1] += heat_transfer[i]
        
        # Update surface temperature from first ocean layer (vectorized)
        self.sim.temperature_celsius[is_ocean] = self.sim.ocean_layers[0][is_ocean]
        
        # Clip ocean temperatures to realistic values (vectorized)
        self.sim.temperature_celsius[is_ocean] = np.clip(
            self.sim.temperature_celsius[is_ocean], 
            -2,  # Minimum ocean temperature
            30   # Maximum ocean temperature
        )

        # Track ocean heat content for diagnostics
        try:
            # Check if oceans exist and initialize energy_budget keys if needed
            if np.any(is_ocean):
                # Ensure these keys exist in energy_budget
                if 'ocean_heat_content' not in self.sim.energy_budget:
                    self.sim.energy_budget['ocean_heat_content'] = 0.0
                    self.sim.energy_budget['ocean_heat_change'] = 0.0
                    self.sim.energy_budget['ocean_flux'] = 0.0
                    
                # Calculate ocean heat content (simplified)
                ocean_temps = self.sim.temperature_celsius[is_ocean]
                ocean_volume = np.sum(is_ocean) * self.sim.grid_spacing_x * self.sim.grid_spacing_y * 100  # Assume 100m depth
                specific_heat_water = 4186  # J/(kg·K)
                density_water = 1000  # kg/m³
                
                # Total heat content in Joules (using absolute temperature)
                heat_content = np.mean(ocean_temps + 273.15) * ocean_volume * density_water * specific_heat_water
                
                # Calculate heat exchange with atmosphere (simplified)
                # Only calculate if there are both land and ocean cells
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
                
                # Set flag to indicate ocean data is available
                self.sim.ocean_data_available = True
                
        except Exception as e:
            print(f"Error calculating ocean diagnostics: {e}")
            # Don't let diagnostics crash the main simulation
            traceback.print_exc()  # Print the full traceback to help debugging
            self.sim.ocean_data_available = False

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