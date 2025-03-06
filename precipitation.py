import numpy as np
import traceback
from scipy.ndimage import gaussian_filter

class Precipitation:
    """Class responsible for managing precipitation, humidity, and cloud dynamics"""
    
    def __init__(self, sim):
        """Initialize the precipitation system with reference to main simulation"""
        self.sim = sim
        
        # Main precipitation and humidity fields
        self.humidity = None
        self.precipitation = None
        self.cloud_cover = None
        
        # Internal fields for calculations
        self._prev_precipitation = None
        self._slope_x = None
        self._slope_y = None
        self._relative_humidity = None
        
    def initialize(self):
        """Initialize spatial humidity map"""
        # Base humidity values
        equator_humidity = 0.8  # High humidity near the equator
        pole_humidity = 0.4     # Lower humidity near the poles

        # Create a latitude-based gradient for humidity
        # Higher humidity near the equator, decreasing towards the poles
        latitude_humidity_gradient = pole_humidity + (equator_humidity - pole_humidity) * np.cos(np.deg2rad(self.sim.latitude))

        # Enhance humidity near oceans
        is_ocean = self.sim.elevation <= 0
        ocean_humidity = 0.7  # Higher humidity over oceans
        land_humidity = 0.5   # Moderate humidity over land

        # Initialize humidity map based on land and ocean
        self.humidity = np.where(is_ocean, ocean_humidity, land_humidity)

        # Blend with latitude gradient for more realism
        blend_factor = 0.4  # Increased influence of latitude on land humidity
        self.humidity = self.humidity * (1 - blend_factor) + latitude_humidity_gradient * blend_factor

        # Add random variability for more natural distribution
        noise = np.random.normal(loc=0.0, scale=0.05, size=self.humidity.shape)
        self.humidity += noise
        self.humidity = np.clip(self.humidity, 0.4, 0.9)  # Ensure reasonable humidity range
        
        # Initialize precipitation field
        self.precipitation = np.zeros((self.sim.map_height, self.sim.map_width), dtype=np.float32)
        self._prev_precipitation = np.zeros_like(self.precipitation)
        
        # Initialize cloud cover
        self.update_clouds()

        print("Humidity map initialized.")
        
    def update(self):
        """Update humidity, precipitation, and cloud cover"""
        try:
            # Initialize if needed
            if self.humidity is None:
                self.initialize()
                
            # Reset visualization cache since precipitation data will change
            if hasattr(self.sim, '_cached_precip_image'):
                self.sim._cached_precip_image = None
                
            # Ensure time_step_seconds is defined
            time_step_seconds = getattr(self.sim, 'time_step_seconds', 3600.0)  # Default to 1 hour
            
            # --- SURFACE EVAPORATION ---
            # Ocean evaporation is temperature dependent
            is_ocean = self.sim.elevation <= 0
            is_land = ~is_ocean
            
            # Calculate saturation vapor pressure at current temperature
            T = self.sim.temperature_celsius
            
            # Use the temperature module's method
            saturation_vapor_pressure = self.sim.temperature.calculate_water_vapor_saturation(T)
            
            # Vectorized evaporation calculation
            evaporation_base_rate = np.full_like(self.humidity, 0.005, dtype=np.float32)  # Start with land rate
            evaporation_base_rate[is_ocean] = 0.02  # Set ocean rate
            
            # Temperature effect on evaporation
            T_ref = 15.0  # Reference temperature
            temp_factor = 1.0 + 0.07 * (T - T_ref)
            np.clip(temp_factor, 0.2, 5.0, out=temp_factor)  # In-place clipping
            
            # Combine factors into evaporation rate - mm/hour
            evaporation_rate = evaporation_base_rate * temp_factor
            
            # --- PRECIPITATION ---
            # Calculate relative humidity
            vapor_pressure = self.humidity * saturation_vapor_pressure
            relative_humidity = vapor_pressure / saturation_vapor_pressure
            
            # Calculate precipitation rate based on RH
            precipitation_threshold = 0.7
            new_precipitation_rate = np.zeros_like(relative_humidity, dtype=np.float32)
            precip_mask = relative_humidity > precipitation_threshold
            
            if np.any(precip_mask):
                # Precipitation rate increases with excess RH
                excess_humidity = relative_humidity[precip_mask] - precipitation_threshold
                new_precipitation_rate[precip_mask] = excess_humidity**2 * 15.0  # mm/hour
            
            # Apply orographic effect (more rain on mountains facing wind)
            # Calculate terrain slope if needed
            if self._slope_x is None:
                dy, dx = np.gradient(self.sim.elevation)
                self._slope_x = dx / self.sim.grid_spacing_x
                self._slope_y = dy / self.sim.grid_spacing_y
            
            # Orographic factor (precipitation enhancement on windward slopes)
            orographic_factor = np.zeros_like(new_precipitation_rate, dtype=np.float32)
            
            # Apply only on land
            if np.any(is_land):
                # Wind direction effect (wind flowing uphill causes more rain)
                wind_upslope = -(self.sim.u * self._slope_x + self.sim.v * self._slope_y)
                # Only enhance when wind blows uphill
                enhancement_mask = is_land & (wind_upslope > 0)
                if np.any(enhancement_mask):
                    orographic_factor[enhancement_mask] = wind_upslope[enhancement_mask] * 0.5
            
            # Apply orographic enhancement
            orographic_enhancement = np.zeros_like(new_precipitation_rate, dtype=np.float32)
            orographic_enhancement[is_land] = orographic_factor[is_land]
            new_precipitation_rate += orographic_enhancement
            
            # Add persistence to precipitation (rain doesn't stop immediately)
            precipitation_persistence = 0.85  # Higher value means more persistent rain patterns
            
            # Use spatial smoothing to soften transitions
            smoothed_precip = gaussian_filter(new_precipitation_rate, sigma=2.0)
            
            # Smooth the precipitation field over time to reduce stark day/night contrasts
            self.precipitation = (precipitation_persistence * self._prev_precipitation + 
                                 (1 - precipitation_persistence) * smoothed_precip)
            
            # Additional smoothing to further reduce stark contrasts between regions
            self.precipitation = gaussian_filter(self.precipitation, sigma=1.5)
            
            # Store current precipitation for next cycle
            self._prev_precipitation = self.precipitation.copy()
            
            # --- CLOUD FORMATION ---
            # Update cloud cover
            self.update_clouds()
            
            # --- UPDATE HUMIDITY ---
            # Use actual precipitation for humidity changes (not smoothed precipitation)
            total_factor = time_step_seconds
            humidity_change = (evaporation_rate - new_precipitation_rate) * total_factor
            self.humidity += humidity_change
            
            # Apply diffusion and constraints
            self.humidity = gaussian_filter(self.humidity, sigma=1.0)
            np.clip(self.humidity, 0.01, 1.0, out=self.humidity)  # In-place clipping
            
            # --- TRACK WATER CYCLE BUDGET ---
            # Store diagnostic values in the simulation's energy budget
            self.sim.energy_budget['evaporation'] = float(np.mean(evaporation_rate))
            self.sim.energy_budget['precipitation'] = float(np.mean(self.precipitation))
            
        except Exception as e:
            print(f"Error updating precipitation: {e}")
            traceback.print_exc()
    
    def update_clouds(self):
        """Calculate cloud cover based on humidity and temperature"""
        try:
            # Basic cloud formation based on humidity
            if self.humidity is None:
                self.cloud_cover = np.zeros_like(self.sim.temperature_celsius)
                return
            
            # Scale humidity 0.5-1.0 to cloud cover 0-1
            self.cloud_cover = np.clip(self.humidity - 0.5, 0, 1) * 2
            
            # Additional cloud formation factors could be added here
            # For example, temperature effects, altitude effects, etc.
            
        except Exception as e:
            print(f"Error updating clouds: {e}")
            traceback.print_exc()
            if self.cloud_cover is None:
                self.cloud_cover = np.zeros_like(self.sim.temperature_celsius)
    
    def get_relative_humidity(self):
        """Calculate relative humidity from absolute humidity"""
        try:
            if self.humidity is None:
                return np.zeros_like(self.sim.temperature_celsius)
                
            # Calculate saturation vapor pressure
            T = self.sim.temperature_celsius
            saturation_vapor_pressure = self.sim.temperature.calculate_water_vapor_saturation(T)
            
            # Calculate actual vapor pressure and RH
            vapor_pressure = self.humidity * saturation_vapor_pressure
            relative_humidity = vapor_pressure / saturation_vapor_pressure
            
            return relative_humidity
            
        except Exception as e:
            print(f"Error calculating relative humidity: {e}")
            return np.zeros_like(self.sim.temperature_celsius) 