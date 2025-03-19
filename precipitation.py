import numpy as np
import traceback
from scipy.ndimage import gaussian_filter, zoom
import time

class Precipitation:
    """
    Manages precipitation, humidity, and cloud dynamics in the simulation.
    
    This class handles the water cycle including:
    - Humidity calculation and transport
    - Cloud formation
    - Precipitation events
    - Ocean-atmosphere water exchange
    
    The implementation includes both full physics and statistical update modes
    with enhanced stability mechanisms to prevent oscillation in humidity and cloud cover:
    - Stronger humidity reduction from precipitation (factor of 2.0)
    - Robust temporal smoothing of cloud cover (persistence factor of 0.85)
    - Day-night cycle modulation of evaporation rates (60% reduction at night)
    - Consistent handling across both physics models
    """
    
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
        
        # Adaptive resolution settings
        self._adaptive_mode = False
        self._downsampling_factor = 2  # Default reduction factor
        
        # Statistical Precipitation Patterns (Solution 2)
        self._statistical_mode = True  # Default to statistical mode for better performance
        self._update_counter = 0       # Track updates for alternating between modes
        self._full_physics_freq = 10   # Only do full physics calculation every N steps
        
        # Statistical pattern caches
        self._temp_precip_relation = None  # Cache temperature-precipitation relationship
        self._humidity_precip_relation = None  # Cache humidity-precipitation relationship
        self._wind_precip_relation = None  # Cache wind-precipitation relationship
        self._precip_statistics = {}  # Cache typical precipitation patterns
        
        # Trigger thresholds
        self._monsoon_threshold = 0.85  # High humidity threshold to trigger monsoon physics
        self._storm_threshold = 20.0    # Wind speed threshold to trigger storm physics
        self._extreme_temp_threshold = 35.0  # Temperature threshold for extreme events
        
        # Set to true to force full physics calculation on next update
        self._force_full_physics = False
        
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
        """Update humidity, precipitation, and cloud cover using statistical patterns"""
        try:
            # Initialize if needed
            if self.humidity is None:
                self.initialize()
                
            # Make sure statistical relations are initialized
            if self._temp_precip_relation is None or self._humidity_precip_relation is None or self._wind_precip_relation is None:
                self._initialize_statistical_relations()
                
            # Reset visualization cache since precipitation data will change
            if hasattr(self.sim, '_cached_precip_image'):
                self.sim._cached_precip_image = None
                
            # Check if high-speed mode is active
            high_speed = getattr(self.sim, 'high_speed_mode', False)
            
            # Update adaptive mode status
            if high_speed and high_speed != self._adaptive_mode:
                self._adaptive_mode = True
                
            elif not high_speed and self._adaptive_mode:
                self._adaptive_mode = False
                
            # Use adaptive resolution in high-speed mode
            if self._adaptive_mode:
                self._update_adaptive()
                return
                
            # Increment counter
            self._update_counter += 1
            
            # Determine whether to use full physics or statistical pattern calculation
            use_full_physics = False
            
            # Check for extreme conditions that require full physics
            extreme_conditions = self._check_extreme_conditions()
            
            # Periodic full physics update or extreme conditions detected
            if (self._update_counter % self._full_physics_freq == 0) or extreme_conditions or self._force_full_physics:
                use_full_physics = True
                self._force_full_physics = False
                # Update statistical relations when using full physics
                self._should_update_statistics = True
            else:
                self._should_update_statistics = False
            
            # Choose update method based on conditions
            if use_full_physics:
                self._update_full_physics()
            else:
                self._update_statistical()
            
            # --- UPDATE CLOUD COVER ---
            # Always update clouds with the currently chosen model
            self.update_clouds()
            
            # --- TRACK WATER CYCLE BUDGET ---
            # Store diagnostic values in the simulation's energy budget
            self.sim.energy_budget['evaporation'] = self._get_mean_evaporation_rate()
            self.sim.energy_budget['precipitation'] = float(np.mean(self.precipitation))
            
        except Exception as e:
            print(f"Error updating precipitation: {e}")
            traceback.print_exc()
            
    def _check_extreme_conditions(self):
        """Check for extreme weather conditions that require full physics calculation"""
        try:
            # 1. Check for extreme humidity (monsoon conditions)
            if np.max(self.humidity) > self._monsoon_threshold:
                return True
                
            # 2. Check for extreme winds (storms)
            if hasattr(self.sim, 'u') and hasattr(self.sim, 'v'):
                wind_speed = np.sqrt(self.sim.u**2 + self.sim.v**2)
                if np.max(wind_speed) > self._storm_threshold:
                    return True
                    
            # 3. Check for extreme temperatures
            if np.max(self.sim.temperature_celsius) > self._extreme_temp_threshold:
                return True
                
            # 4. Check for rapidly changing conditions
            if hasattr(self, '_prev_temperature'):
                temp_change = np.max(np.abs(self.sim.temperature_celsius - self._prev_temperature))
                if temp_change > 5.0:  # More than 5°C change
                    return True
            
            # Store current temperature for next comparison
            self._prev_temperature = self.sim.temperature_celsius.copy()
            
            # No extreme conditions detected
            return False
            
        except Exception as e:
            print(f"Error checking for extreme conditions: {e}")
            traceback.print_exc()
            return False
            
    def _update_full_physics(self):
        """Update precipitation using full physics calculation"""
        # This is essentially the original update method logic
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
        evaporation_base_rate = np.full_like(self.humidity, 0.06, dtype=np.float32)  # Increased from 0.02
        evaporation_base_rate[is_ocean] = 0.15  # Increased from 0.05
        
        # Temperature effect on evaporation
        T_ref = 15.0
        temp_factor = 1.0 + 0.07 * (T - T_ref)
        np.clip(temp_factor, 0.2, 5.0, out=temp_factor)  # In-place clipping
        
        # ADDED: Day-night cycle effect on evaporation
        # Get day-night factor from solar angle (0=night, 1=full day)
        day_night_factor = 0.6  # Default if not available
        if hasattr(self.sim, 'latitudes_rad'):
            # Calculate solar angle effect (vectorized) - same calculation as in temperature module
            cos_phi = np.cos(self.sim.latitudes_rad).astype(np.float32)
            day_night_factor = np.clip(cos_phi, 0, 1)  # Day/night cycle
            
            # Calculate average day-night factor across the map
            avg_day_factor = np.mean(day_night_factor)
            
            # Reduce evaporation at night, but don't stop it completely
            # During night (avg_day_factor < 0.3), reduce evaporation to 40-60% of daytime values
            # During day (avg_day_factor > 0.7), use 90-100% of calculated values
            # Apply smoothly in between for dawn/dusk
            night_reduction = 0.6  # Night evaporation is 40-60% of daytime (increased from complete reduction)
            if avg_day_factor < 0.3:  # Night
                day_night_multiplier = night_reduction + (1.0 - night_reduction) * (avg_day_factor / 0.3)
            elif avg_day_factor > 0.7:  # Day
                day_night_multiplier = 0.9 + 0.1 * ((avg_day_factor - 0.7) / 0.3)
            else:  # Dawn/dusk
                transition_factor = (avg_day_factor - 0.3) / 0.4
                day_night_multiplier = night_reduction + (1.0 - night_reduction) * transition_factor
            
            # Apply day-night effect to evaporation
            evaporation_rate = evaporation_base_rate * temp_factor * day_night_multiplier
        else:
            # If day-night cycle data isn't available, proceed without adjustment
            evaporation_rate = evaporation_base_rate * temp_factor
        
        # --- PRECIPITATION ---
        # Calculate relative humidity
        vapor_pressure = self.humidity * saturation_vapor_pressure
        relative_humidity = vapor_pressure / saturation_vapor_pressure
        self._relative_humidity = relative_humidity  # Store for statistical analysis
        
        # Calculate precipitation rate based on RH
        precipitation_threshold = 0.7
        new_precipitation_rate = np.zeros_like(relative_humidity, dtype=np.float32)
        precip_mask = relative_humidity > precipitation_threshold
        
        if np.any(precip_mask):
            # Precipitation rate increases with excess RH
            excess_humidity = relative_humidity[precip_mask] - precipitation_threshold
            new_precipitation_rate[precip_mask] = excess_humidity**2 * 5.0  # Reduced from 15.0
        
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
        
        # --- UPDATE HUMIDITY ---
        # Use actual precipitation for humidity changes (not smoothed precipitation)
        total_factor = time_step_seconds
        # MODIFIED: Add a stronger precipitation reduction coefficient to fix oscillation issue
        precipitation_humidity_reduction_factor = 2.0  # Increase this to remove more humidity when it rains
        humidity_change = (evaporation_rate - new_precipitation_rate * precipitation_humidity_reduction_factor) * total_factor
        self.humidity += humidity_change
        
        # Apply diffusion and constraints
        self.humidity = gaussian_filter(self.humidity, sigma=1.0)
        np.clip(self.humidity, 0.01, 1.0, out=self.humidity)  # In-place clipping
        
        # --- UPDATE STATISTICAL MODELS ---
        if self._should_update_statistics:
            self._update_statistical_relations(T, relative_humidity, evaporation_rate, new_precipitation_rate)
    
    def _update_statistical(self):
        """Update precipitation using statistical patterns for better performance"""
        try:
            # Get current conditions
            T = self.sim.temperature_celsius
            is_ocean = self.sim.elevation <= 0
            is_land = ~is_ocean
            time_step_seconds = getattr(self.sim, 'time_step_seconds', 3600.0)
            
            # --- 1. STATISTICAL EVAPORATION ---
            # Use simple temperature-based relation
            if self._temp_precip_relation is None:
                # Initialize with default values if not available
                self._initialize_statistical_relations()
                
            # Calculate evaporation based on temperature
            evaporation_base = np.zeros_like(T, dtype=np.float32)
            
            # Different rates for land and ocean
            evaporation_base[is_land] = 0.015  # Increased from 0.005 (3x)
            evaporation_base[is_ocean] = 0.06  # Increased from 0.02 (3x)
            
            # Apply temperature relation
            temp_bins = self._temp_precip_relation['temp_bins']
            evap_factors = self._temp_precip_relation['evap_factors']
            
            # Find which temperature bin each grid cell belongs to
            temp_indices = np.zeros_like(T, dtype=np.int32)
            for i in range(len(temp_bins)-1):
                temp_indices = np.where((T >= temp_bins[i]) & (T < temp_bins[i+1]), i, temp_indices)
            
            # Apply the corresponding evaporation factor
            evap_map = np.zeros_like(T, dtype=np.float32)
            for i in range(len(evap_factors)):
                evap_map = np.where(temp_indices == i, evap_factors[i], evap_map)
                
            # Calculate final evaporation rate
            evaporation_rate = evaporation_base * evap_map
            
            # ADDED: Day-night cycle effect on evaporation
            # Apply the same day-night cycle effect as in the full physics version
            day_night_factor = 0.6  # Default if not available
            if hasattr(self.sim, 'latitudes_rad'):
                # Calculate solar angle effect (vectorized)
                cos_phi = np.cos(self.sim.latitudes_rad).astype(np.float32)
                day_night_factor = np.clip(cos_phi, 0, 1)  # Day/night cycle
                
                # Calculate average day-night factor across the map
                avg_day_factor = np.mean(day_night_factor)
                
                # Same reductions as in the full physics version
                night_reduction = 0.6  # Night evaporation is 60% of daytime
                if avg_day_factor < 0.3:  # Night
                    day_night_multiplier = night_reduction + (1.0 - night_reduction) * (avg_day_factor / 0.3)
                elif avg_day_factor > 0.7:  # Day
                    day_night_multiplier = 0.9 + 0.1 * ((avg_day_factor - 0.7) / 0.3)
                else:  # Dawn/dusk
                    transition_factor = (avg_day_factor - 0.3) / 0.4
                    day_night_multiplier = night_reduction + (1.0 - night_reduction) * transition_factor
                
                # Apply day-night effect to evaporation
                evaporation_rate = evaporation_rate * day_night_multiplier
            
            # --- 2. STATISTICAL PRECIPITATION ---
            # Use humidity-precipitation relation
            if self._humidity_precip_relation is None:
                # Initialize with default values if not available
                self._initialize_statistical_relations()
                
            # Calculate precipitation based on humidity
            precip_rate = np.zeros_like(self.humidity, dtype=np.float32)
            
            # Apply humidity relation
            humidity_bins = self._humidity_precip_relation['humidity_bins']
            precip_factors = self._humidity_precip_relation['precip_factors']
            
            # Find which humidity bin each grid cell belongs to
            humidity_indices = np.zeros_like(self.humidity, dtype=np.int32)
            for i in range(len(humidity_bins)-1):
                humidity_indices = np.where(
                    (self.humidity >= humidity_bins[i]) & (self.humidity < humidity_bins[i+1]), 
                    i, humidity_indices
                )
            
            # Apply the corresponding precipitation factor
            for i in range(len(precip_factors)):
                precip_rate = np.where(humidity_indices == i, precip_factors[i], precip_rate)
                
            # --- 3. APPLY OROGRAPHIC EFFECTS ---
            # Simplified orographic effect based on precomputed patterns
            if self._wind_precip_relation is None or self._slope_x is None:
                # Calculate slopes if needed
                if self._slope_x is None:
                    dy, dx = np.gradient(self.sim.elevation)
                    self._slope_x = dx / self.sim.grid_spacing_x
                    self._slope_y = dy / self.sim.grid_spacing_y
                
                # Initialize wind relation
                self._initialize_statistical_relations()
            
            # Simple parametric approach for orographic effects
            if hasattr(self.sim, 'u') and hasattr(self.sim, 'v'):
                # Calculate upslope wind component
                wind_upslope = -(self.sim.u * self._slope_x + self.sim.v * self._slope_y)
                
                # Apply enhancement where wind flows uphill
                orographic_mask = is_land & (wind_upslope > 0)
                if np.any(orographic_mask):
                    # Use precomputed relation between upslope and enhancement
                    upslope_bins = self._wind_precip_relation['slope_bins']
                    enhancement_factors = self._wind_precip_relation['wind_factors']
                    
                    # Apply appropriate enhancement for each upslope value
                    for i in range(len(upslope_bins)-1):
                        mask = orographic_mask & (wind_upslope >= upslope_bins[i]) & (wind_upslope < upslope_bins[i+1])
                        if np.any(mask):
                            precip_rate[mask] += enhancement_factors[i]
            
            # --- 4. TEMPORAL SMOOTHING ---
            # Apply temporal persistence (rain doesn't stop immediately)
            precipitation_persistence = 0.85
            
            # Apply spatial smoothing to precipitation rate
            smoothed_precip = gaussian_filter(precip_rate, sigma=1.5)
            
            # Combine with previous precipitation
            new_precipitation = (precipitation_persistence * self._prev_precipitation + 
                                (1 - precipitation_persistence) * smoothed_precip)
            
            # Additional smoothing for consistent patterns
            self.precipitation = gaussian_filter(new_precipitation, sigma=1.0)
            
            # Store for next update
            self._prev_precipitation = self.precipitation.copy()
            
            # --- 5. UPDATE HUMIDITY ---
            # Calculate humidity change
            # MODIFIED: Apply same precipitation humidity reduction factor
            precipitation_humidity_reduction_factor = 2.0  # Match the factor in _update_full_physics
            humidity_change = (evaporation_rate - precip_rate * precipitation_humidity_reduction_factor) * time_step_seconds
            self.humidity += humidity_change
            
            # Apply diffusion and constraints
            self.humidity = gaussian_filter(self.humidity, sigma=1.0)
            np.clip(self.humidity, 0.01, 1.0, out=self.humidity)
            
            # --- STORE FOR DIAGNOSTICS ---
            self._last_evaporation_rate = np.mean(evaporation_rate)
            
        except Exception as e:
            print(f"Error in statistical precipitation update: {e}")
            traceback.print_exc()
            # Fall back to full physics if statistical model fails
            self._force_full_physics = True
            self._update_full_physics()
            
    def _initialize_statistical_relations(self):
        """Initialize statistical relationships based on Earth-like climate patterns"""
        # 1. Temperature-evaporation relation
        self._temp_precip_relation = {
            'temp_bins': [-50, -20, -10, 0, 10, 20, 30, 50],  # °C
            'evap_factors': [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]  # Multipliers
        }
        
        # 2. Humidity-precipitation relation
        self._humidity_precip_relation = {
            'humidity_bins': [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # Humidity levels
            'precip_factors': [0, 0.01, 0.05, 0.2, 1.0, 5.0]  # mm/hour precipitation
        }
        
        # 3. Wind-precipitation relation (for orographic effects)
        self._wind_precip_relation = {
            'slope_bins': [-0.05, -0.02, -0.01, 0, 0.01, 0.02, 0.05],  # Terrain slope
            'wind_factors': [0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0]  # Precipitation multipliers
        }
        
        # 4. Regional precipitation statistics with default values
        self._precip_statistics = {
            'land_avg': 0.01,
            'ocean_avg': 0.02,
            'tropical_avg': 0.03,
            'midlat_avg': 0.02,
            'polar_avg': 0.01
        }
        
    def _update_statistical_relations(self, T, relative_humidity, evaporation_rate, precipitation_rate):
        """Update statistical relationships based on full physics calculation"""
        try:
            # 1. Update temperature-evaporation relation
            # Group evaporation rates by temperature bins
            temp_bins = np.linspace(-50, 50, 8)
            evap_factors = []
            
            for i in range(len(temp_bins)-1):
                bin_mask = (T >= temp_bins[i]) & (T < temp_bins[i+1])
                if np.any(bin_mask):
                    avg_evap = np.mean(evaporation_rate[bin_mask])
                    base_evap = 0.01  # Reference evaporation rate
                    evap_factors.append(float(avg_evap / base_evap))
                else:
                    # Use default value if no data points in this bin
                    evap_factors.append(0.5 + i * 0.25)
                    
            # Update relation
            self._temp_precip_relation = {
                'temp_bins': temp_bins.tolist(),
                'evap_factors': evap_factors
            }
            
            # 2. Update humidity-precipitation relation
            humidity_bins = np.linspace(0, 1.0, 7)
            precip_factors = []
            
            for i in range(len(humidity_bins)-1):
                bin_mask = (relative_humidity >= humidity_bins[i]) & (relative_humidity < humidity_bins[i+1])
                if np.any(bin_mask):
                    avg_precip = np.mean(precipitation_rate[bin_mask])
                    precip_factors.append(float(avg_precip))
                else:
                    # Use default values that increase exponentially with humidity
                    if i < 3:
                        precip_factors.append(0.01 * (i+1))
                    else:
                        precip_factors.append(0.05 * (2 ** (i-2)))
            
            # Update relation
            self._humidity_precip_relation = {
                'humidity_bins': humidity_bins.tolist(),
                'precip_factors': precip_factors
            }
            
            # 3. Update regional precipitation statistics
            is_ocean = self.sim.elevation <= 0
            is_land = ~is_ocean
            
            # Ensure precip_statistics exists
            if self._precip_statistics is None:
                self._precip_statistics = {}
            
            # Get latitude bands
            lat = np.abs(self.sim.latitude)
            tropical_mask = lat < 23.5
            polar_mask = lat > 66.5
            midlat_mask = ~tropical_mask & ~polar_mask
            
            # Calculate average precipitation by region
            if np.any(is_land):
                self._precip_statistics['land_avg'] = float(np.mean(precipitation_rate[is_land]))
            if np.any(is_ocean):
                self._precip_statistics['ocean_avg'] = float(np.mean(precipitation_rate[is_ocean]))
            if np.any(tropical_mask):
                self._precip_statistics['tropical_avg'] = float(np.mean(precipitation_rate[tropical_mask]))
            if np.any(midlat_mask):
                self._precip_statistics['midlat_avg'] = float(np.mean(precipitation_rate[midlat_mask]))
            if np.any(polar_mask):
                self._precip_statistics['polar_avg'] = float(np.mean(precipitation_rate[polar_mask]))
                
        except Exception as e:
            print(f"Error updating statistical relations: {e}")
            traceback.print_exc()
            
    def _get_mean_evaporation_rate(self):
        """Get the mean evaporation rate for diagnostics"""
        if hasattr(self, '_last_evaporation_rate'):
            # Add a multiplier to match the precipitation rate more closely
            return self._last_evaporation_rate * 15.0  # Apply significant multiplier to balance with precipitation
        else:
            # Provide a more realistic default
            return 0.15  # Increased from 0.01
    
    def _update_adaptive(self):
        """Update precipitation and humidity using lower resolution for performance"""
        try:
            factor = self._downsampling_factor
            time_step_seconds = getattr(self.sim, 'time_step_seconds', 3600.0)
            
            # Downsample key input fields
            h, w = self.sim.map_height, self.sim.map_width
            h_low, w_low = h // factor, w // factor
            
            # Downsample temperature and elevation (less frequently to save performance)
            if (not hasattr(self, '_downsampled_fields') or 
                self.sim.time_step % 20 == 0):
                
                # Downsample elevation for orographic calculations
                elevation_low = zoom(self.sim.elevation, 1/factor, order=1)
                is_ocean_low = elevation_low <= 0
                is_land_low = ~is_ocean_low
                
                # Calculate terrain slope for downsampled elevation
                if elevation_low.size > 1:
                    dy_low, dx_low = np.gradient(elevation_low)
                    slope_x_low = dx_low
                    slope_y_low = dy_low
                else:
                    slope_x_low = np.zeros_like(elevation_low)
                    slope_y_low = np.zeros_like(elevation_low)
                
                self._downsampled_fields = {
                    'elevation': elevation_low,
                    'is_ocean': is_ocean_low,
                    'is_land': is_land_low,
                    'slope_x': slope_x_low,
                    'slope_y': slope_y_low,
                    'shape': (h_low, w_low)
                }
            
            # Always downsample dynamic fields
            ds = self._downsampled_fields
            
            # Downsample humidity and temperature
            if self.humidity is None:
                # Initialize if this is the first call
                humidity_low = np.full(ds['shape'], 0.7)  # Default value
            else:
                humidity_low = zoom(self.humidity, 1/factor, order=1)
            
            # Get low-res temperature
            temp_low = zoom(self.sim.temperature_celsius, 1/factor, order=1)
                
            # Wind fields
            u_low = zoom(self.sim.u, 1/factor, order=1)
            v_low = zoom(self.sim.v, 1/factor, order=1)
            
            # --- STATISTICAL PATTERNS AT LOW RESOLUTION ---
            
            # 1. Temperature-based evaporation (statistical approach)
            if self._temp_precip_relation is None:
                self._initialize_statistical_relations()
                
            # Base evaporation rates by surface type
            evaporation_base = np.full_like(humidity_low, 0.015)  # Increased from 0.005 (3x)
            evaporation_base[ds['is_ocean']] = 0.06  # Increased from 0.02 (3x)
            
            # Apply temperature-based factors using statistical relations
            temp_bins = self._temp_precip_relation['temp_bins']
            evap_factors = self._temp_precip_relation['evap_factors']
            
            # Simplified bin lookup for low-res grid
            evap_factor_map = np.ones_like(temp_low)
            for i in range(len(temp_bins)-1):
                bin_mask = (temp_low >= temp_bins[i]) & (temp_low < temp_bins[i+1])
                if i < len(evap_factors) and np.any(bin_mask):
                    evap_factor_map[bin_mask] = evap_factors[i]
            
            # Calculate final evaporation rate
            evaporation_rate = evaporation_base * evap_factor_map
            
            # 2. Humidity-based precipitation (statistical approach)
            if self._humidity_precip_relation is None:
                self._initialize_statistical_relations()
                
            # Apply humidity-precipitation relation
            humidity_bins = self._humidity_precip_relation['humidity_bins']
            precip_factors = self._humidity_precip_relation['precip_factors']
            
            # Simplified bin lookup for precipitation
            precip_rate = np.zeros_like(humidity_low)
            for i in range(len(humidity_bins)-1):
                bin_mask = (humidity_low >= humidity_bins[i]) & (humidity_low < humidity_bins[i+1])
                if i < len(precip_factors) and np.any(bin_mask):
                    precip_rate[bin_mask] = precip_factors[i]
            
            # 3. Add orographic enhancement using wind relation
            if self._wind_precip_relation is None:
                self._initialize_statistical_relations()
                
            # Simplified orographic effect
            if np.any(ds['is_land']):
                wind_upslope = -(u_low * ds['slope_x'] + v_low * ds['slope_y'])
                upslope_bins = self._wind_precip_relation['slope_bins']
                enhancement_factors = self._wind_precip_relation['wind_factors']
                
                # Apply enhancement where wind flows uphill
                for i in range(len(upslope_bins)-1):
                    bin_mask = ds['is_land'] & (wind_upslope > 0) & (wind_upslope >= upslope_bins[i]) & (wind_upslope < upslope_bins[i+1])
                    if i < len(enhancement_factors) and np.any(bin_mask):
                        precip_rate[bin_mask] += enhancement_factors[i]
            
            # 4. Apply temporal smoothing for persistence
            if not hasattr(self, '_prev_precipitation_low') or self._prev_precipitation_low is None:
                self._prev_precipitation_low = precip_rate.copy()
                
            precipitation_persistence = 0.85
            smoothed_precip = gaussian_filter(precip_rate, sigma=1.0)
            
            new_precipitation_low = (precipitation_persistence * self._prev_precipitation_low + 
                                     (1 - precipitation_persistence) * smoothed_precip)
            
            # Apply additional smoothing for stability
            new_precipitation_low = gaussian_filter(new_precipitation_low, sigma=1.0)
            
            # Store for next update
            self._prev_precipitation_low = new_precipitation_low.copy()
            
            # 5. Update humidity
            humidity_change = (evaporation_rate - precip_rate) * time_step_seconds
            humidity_low += humidity_change
            
            # Apply diffusion and constraints
            humidity_low = gaussian_filter(humidity_low, sigma=0.5)
            np.clip(humidity_low, 0.01, 1.0, out=humidity_low)
            
            # 6. Upsample results to full resolution
            self.humidity = zoom(humidity_low, factor, order=1)
            self.precipitation = zoom(new_precipitation_low, factor, order=1)
            
            # Fix any dimension issues due to rounding
            if self.humidity.shape != (h, w):
                self.humidity = np.resize(self.humidity, (h, w))
            if self.precipitation.shape != (h, w):
                self.precipitation = np.resize(self.precipitation, (h, w))
            
            # Update diagnostics
            self.sim.energy_budget['evaporation'] = float(np.mean(evaporation_rate))
            self.sim.energy_budget['precipitation'] = float(np.mean(new_precipitation_low))
            self._last_evaporation_rate = float(np.mean(evaporation_rate))
            
            # 7. Update clouds
            self.update_clouds()
            
        except Exception as e:
            print(f"Error in adaptive precipitation update: {e}")
            traceback.print_exc()
            # Fall back to statistical method at full resolution
            self._update_statistical()

    def update_clouds(self):
        """Calculate cloud cover based on humidity and temperature"""
        try:
            # Basic cloud formation based on humidity
            if self.humidity is None:
                self.cloud_cover = np.zeros_like(self.sim.temperature_celsius)
                return
            
            # Store previous cloud cover for temporal smoothing
            if not hasattr(self, '_prev_cloud_cover') or self._prev_cloud_cover is None:
                self._prev_cloud_cover = np.copy(self.cloud_cover) if self.cloud_cover is not None else np.zeros_like(self.sim.temperature_celsius)
            
            # Scale humidity 0.5-1.0 to cloud cover 0-1
            target_cloud_cover = np.clip(self.humidity - 0.5, 0, 1) * 2
            
            # Apply temporal smoothing to reduce oscillations
            cloud_persistence = 0.85  # Increased from 0.7 to 0.85 for stronger persistence
            self.cloud_cover = cloud_persistence * self._prev_cloud_cover + (1 - cloud_persistence) * target_cloud_cover
            
            # Store current cloud cover for next update
            self._prev_cloud_cover = np.copy(self.cloud_cover)
            
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