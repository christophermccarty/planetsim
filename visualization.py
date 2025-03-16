import numpy as np
import time
import traceback
import tkinter as tk
from PIL import Image, ImageTk
from scipy.ndimage import gaussian_filter
from map_generation import MapGenerator
import threading

class Visualization:
    """Class responsible for visualization of simulation data"""
    
    def __init__(self, simulation):
        """Initialize visualization system with reference to main simulation"""
        self.sim = simulation
        self.root = simulation.root
        self.canvas = simulation.canvas
        
        # Cache for PhotoImage objects to prevent garbage collection
        self.image_cache = {}
        
        # Tracking variables
        self._last_update_time = time.time()
        self._last_selected_layer = None
        
        # Cache for pressure and precipitation images
        self._pressure_image = None
        self._cached_precip_image = None
        
    def map_to_grayscale(self, data):
        """Convert data to grayscale values"""
        grayscale = np.clip(data, 0, 1)
        grayscale = (grayscale * 255).astype(np.uint8)
        rgb_array = np.stack((grayscale,)*3, axis=-1)
        return rgb_array
        
    def map_altitude_to_color(self, elevation):
        """Map elevation data to RGB colors using standard elevation palette"""
        # Create RGB array
        rgb = np.zeros((self.sim.map_height, self.sim.map_width, 3), dtype=np.uint8)
        
        # Get the actual grayscale range from the image
        min_gray = self.sim.altitude.min()  # Should be 0
        max_gray = self.sim.altitude.max()  # Should be 510
        sea_level = 3  # First land pixel value
        
        # Create masks based on raw grayscale values
        ocean = self.sim.altitude <= 2
        land = self.sim.altitude > 2
        
        # Ocean (blue)
        rgb[ocean] = [0, 0, 255]
        
        # Land areas
        if np.any(land):
            land_colors = np.array([
                [4, 97, 2],      # Dark Green (lowest elevation)
                [43, 128, 42],   # Light Green
                [94, 168, 93],   # Light yellow
                [235, 227, 87],  # Yellow
                [171, 143, 67],  # Brown
                [107, 80, 0],    # Dark Brown
                [227, 227, 227]  # Gray (highest elevation)
            ])
            
            # Get grayscale values for land only
            land_grayscale = self.sim.altitude[land]
            
            # Normalize land heights to [0,1] range
            normalized_heights = (land_grayscale - sea_level) / (max_gray - sea_level)
            
            # Calculate indices for color interpolation
            color_indices = (normalized_heights * (len(land_colors) - 1)).astype(int)
            color_indices = np.clip(color_indices, 0, len(land_colors) - 2)
            
            # Calculate interpolation ratios
            ratios = (normalized_heights * (len(land_colors) - 1)) % 1
            
            # Interpolate colors
            start_colors = land_colors[color_indices]
            end_colors = land_colors[color_indices + 1]
            
            # Apply interpolated colors to land areas
            rgb[land] = (start_colors * (1 - ratios[:, np.newaxis]) + 
                        end_colors * ratios[:, np.newaxis]).astype(np.uint8)
                 
        return rgb
        
    def map_temperature_to_color(self, temperature_data):
        """
        Map temperature to a blue-red color gradient with more distinct transitions.
        Blue indicates cold, red indicates hot.
        """
        # Create RGBA array
        rgba_array = np.zeros((self.sim.map_height, self.sim.map_width, 4), dtype=np.uint8)
        
        # Set fixed temperature range for consistent coloring
        min_temp = -50.0  # Minimum expected temperature
        max_temp = 50.0   # Maximum expected temperature
        
        # Normalize temperature to [0,1] using fixed range
        temp_normalized = np.clip((temperature_data - min_temp) / (max_temp - min_temp), 0, 1)
        
        # Create more distinct temperature gradient
        # Red component (increases with temperature)
        rgba_array[..., 0] = (np.power(temp_normalized, 1.2) * 255).astype(np.uint8)
        
        # Green component (peaks in middle temperatures)
        green = np.minimum(2 * temp_normalized, 2 * (1 - temp_normalized))
        rgba_array[..., 1] = (green * 180).astype(np.uint8)  # Reduced max green for more vibrant colors
        
        # Blue component (decreases with temperature)
        rgba_array[..., 2] = (np.power(1 - temp_normalized, 1.2) * 255).astype(np.uint8)
        
        # Set alpha channel
        rgba_array[..., 3] = 255  # Full opacity
        
        # Make water areas slightly transparent
        is_water = self.sim.elevation <= 0
        rgba_array[is_water, 3] = 192  # 75% opacity for water
        
        return rgba_array
    
    def map_ocean_temperature_to_color(self, data_normalized):
        """Map ocean temperature data to RGB colors"""
        # Create output array for RGB data
        rgb = np.zeros((self.sim.map_height, self.sim.map_width, 3), dtype=np.uint8)
        
        # Create land-ocean mask
        is_land = self.sim.elevation > 0
        is_ocean = ~is_land
        
        # Set land to a neutral gray
        rgb[is_land] = [220, 220, 220]  # Light gray for land
        
        # Define ocean temperature breakpoints
        breakpoints = np.array([-5, 0, 5, 10, 15, 20, 25, 30])
        
        # Define colors for each breakpoint (from cold blue to warm blue-green)
        colors = np.array([
            [180, 180, 255],  # Very light blue (coldest ocean)
            [100, 100, 255],  # Light blue
            [30, 30, 255],    # Blue
            [0, 50, 200],     # Medium blue
            [0, 100, 180],    # Blue-green
            [0, 150, 160],    # Teal
            [0, 180, 140],    # Green-blue
            [0, 200, 120]     # Green-teal (warmest ocean)
        ])
        
        # Apply to ocean areas only
        ocean_temps = self.sim.temperature_celsius.copy()
        
        # For each temperature value in ocean, find where it fits in the breakpoints
        for i in range(len(breakpoints) - 1):
            lower_temp = breakpoints[i]
            upper_temp = breakpoints[i+1]
            lower_color = colors[i]
            upper_color = colors[i+1]
            
            # Create a mask for temperature values within this range
            mask = is_ocean & (ocean_temps >= lower_temp) & (ocean_temps < upper_temp)
            
            if np.any(mask):
                # Calculate how far between breakpoints (0 to 1)
                t = (ocean_temps[mask] - lower_temp) / (upper_temp - lower_temp)
                
                # Interpolate RGB values
                r = lower_color[0] + t * (upper_color[0] - lower_color[0])
                g = lower_color[1] + t * (upper_color[1] - lower_color[1])
                b = lower_color[2] + t * (upper_color[2] - lower_color[2])
                
                # Assign the interpolated RGB values
                rgb[mask, 0] = np.clip(r, 0, 255).astype(np.uint8)
                rgb[mask, 1] = np.clip(g, 0, 255).astype(np.uint8)
                rgb[mask, 2] = np.clip(b, 0, 255).astype(np.uint8)
        
        # Handle values below the lowest breakpoint
        mask_below = is_ocean & (ocean_temps < breakpoints[0])
        if np.any(mask_below):
            rgb[mask_below] = colors[0]
        
        # Handle values above the highest breakpoint
        mask_above = is_ocean & (ocean_temps >= breakpoints[-1])
        if np.any(mask_above):
            rgb[mask_above] = colors[-1]
            
        return rgb
    
    def map_pressure_to_color(self, pressure_data):
        """Convert pressure data to color visualization with less frequent isolines"""
        try:
            # Reuse cached pressure visualization if pressure data hasn't changed significantly
            if hasattr(self, '_cached_pressure_viz') and hasattr(self, '_cached_pressure_data'):
                # Only recalculate if pressure has changed significantly
                if np.array_equal(pressure_data, self._cached_pressure_data):
                    return self._cached_pressure_viz
                
                # Check if changes are minor - use a smaller threshold to detect more changes
                # Reduced threshold further to ensure more frequent updates
                elif np.abs(np.mean(pressure_data - self._cached_pressure_data)) < 0.0001 * np.mean(np.abs(self._cached_pressure_data)):
                    if hasattr(self, '_pressure_viz_skip_counter'):
                        self._pressure_viz_skip_counter += 1
                        # Only skip every other frame at most
                        if self._pressure_viz_skip_counter < 1:
                            return self._cached_pressure_viz
                        else:
                            self._pressure_viz_skip_counter = 0
                    else:
                        self._pressure_viz_skip_counter = 0
            
            # Get terrain colors first (cache if possible)
            if not hasattr(self, '_cached_terrain_colors'):
                self._cached_terrain_colors = self.map_elevation_to_color(self.sim.elevation)
            terrain_colors = self._cached_terrain_colors
            
            # Convert pressure to hPa once
            pressure_hpa = pressure_data / 100.0
            
            # Set fixed pressure range (870-1086 hPa)
            p_min, p_max = 870, 1086
            
            # Pre-allocate the RGBA array
            rgba_array = np.zeros((*terrain_colors.shape[:2], 4), dtype=np.uint8)
            
            # Normalize pressure in one step
            normalized_pressure = 2 * (pressure_hpa - (p_min + p_max)/2) / (p_max - p_min)
            
            # Vectorized color calculations (all at once instead of channel by channel)
            pressure_colors = np.zeros((*terrain_colors.shape[:2], 4), dtype=np.uint8)
            
            # Create color lookup tables
            pressure_ranges = np.array([-1, -0.5, 0, 0.5, 1])
            r_values = np.array([65, 150, 255, 255, 255])
            g_values = np.array([200, 230, 255, 230, 200])
            b_values = np.array([255, 255, 255, 150, 65])
            
            # Vectorized interpolation for all channels at once
            pressure_colors[..., 0] = np.interp(normalized_pressure, pressure_ranges, r_values)
            pressure_colors[..., 1] = np.interp(normalized_pressure, pressure_ranges, g_values)
            pressure_colors[..., 2] = np.interp(normalized_pressure, pressure_ranges, b_values)
            pressure_colors[..., 3] = 170  # Constant opacity
            
            # Efficient alpha blending
            alpha = pressure_colors[..., 3:] / 255.0
            rgba_array[..., :3] = ((pressure_colors[..., :3] * alpha) + 
                                 (terrain_colors[..., :3] * (1 - alpha))).astype(np.uint8)
            rgba_array[..., 3] = 255
            
            # Always draw isobars - removed the skipping logic
            # Optimize isobar calculation
            if not hasattr(self, '_pressure_levels'):
                # Calculate pressure levels once and cache
                min_level = np.floor(p_min / 25) * 25
                max_level = np.ceil(p_max / 25) * 25
                self._pressure_levels = np.arange(min_level, max_level + 25, 25)
            
            # Reduce computation by using cached smoothed pressure when possible
            if not hasattr(self, '_last_smoothed_pressure_data') or not np.array_equal(pressure_data, self._last_smoothed_pressure_data):
                # Use pre-smoothed pressure field for isobars
                smoothed_pressure = gaussian_filter(pressure_hpa, sigma=1.0)
                self._last_smoothed_pressure = smoothed_pressure
                self._last_smoothed_pressure_data = pressure_data.copy()
            else:
                smoothed_pressure = self._last_smoothed_pressure
            
            # Reduce number of isobar levels for better performance - use fewer pressure levels
            reduced_pressure_levels = self._pressure_levels[::2]  # Only use every other level
            
            # Vectorized isobar calculation
            isobar_mask = np.zeros_like(pressure_hpa, dtype=bool)
            for level in reduced_pressure_levels:
                isobar_mask |= np.abs(smoothed_pressure - level) < 0.5
            
            # Apply smoothing to mask more efficiently
            isobar_mask = gaussian_filter(isobar_mask.astype(float), sigma=0.5) > 0.3
            
            # Apply isobars efficiently
            rgba_array[isobar_mask, :3] = 255  # White lines
            rgba_array[isobar_mask, 3] = 180  # Slightly transparent
            
            # Cache the result
            self._cached_pressure_viz = rgba_array
            self._cached_pressure_data = pressure_data.copy()
            
            return rgba_array
                
        except Exception as e:
            print(f"Error in map_pressure_to_color: {e}")
            if hasattr(self, '_cached_pressure_viz'):
                return self._cached_pressure_viz
            return np.zeros((self.sim.map_height, self.sim.map_width, 4), dtype=np.uint8)
    
    def map_precipitation_to_color(self, precipitation_data):
        """Map precipitation to color"""
        try:
            # Performance optimization: Check if precipitation is all zeros
            if np.all(precipitation_data < 0.0001):
                # No precipitation, just return terrain
                return self.map_elevation_to_color(self.sim.elevation)
                
            # Get base map colors first (use terrain for context)
            terrain_colors = self.map_elevation_to_color(self.sim.elevation)
            
            # Pre-allocate the RGBA array
            rgba_array = np.zeros((*terrain_colors.shape[:2], 4), dtype=np.uint8)
            
            # Copy terrain as base
            rgba_array[..., :3] = terrain_colors[..., :3]
            rgba_array[..., 3] = 255
            
            # Create masks for different precipitation levels
            # Use less expensive operations and avoid unnecessary computations
            light_precip_mask = precipitation_data > 0.0005  # Very light precipitation
            
            # If no precipitation above threshold, just return terrain
            if not np.any(light_precip_mask):
                return rgba_array
                
            precip_mask = precipitation_data > 0.001  # Regular precipitation
            heavy_precip_mask = precipitation_data > 0.05  # Heavy precipitation
            
            # Normalize precipitation for coloring (logarithmic scale for better visualization)
            # Cap at reasonable rainfall values (0-50 mm/hr)
            capped_precip = np.clip(precipitation_data, 0, 50)
            
            # Use log scale for better visualization (log(x+1) to handle zeros)
            # More efficient scaling to avoid excessive computation
            log_precip = np.log1p(capped_precip * 5) / np.log1p(250)  # Simplified scaling
            
            # Create blue-scale for rain intensity - optimize memory usage
            rain_colors = np.zeros((*precipitation_data.shape, 4), dtype=np.uint8)
            
            # More gradual color transitions for very light rain
            # Optimize by only computing where needed
            if np.any(light_precip_mask):
                rain_colors[light_precip_mask, 0] = 55 + 20 * log_precip[light_precip_mask]  # Slight red
                rain_colors[light_precip_mask, 1] = 125 + 20 * log_precip[light_precip_mask]  # Some green
                rain_colors[light_precip_mask, 2] = 180 + 20 * log_precip[light_precip_mask]  # Blue component
                rain_colors[light_precip_mask, 3] = 30 + 40 * log_precip[light_precip_mask]  # Low opacity
            
            # Regular precipitation (more visible)
            if np.any(precip_mask):
                rain_colors[precip_mask, 0] = 55 + 100 * log_precip[precip_mask]  # Slight red
                rain_colors[precip_mask, 1] = 125 + 100 * log_precip[precip_mask]  # Some green
                rain_colors[precip_mask, 2] = 200 + 55 * log_precip[precip_mask]  # Strong blue
                rain_colors[precip_mask, 3] = 50 + 150 * log_precip[precip_mask]  # More opacity
            
            # Heavy precipitation (most intense coloring) - only compute if needed
            if np.any(heavy_precip_mask):
                rain_colors[heavy_precip_mask, 0] = 90 + 50 * log_precip[heavy_precip_mask]
                rain_colors[heavy_precip_mask, 1] = 160 + 60 * log_precip[heavy_precip_mask]
                rain_colors[heavy_precip_mask, 2] = 240
                rain_colors[heavy_precip_mask, 3] = 150 + 105 * log_precip[heavy_precip_mask]
            
            # Blend rain with terrain
            alpha = rain_colors[..., 3:] / 255.0
            rgba_array[..., :3] = ((rain_colors[..., :3] * alpha) + 
                                (terrain_colors[..., :3] * (1 - alpha))).astype(np.uint8)
            
            # Special effects for heavy rain - only add if we have heavy rain
            # and limit the number of raindrops for performance
            if np.any(precipitation_data > 0.1):  # Only for significant rainfall
                heavy_rain = log_precip > 0.6
                if np.sum(heavy_rain) > 0:
                    # Limit the number of droplets for performance (max 200 droplets)
                    max_droplets = min(200, int(np.sum(heavy_rain) * 0.03))
                    if max_droplets > 0:
                        # Get coordinates of heavy rain areas
                        heavy_y, heavy_x = np.where(heavy_rain)
                        if len(heavy_y) > 0:
                            # Randomly select a subset of coordinates
                            selected_indices = np.random.choice(len(heavy_y), 
                                                              size=min(max_droplets, len(heavy_y)), 
                                                              replace=False)
                            # Apply droplets only at selected points
                            for idx in selected_indices:
                                y, x = heavy_y[idx], heavy_x[idx]
                                rgba_array[y, x, 0] = 180
                                rgba_array[y, x, 1] = 220
                                rgba_array[y, x, 2] = 255
            
            return rgba_array
            
        except Exception as e:
            print(f"Error in map_precipitation_to_color: {e}")
            traceback.print_exc()
            # Return a safe fallback
            return self.map_elevation_to_color(self.sim.elevation)
    
    def map_clouds_to_color(self):
        """Map cloud cover data to RGBA colors for overlay"""
        # Create output array for RGBA data (RGB + Alpha)
        rgba = np.zeros((self.sim.map_height, self.sim.map_width, 4), dtype=np.uint8)
        
        # No cloud cover data available
        if not hasattr(self.sim, 'cloud_cover') or self.sim.cloud_cover is None:
            return rgba
        
        # Set base RGB to white
        rgba[:, :, 0:3] = 255
        
        # Set alpha channel based on cloud cover (0 = fully transparent, 255 = fully opaque)
        alpha = (self.sim.cloud_cover * 255).astype(np.uint8)
        
        # Apply minimum visibility threshold for clouds
        cloud_threshold = 0.1  # Minimum cloud coverage to be visible
        alpha[self.sim.cloud_cover < cloud_threshold] = 0
        
        # Store alpha in the 4th channel
        rgba[:, :, 3] = alpha
        
        return rgba
    
    def map_elevation_to_color(self, elevation_data):
        """Map elevation to color with improved coloring"""
        # Create RGB array
        rgb = np.zeros((self.sim.map_height, self.sim.map_width, 3), dtype=np.uint8)
        
        # Normalize elevation data
        normalized_elevation = (elevation_data - np.min(elevation_data)) / (np.max(elevation_data) - np.min(elevation_data))
        
        # Create terrain colors
        # Below sea level (blues)
        ocean_mask = elevation_data <= 0
        rgb[ocean_mask, 0] = (50 * normalized_elevation[ocean_mask]).astype(np.uint8)  # R
        rgb[ocean_mask, 1] = (100 * normalized_elevation[ocean_mask]).astype(np.uint8)  # G
        rgb[ocean_mask, 2] = (150 + 105 * normalized_elevation[ocean_mask]).astype(np.uint8)  # B
        
        # Above sea level (greens and browns)
        land_mask = elevation_data > 0
        normalized_land = normalized_elevation[land_mask]
        rgb[land_mask, 0] = (100 + 155 * normalized_land).astype(np.uint8)  # R
        rgb[land_mask, 1] = (100 + 100 * normalized_land).astype(np.uint8)  # G
        rgb[land_mask, 2] = (50 + 50 * normalized_land).astype(np.uint8)   # B
        
        return rgb
    
    def draw_wind_vectors(self):
        """Draw wind vectors at specific latitudes"""
        try:
            # Define specific latitudes where we want vectors (north and south)
            target_latitudes = [80, 70, 60, 50, 40, 30, 20, 10, 0, -10, -20, -30, -40, -50, -60, -70, -80]
            
            # Base step size for longitude spacing
            x_step = self.sim.map_width // 30
            x_indices = np.arange(0, self.sim.map_width, x_step)
            
            # Find y-coordinates for each target latitude
            y_indices = []
            
            # Cache the equator index if we haven't already
            if not hasattr(self, '_equator_y_index'):
                # For each row, get the first column's latitude
                latitudes = self.sim.latitude[:, 0].copy()  # Make a copy to avoid reference issues
                
                # Find where latitude changes from positive to negative
                equator_indices = np.where(np.diff(np.signbit(latitudes)))[0]
                if len(equator_indices) > 0:
                    # If we found a sign change, use that as the equator
                    self._equator_y_index = equator_indices[0]
                else:
                    # Otherwise, find the closest to zero
                    self._equator_y_index = np.abs(latitudes).argmin()
                    
                # Also cache other latitude indices
                self._latitude_indices = {}
                for lat in target_latitudes:
                    if lat != 0:  # Skip equator as we handle it separately
                        self._latitude_indices[lat] = np.abs(latitudes - lat).argmin()
            
            # Add all latitude indices
            for lat in target_latitudes:
                if lat == 0:
                    y_indices.append(self._equator_y_index)
                else:
                    y_indices.append(self._latitude_indices[lat])
            
            # Create coordinate grids
            X, Y = np.meshgrid(x_indices, y_indices)
            
            # Sample wind components
            u_sampled = self.sim.u[Y, X]
            v_sampled = self.sim.v[Y, X]
            
            # Calculate magnitudes for scaling
            magnitudes = np.sqrt(u_sampled**2 + v_sampled**2)
            max_mag = np.max(magnitudes)
            
            # Calculate a reasonable scale factor for arrows
            scale = 0.6  # Adjust for visual appeal
            
            # Normalize vectors for display
            if max_mag > 0:
                u_scaled = u_sampled / max_mag * scale
                v_scaled = v_sampled / max_mag * scale
            else:
                u_scaled = u_sampled * 0
                v_scaled = v_sampled * 0
            
            # Clear existing wind vectors
            self.canvas.delete("wind_vector")
            
            # Draw each arrow
            for i in range(len(y_indices)):
                for j in range(len(x_indices)):
                    # Get coordinates
                    x, y = X[i, j], Y[i, j]
                    
                    # Get scaled vector components
                    u = u_scaled[i, j]
                    v = v_scaled[i, j]
                    
                    # Always show equator vectors even if small
                    if i == target_latitudes.index(0):
                        # Force minimum magnitude for equator vectors to ensure visibility
                        mag = max(np.sqrt(u**2 + v**2), 0.01)
                        if mag > 0:
                            u = u / mag * 0.01
                            v = v / mag * 0.01
                    else:
                        # Skip very small vectors for other latitudes
                        if abs(u) < 1e-6 and abs(v) < 1e-6:
                            continue
                    
                    # Calculate endpoint
                    arrow_length = 15  # Base length in pixels
                    dx = u * arrow_length
                    dy = v * arrow_length
                    
                    # Use white color for all wind vectors
                    color = "#FFFFFF"  # White
                    
                    # Draw arrow
                    self.canvas.create_line(
                        x, y, x + dx, y + dy,
                        arrow=tk.LAST,
                        width=1,
                        fill=color,
                        tags="wind_vector"
                    )
                    
        except Exception as e:
            print(f"Error drawing wind vectors: {e}")
            traceback.print_exc()
    
    def draw_current_arrows(self):
        """Draw ocean current vectors"""
        try:
            # Skip if ocean currents aren't available
            if not hasattr(self.sim, 'ocean_u') or self.sim.ocean_u is None:
                return
                
            # Sample ocean currents at a lower resolution for visualization
            sample_step = 10  # Display every Nth grid point
            
            # Create grid of points to sample
            x_indices = np.arange(0, self.sim.map_width, sample_step)
            y_indices = np.arange(0, self.sim.map_height, sample_step)
            X, Y = np.meshgrid(x_indices, y_indices)
            
            # Sample current components
            u_sampled = self.sim.ocean_u[Y, X]
            v_sampled = self.sim.ocean_v[Y, X]
            
            # Mask for ocean points
            is_ocean = self.sim.elevation[Y, X] <= 0
            
            # Calculate magnitudes for scaling
            magnitudes = np.sqrt(u_sampled**2 + v_sampled**2)
            max_mag = np.max(magnitudes)
            
            # Calculate scale factor for arrows (proportional to wind vectors but different scale)
            scale = 1.0  # Adjust for visual appeal
            
            # Normalize vectors for display
            if max_mag > 0:
                u_scaled = u_sampled / max_mag * scale
                v_scaled = v_sampled / max_mag * scale
            else:
                u_scaled = u_sampled * 0
                v_scaled = v_sampled * 0
            
            # Clear existing current vectors
            self.canvas.delete("current_vector")
            
            # Draw each arrow
            for i in range(len(y_indices)):
                for j in range(len(x_indices)):
                    # Skip land points
                    if not is_ocean[i, j]:
                        continue
                        
                    # Get coordinates
                    x, y = X[i, j], Y[i, j]
                    
                    # Get scaled vector components
                    u = u_scaled[i, j]
                    v = v_scaled[i, j]
                    
                    # Skip very small vectors
                    if abs(u) < 1e-5 and abs(v) < 1e-5:
                        continue
                    
                    # Calculate endpoint
                    arrow_length = 12  # Base length in pixels
                    dx = u * arrow_length
                    dy = v * arrow_length
                    
                    # Use magnitude to determine color
                    mag = magnitudes[i, j]
                    if max_mag > 0:
                        intensity = int(mag / max_mag * 200)
                    else:
                        intensity = 0
                    
                    # Ocean current color gradient (cyan to dark blue)
                    r = 0
                    g = 128 + intensity // 2
                    b = 192 + intensity // 4
                    
                    color = f"#{r:02x}{g:02x}{b:02x}"
                    
                    # Draw arrow
                    self.canvas.create_line(
                        x, y, x + dx, y + dy,
                        arrow=tk.LAST,
                        width=1,
                        fill=color,
                        tags="current_vector"
                    )
                    
        except Exception as e:
            print(f"Error drawing current vectors: {e}")
            traceback.print_exc()
            
    def _delayed_pressure_update(self):
        """Handle delayed pressure map update"""
        try:
            # Calculate pressure visualization
            rgba_data = self.map_pressure_to_color(self.sim.pressure)
            
            # Convert to PIL and then to PhotoImage
            image = Image.fromarray(rgba_data, 'RGBA')
            photo_img = ImageTk.PhotoImage(image)
            
            # Store in cache
            self.image_cache['pressure'] = photo_img
            self._pressure_image = photo_img
            
            # Clear canvas and add the image
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo_img, tags="map")
        except Exception as e:
            print(f"Error in _delayed_pressure_update: {e}")
            traceback.print_exc()
            
    def _delayed_precipitation_update(self):
        """Handle delayed precipitation map update"""
        try:
            # Skip if no precipitation data
            if not hasattr(self.sim, 'precipitation') or self.sim.precipitation is None:
                # Display a placeholder
                self.canvas.delete("all")
                self.canvas.create_text(
                    self.canvas.winfo_width() // 2,
                    self.canvas.winfo_height() // 2,
                    text="No precipitation data available",
                    fill="white"
                )
                return
                
            # Generate precipitation visualization
            colors = self.map_precipitation_to_color(self.sim.precipitation)
            
            # Convert to PIL and then to PhotoImage
            image = Image.fromarray(colors)
            photo_img = ImageTk.PhotoImage(image)
            
            # Store in cache
            self.image_cache['precipitation'] = photo_img
            self._cached_precip_image = photo_img
            
            # Clear canvas and add the image
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo_img, tags="map")
        except Exception as e:
            print(f"Error in _delayed_precipitation_update: {e}")
            traceback.print_exc()
    
    def update_map(self):
        """Update the map display with improved visualization"""
        try:
            # Check if we're running in the main thread
            if not hasattr(self.sim, 'root') or not self.sim.root.winfo_exists():
                print("Skipping update_map: root widget doesn't exist or isn't ready")
                return
            
            # Track if the selected layer has changed to force a full update
            if self._last_selected_layer is None:
                self._last_selected_layer = self.sim.selected_layer.get()
            layer_changed = self._last_selected_layer != self.sim.selected_layer.get()
            self._last_selected_layer = self.sim.selected_layer.get()
            
            # Defer expensive operations when switching layers
            if layer_changed and self.sim.selected_layer.get() == "Pressure":
                # When first switching to pressure, show a loading indicator
                self.canvas.delete("all")
                self.canvas.create_text(
                    self.canvas.winfo_width() // 2, 
                    self.canvas.winfo_height() // 2,
                    text="Loading pressure map...",
                    fill="white",
                    font=("Arial", 14)
                )
                self.sim.root.after(100, self._delayed_pressure_update)
                return
                
            # Same for precipitation
            if layer_changed and self.sim.selected_layer.get() == "Precipitation":
                # Show loading indicator
                self.canvas.delete("all")
                self.canvas.create_text(
                    self.canvas.winfo_width() // 2, 
                    self.canvas.winfo_height() // 2,
                    text="Loading precipitation map...",
                    fill="white",
                    font=("Arial", 14)
                )
                self.sim.root.after(100, self._delayed_precipitation_update)
                return
            
            # Special case for Wind layer
            if self.sim.selected_layer.get() == "Wind":
                # Use altitude map as background for wind vectors
                display_data = self.map_altitude_to_color(self.sim.elevation)
                image = Image.fromarray(display_data.astype('uint8'))
                photo_img = ImageTk.PhotoImage(image)
                
                # Store reference to prevent garbage collection
                self.image_cache['wind_bg'] = photo_img
                
                # Update canvas with terrain background
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, anchor=tk.NW, image=photo_img)
                
                # Always redraw wind vectors for Wind layer to ensure they're current
                self.draw_wind_vectors()
                return
            
            # Get current layer
            current_layer = self.sim.selected_layer.get()
            
            # Check if layer has changed and force a cache clean
            if hasattr(self, '_last_selected_layer') and self._last_selected_layer != current_layer:
                self._clean_image_cache()
            
            # Update the last selected layer
            self._last_selected_layer = current_layer
            
            # If it's pressure and we have a cached image, use that
            if current_layer == "Pressure" and self._pressure_image is not None:
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self._pressure_image, tags="map")
                return
                
            # If it's precipitation and we have a cached image, use that  
            if current_layer == "Precipitation" and self._cached_precip_image is not None:
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self._cached_precip_image, tags="map")
                return
            
            # Generate visualization data based on selected layer
            if current_layer == "Elevation":
                rgb_data = self.map_to_grayscale(self.sim.elevation_normalized)
                img = Image.fromarray(rgb_data)
                photo_img = ImageTk.PhotoImage(image=img)
                self.image_cache['elevation'] = photo_img
            elif current_layer == "Altitude":
                rgb_data = self.map_altitude_to_color(self.sim.elevation)
                img = Image.fromarray(rgb_data)
                photo_img = ImageTk.PhotoImage(image=img)
                self.image_cache['altitude'] = photo_img
            elif current_layer == "Temperature":
                rgba_data = self.map_temperature_to_color(self.sim.temperature_celsius)
                img = Image.fromarray(rgba_data, 'RGBA')
                photo_img = ImageTk.PhotoImage(image=img)
                self.image_cache['temperature'] = photo_img
            elif current_layer == "Ocean Temperature":
                # Create normalized ocean temperature data
                # Scale from -2°C to 30°C (typical ocean temperature range)
                min_ocean_temp = -2
                max_ocean_temp = 30
                normalized_temp = (self.sim.temperature_celsius - min_ocean_temp) / (max_ocean_temp - min_ocean_temp)
                np.clip(normalized_temp, 0, 1, out=normalized_temp)  # Ensure values are between 0-1
                rgb_data = self.map_ocean_temperature_to_color(normalized_temp)
                img = Image.fromarray(rgb_data)
                photo_img = ImageTk.PhotoImage(image=img)
                self.image_cache['ocean_temperature'] = photo_img
            elif current_layer == "Pressure":
                # Generate the pressure visualization directly
                rgba_data = self.map_pressure_to_color(self.sim.pressure)
                img = Image.fromarray(rgba_data, 'RGBA')
                # Cache the image
                self._pressure_image = ImageTk.PhotoImage(img)
                self.image_cache['pressure'] = self._pressure_image
                photo_img = self._pressure_image
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, anchor=tk.NW, image=photo_img, tags="map")
                return
            elif current_layer == "Precipitation" and hasattr(self.sim, 'precipitation'):
                rgba_data = self.map_precipitation_to_color(self.sim.precipitation)
                img = Image.fromarray(rgba_data, 'RGBA')
            else:
                # Default to elevation if layer not recognized
                rgb_data = self.map_to_grayscale(self.sim.elevation_normalized)
                img = Image.fromarray(rgb_data)
            
            # Convert to PhotoImage for Tkinter
            photo_img = ImageTk.PhotoImage(image=img)
            
            # Store reference to prevent garbage collection
            self.image_cache[current_layer] = photo_img
            
            # Clear canvas
            self.canvas.delete("all")
            
            # Draw base map
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo_img, tags="map")
            
            # Draw cloud overlay if enabled and we're on temperature layer
            if current_layer in ["Temperature"] and hasattr(self.sim, 'show_clouds') and self.sim.show_clouds.get():
                # Get cloud RGBA data
                cloud_rgba = self.map_clouds_to_color()
                
                # If we have cloud data with non-zero alpha channel
                if np.any(cloud_rgba[:, :, 3] > 0):
                    cloud_img = Image.fromarray(cloud_rgba, 'RGBA')
                    cloud_photo = ImageTk.PhotoImage(image=cloud_img)
                    self.image_cache['clouds'] = cloud_photo
                    
                    # Draw cloud overlay
                    self.canvas.create_image(0, 0, anchor=tk.NW, image=cloud_photo, tags="clouds")
            
            # Draw wind vectors if on temperature layer with wind overlay enabled
            if current_layer == "Temperature" and hasattr(self.sim, 'show_wind') and self.sim.show_wind.get():
                self.draw_wind_vectors()
                
            # Draw ocean currents if on ocean temperature layer
            if current_layer == "Ocean Temperature" and hasattr(self.sim, 'show_currents') and self.sim.show_currents.get():
                self.draw_current_arrows()
                
        except Exception as e:
            print(f"Error in update_map: {e}")
            traceback.print_exc()
    
    def update_visualization_loop(self):
        """Continuous loop for updating visualization in separate thread"""
        try:
            print("Visualization thread started")
            
            # Run continuously until thread is explicitly stopped
            while True:
                # Check if we should continue running
                if hasattr(self.sim, 'visualization_active'):
                    if not self.sim.visualization_active.is_set():
                        print("Visualization thread stopping (flag cleared)")
                        break
                else:
                    # If visualization_active doesn't exist, create it
                    print("Creating missing visualization_active attribute")
                    self.sim.visualization_active = threading.Event()
                    self.sim.visualization_active.set()
                
                # Schedule update in the main thread
                try:
                    if hasattr(self.sim, 'root') and self.sim.root.winfo_exists():
                        # Cancel any existing scheduled update
                        if hasattr(self, '_update_after_id'):
                            try:
                                self.sim.root.after_cancel(self._update_after_id)
                            except:
                                pass  # Ignore errors if the scheduled update no longer exists
                        
                        # Schedule a new update
                        self._update_after_id = self.sim.root.after(0, self.update_visualization)
                except Exception as e:
                    print(f"Error scheduling visualization update: {e}")
                
                # Sleep for a short time to prevent excessive CPU usage
                time.sleep(0.033)  # ~30 FPS
            
            print("Visualization thread ended")
        except Exception as e:
            print(f"Error in visualization loop: {e}")
            traceback.print_exc()

    def update_visualization(self):
        """Update the map display with current data"""
        try:
            # This method is called in the main thread via root.after
            
            # Ensure we're not updating too frequently
            current_time = time.time()
            if current_time - self._last_update_time < 0.033:  # ~30 FPS max
                return
                
            self._last_update_time = current_time
            
            # Call through to the safe update method
            self._update_visualization_safe()
            
            # Clear expired cache items
            self._clean_image_cache()
            
        except Exception as e:
            print(f"Error in update_visualization: {e}")
            traceback.print_exc()
    
    def _clean_image_cache(self):
        """Clean up the image cache to prevent memory leaks"""
        # Keep only the current layer's image and a few essential ones
        current_layer = getattr(self.sim, 'selected_layer', None)
        if current_layer:
            current_layer = current_layer.get()
        
        # Create a new cache with only the items we want to keep
        new_cache = {}
        
        # Keep the current layer's image
        if current_layer and current_layer in self.image_cache:
            new_cache[current_layer] = self.image_cache[current_layer]
        
        # Also keep alternative maps for the same layer (like temperature variations)
        # These specific combinations should be preserved for layer-switching performance
        layer_combinations = {
            'Temperature': ['temperature', 'temperature_with_clouds'],
            'Wind': ['wind_bg', 'wind_vectors'],
            'Pressure': ['pressure'],
            'Precipitation': ['precipitation'],
            'Elevation': ['elevation'],
            'Altitude': ['altitude'],
            'Ocean Temperature': ['ocean_temperature']
        }
        
        # Preserve any related images for the current layer
        if current_layer and current_layer in layer_combinations:
            for key in layer_combinations[current_layer]:
                if key in self.image_cache:
                    new_cache[key] = self.image_cache[key]
        
        # Preserve any special images that are actively used
        if self._last_selected_layer and self._last_selected_layer in self.image_cache:
            new_cache[self._last_selected_layer] = self.image_cache[self._last_selected_layer]
        
        # Replace the old cache with the new one
        self.image_cache = new_cache
    
    def _update_visualization_safe(self):
        """Safely update visualization with error handling"""
        try:
            # Only update if GUI exists and is ready
            if not hasattr(self.sim, 'root') or not self.sim.root.winfo_exists():
                return
            
            # Check if we need to update wind vectors
            is_wind_layer = self.sim.selected_layer.get() == "Wind"
            needs_wind_update = is_wind_layer or (
                self.sim.selected_layer.get() == "Temperature" and 
                hasattr(self.sim, 'show_wind') and 
                self.sim.show_wind.get()
            )
            
            # Update the map display
            self.update_map()
            
            # Force redraw of wind vectors if needed
            if needs_wind_update:
                self.draw_wind_vectors()
                
        except Exception as e:
            print(f"Error in _update_visualization_safe: {e}")
            traceback.print_exc()
            
    def update_zoom_view(self, event):
        """
        Update the zoom window display when the user moves the mouse over the main map or zoomed view.
        
        This method:
        1. Takes mouse coordinates from the event
        2. Determines the region around the cursor to display in the zoom window
        3. Creates a magnified view of that region
        4. Updates the zoom dialog with the new view
        
        When the mouse is over the zoom window itself, the coordinates are translated
        back to the main map coordinates by the zoom window's event handler.
        """
        try:
            # Skip if zoom_dialog doesn't exist
            if not hasattr(self.sim, 'zoom_dialog') or self.sim.zoom_dialog is None:
                return
                
            if not self.sim.zoom_dialog or not hasattr(self.sim, 'zoom_dialog') or not self.sim.zoom_dialog.winfo_exists():
                return
                
            # Get current layer
            current_layer = self.sim.selected_layer.get()
            
            # Get map dimensions
            map_height, map_width = self.sim.map_height, self.sim.map_width
            
            # Get mouse coordinates
            x, y = event.x, event.y
            
            # Skip if out of bounds
            if x < 0 or y < 0 or x >= map_width or y >= map_height:
                return
                
            # Get view size
            view_size = self.sim.zoom_dialog.view_size
            
            # Calculate the region to display
            half_view = view_size // 2
            x_start = max(0, x - half_view)
            y_start = max(0, y - half_view)
            x_end = min(map_width, x + half_view + 1)
            y_end = min(map_height, y + half_view + 1)
            
            # Ensure we get exactly view_size pixels
            if x_end - x_start < view_size:
                if x_start == 0:
                    x_end = min(map_width, x_start + view_size)
                else:
                    x_start = max(0, x_end - view_size)
            
            if y_end - y_start < view_size:
                if y_start == 0:
                    y_end = min(map_height, y_start + view_size)
                else:
                    y_start = max(0, y_end - view_size)
            
            # Get data based on selected layer
            if current_layer == "Elevation":
                view_data = self.map_elevation_to_color(self.sim.elevation)[y_start:y_end, x_start:x_end]
            elif current_layer == "Temperature":
                view_data = self.map_temperature_to_color(self.sim.temperature_celsius)[y_start:y_end, x_start:x_end]
            elif current_layer == "Pressure":
                view_data = self.map_pressure_to_color(self.sim.pressure)[y_start:y_end, x_start:x_end]
            elif current_layer == "Wind":
                view_data = self.map_altitude_to_color(self.sim.elevation)[y_start:y_end, x_start:x_end]
                # We'll add wind vectors separately
            elif current_layer == "Biomes":
                if hasattr(self.sim, 'biomes') and self.sim.biomes is not None:
                    # Implement biome color mapping when available
                    view_data = self.map_altitude_to_color(self.sim.elevation)[y_start:y_end, x_start:x_end]
                else:
                    view_data = self.map_altitude_to_color(self.sim.elevation)[y_start:y_end, x_start:x_end]
            elif current_layer == "Ocean Temperature":
                # Normalize ocean temps to 0-1 range for better visualization
                ocean_temp_normalized = MapGenerator.normalize_data(self.sim.ocean_temperature)
                view_data = self.map_ocean_temperature_to_color(ocean_temp_normalized)[y_start:y_end, x_start:x_end]
            elif current_layer == "Precipitation":
                view_data = self.map_precipitation_to_color(self.sim.precipitation)[y_start:y_end, x_start:x_end]
            elif current_layer == "Ocean Currents":
                # For ocean currents, still use elevation as background
                view_data = self.map_altitude_to_color(self.sim.elevation)[y_start:y_end, x_start:x_end]
            else:
                # Default to terrain
                view_data = self.map_altitude_to_color(self.sim.elevation)[y_start:y_end, x_start:x_end]
            
            # Create zoomed image
            img = Image.fromarray(view_data)
            img = img.resize((self.sim.zoom_dialog.view_size * self.sim.zoom_dialog.zoom_factor,) * 2, 
                            Image.Resampling.NEAREST)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            self.sim.zoom_dialog.canvas.delete("image")  # Only delete image, keep crosshair
            self.sim.zoom_dialog.canvas.create_image(0, 0, anchor="nw", image=photo, tags="image")
            self.sim.zoom_dialog.canvas.tag_raise('crosshair')  # Ensure crosshair stays on top
            self.sim.zoom_dialog.image = photo  # Keep reference
            
            # Position dialog near cursor but not under it
            dialog_x = self.sim.root.winfo_rootx() + event.x + 20
            dialog_y = self.sim.root.winfo_rooty() + event.y + 20
            
            # Ensure dialog stays within screen bounds
            screen_width = self.sim.root.winfo_screenwidth()
            screen_height = self.sim.root.winfo_screenheight()
            dialog_width = self.sim.zoom_dialog.view_size * self.sim.zoom_dialog.zoom_factor
            dialog_height = dialog_width
            
            # Adjust position if would be off-screen
            if dialog_x + dialog_width > screen_width:
                dialog_x = screen_width - dialog_width - 20
            if dialog_y + dialog_height > screen_height:
                dialog_y = screen_height - dialog_height - 20
            
            # Position the dialog
            self.sim.zoom_dialog.geometry(f"{dialog_width}x{dialog_height}+{dialog_x}+{dialog_y}")
            
            # Add wind vectors if showing wind layer
            if current_layer == "Wind":
                # Clear existing vectors
                self.sim.zoom_dialog.canvas.delete("vector")
                
                # Draw wind vectors on the zoomed view
                vector_spacing = 5  # Draw vectors every N pixels
                vector_scale = 3.0  # Scale factor for vectors
                
                for i in range(0, view_size, vector_spacing):
                    for j in range(0, view_size, vector_spacing):
                        if y_start + i < map_height and x_start + j < map_width:
                            u_val = self.sim.u[y_start + i, x_start + j]
                            v_val = self.sim.v[y_start + i, x_start + j]
                            
                            # Scale vectors based on wind strength
                            magnitude = np.sqrt(u_val**2 + v_val**2)
                            if magnitude > 0:
                                # Scale vector length by zoom factor
                                dx = -u_val / magnitude * vector_scale * self.sim.zoom_dialog.zoom_factor
                                dy = -v_val / magnitude * vector_scale * self.sim.zoom_dialog.zoom_factor
                                
                                # Calculate pixel positions
                                x1 = j * self.sim.zoom_dialog.zoom_factor
                                y1 = i * self.sim.zoom_dialog.zoom_factor
                                x2 = x1 + dx
                                y2 = y1 + dy
                                
                                # Draw the vector
                                self.sim.zoom_dialog.canvas.create_line(
                                    x1, y1, x2, y2, 
                                    fill="white", 
                                    arrow=tk.LAST,
                                    width=1,
                                    tags="vector"
                                )
            
            # Draw ocean currents if showing ocean current layer
            elif current_layer == "Ocean Currents":
                # Clear existing vectors
                self.sim.zoom_dialog.canvas.delete("vector")
                
                if hasattr(self.sim, 'current_u') and hasattr(self.sim, 'current_v'):
                    # Draw current vectors on the zoomed view
                    vector_spacing = 7  # Draw vectors every N pixels
                    vector_scale = 5.0  # Scale factor for current vectors
                    
                    for i in range(0, view_size, vector_spacing):
                        for j in range(0, view_size, vector_spacing):
                            if y_start + i < map_height and x_start + j < map_width:
                                # Only draw vectors in ocean areas
                                if self.sim.elevation[y_start + i, x_start + j] <= 0:
                                    u_val = self.sim.current_u[y_start + i, x_start + j]
                                    v_val = self.sim.current_v[y_start + i, x_start + j]
                                    
                                    # Scale vectors based on current strength
                                    magnitude = np.sqrt(u_val**2 + v_val**2)
                                    if magnitude > 0.01:  # Only draw if magnitude is significant
                                        # Scale vector length by zoom factor
                                        dx = -u_val / magnitude * vector_scale * self.sim.zoom_dialog.zoom_factor
                                        dy = -v_val / magnitude * vector_scale * self.sim.zoom_dialog.zoom_factor
                                        
                                        # Calculate pixel positions
                                        x1 = j * self.sim.zoom_dialog.zoom_factor
                                        y1 = i * self.sim.zoom_dialog.zoom_factor
                                        x2 = x1 + dx
                                        y2 = y1 + dy
                                        
                                        # Draw the vector
                                        self.sim.zoom_dialog.canvas.create_line(
                                            x1, y1, x2, y2, 
                                            fill="#00AAFF",  # Light blue for ocean currents
                                            arrow=tk.LAST,
                                            width=1,
                                            tags="vector"
                                        )
            
        except Exception as e:
            print(f"Error updating zoom view: {e}")
            traceback.print_exc() 