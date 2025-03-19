import numpy as np
import time
import traceback
import tkinter as tk
from PIL import Image, ImageTk
from scipy.ndimage import gaussian_filter
from map_generation import MapGenerator
import threading
import queue

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
        
        # Zoom view update tracking
        self._last_zoom_update_time = 0
        self._zoom_update_pending = False
        self._zoom_debounce_delay = 100  # Milliseconds between zoom updates
        
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
        
        # Scale down the alpha a bit to avoid completely white areas
        max_alpha = 200
        alpha = np.clip(alpha, 0, max_alpha)
        
        # Set alpha channel
        rgba[:, :, 3] = alpha
        
        return rgba
    
    def map_humidity_to_color(self, humidity_data):
        """Map humidity data to RGB colors"""
        try:
            # Create output array for RGB
            rgb = np.zeros((humidity_data.shape[0], humidity_data.shape[1], 3), dtype=np.uint8)
            
            # Define colors for humidity gradient (pale blue to deep blue)
            # Format: low_humidity_color, high_humidity_color
            humidity_colors = np.array([
                [220, 230, 255],  # Pale blue (low humidity)
                [20, 80, 180]     # Deep blue (high humidity)
            ])
            
            # Normalize humidity data to 0-1 range if it's not already
            humidity_normalized = np.clip(humidity_data, 0, 1)
            
            # Apply color gradient based on humidity
            # For each RGB channel
            for c in range(3):
                # Interpolate between low and high humidity colors
                rgb[:, :, c] = humidity_colors[0, c] + humidity_normalized * (humidity_colors[1, c] - humidity_colors[0, c])
            
            return rgb.astype(np.uint8)
        except Exception as e:
            print(f"Error in map_humidity_to_color: {e}")
            traceback.print_exc()
            # Return a safe fallback
            return self.map_elevation_to_color(self.sim.elevation)
            
    def map_cloud_cover_to_color(self, cloud_cover_data):
        """Map cloud cover data to RGB colors for direct visualization"""
        try:
            # Create output array for RGB
            rgb = np.zeros((cloud_cover_data.shape[0], cloud_cover_data.shape[1], 3), dtype=np.uint8)
            
            # Define colors for cloud gradient (light gray to white)
            # Format: clear_sky_color, full_cloud_color
            cloud_colors = np.array([
                [135, 206, 235],  # Sky blue (clear sky)
                [255, 255, 255]   # White (full cloud cover)
            ])
            
            # Normalize cloud cover data to 0-1 range if it's not already
            cloud_normalized = np.clip(cloud_cover_data, 0, 1)
            
            # Apply color gradient based on cloud cover
            # For each RGB channel
            for c in range(3):
                # Interpolate between clear and cloudy colors
                rgb[:, :, c] = cloud_colors[0, c] + cloud_normalized * (cloud_colors[1, c] - cloud_colors[0, c])
            
            return rgb.astype(np.uint8)
        except Exception as e:
            print(f"Error in map_cloud_cover_to_color: {e}")
            traceback.print_exc()
            # Return a safe fallback
            return self.map_elevation_to_color(self.sim.elevation)
    
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
            
            # Use a multi-phase rendering approach to prevent UI blocking
            current_layer = self.sim.selected_layer.get()
            
            # Phase 1: Clear canvas and show loading indicator for expensive operations
            if layer_changed or current_layer in ["Pressure", "Wind", "Ocean Currents"]:
                # Show loading indicator
                self.canvas.delete("all")
                self.canvas.create_text(
                    self.canvas.winfo_width() // 2, 
                    self.canvas.winfo_height() // 2,
                    text=f"Loading {current_layer.lower()} map...",
                    fill="white",
                    font=("Arial", 14)
                )
                
                # Schedule Phase 2 with a slight delay to allow UI to update
                if current_layer == "Pressure":
                    self.sim.root.after(20, self._delayed_pressure_update)
                elif current_layer == "Precipitation":
                    self.sim.root.after(20, self._delayed_precipitation_update)
                else:
                    # For other layers, proceed directly to final phase
                    self.sim.root.after(20, self._complete_map_update)
                
                return  # Exit early, next phase will be called via after()
            
            # For simple layers, proceed directly to complete update
            self._complete_map_update()
            
        except Exception as e:
            print(f"Error in update_map: {e}")
            traceback.print_exc()
    
    def _complete_map_update(self):
        """Complete the map update - called as the final phase of updating"""
        try:
            # Process idle tasks to prevent UI freezing
            if hasattr(self.sim, 'root') and self.sim.root.winfo_exists():
                self.sim.root.update_idletasks()
            
            current_layer = self.sim.selected_layer.get()
            
            # Schedule different parts of the rendering process with small delays in between
            # This allows the UI thread to process events between render phases
            
            # Phase 1: Clear canvas and add loading indicator
            self.canvas.delete("all")
            
            # For expensive layers, show loading indicator while rendering
            if current_layer in ["Pressure", "Wind", "Ocean Currents", "Precipitation"]:
                loading_id = self.canvas.create_text(
                    self.canvas.winfo_width() // 2, 
                    self.canvas.winfo_height() // 2,
                    text=f"Rendering {current_layer}...",
                    fill="white",
                    font=("Arial", 14)
                )
                
                # Update idle tasks to show loading indicator
                if hasattr(self.sim, 'root') and self.sim.root.winfo_exists():
                    self.sim.root.update_idletasks()
                
                # Schedule Phase 2 with a slight delay to allow UI to update
                if hasattr(self.sim, 'root') and self.sim.root.winfo_exists():
                    self.sim.root.after(10, lambda: self._render_layer_phase2(current_layer, loading_id))
                return
            else:
                # For simpler layers, proceed directly
                self._render_layer_direct(current_layer)
            
        except Exception as e:
            print(f"Error completing map update: {e}")
            traceback.print_exc()
    
    def _render_layer_phase2(self, current_layer, loading_id):
        """Phase 2 of the layer rendering process"""
        try:
            # Handle based on the selected layer
            if current_layer == "Elevation":
                self._render_elevation_layer()
            elif current_layer == "Temperature":
                self._render_temperature_layer()
            elif current_layer == "Pressure":
                # For the most expensive layers, use separate phases with delays
                if hasattr(self.sim, 'root') and self.sim.root.winfo_exists():
                    self.sim.root.after(5, lambda: self._render_pressure_layer())
            elif current_layer == "Wind":
                # For wind layer, handle background first, then vectors separately
                self._render_wind_background()
                # Schedule vector drawing with a delay
                if hasattr(self.sim, 'root') and self.sim.root.winfo_exists():
                    self.sim.root.after(10, self.draw_wind_vectors)
            elif current_layer == "Ocean Temperature":
                self._render_ocean_temperature_layer()
            elif current_layer == "Precipitation":
                if hasattr(self.sim, 'root') and self.sim.root.winfo_exists():
                    self.sim.root.after(5, lambda: self._render_precipitation_layer())
            elif current_layer == "Ocean Currents":
                # First render the ocean temperature background
                self._render_ocean_temperature_layer()
                # Then schedule current vectors with a delay
                if hasattr(self.sim, 'root') and self.sim.root.winfo_exists():
                    self.sim.root.after(10, self.draw_current_arrows)
            elif current_layer == "Humidity":
                self._render_humidity_layer()
            elif current_layer == "Cloud Cover":
                self._render_cloud_cover_layer()
            else:
                # Default to elevation if unknown layer
                self._render_elevation_layer()
            
            # Remove loading indicator if it exists
            if loading_id:
                self.canvas.delete(loading_id)
            
        except Exception as e:
            print(f"Error in rendering phase 2: {e}")
            traceback.print_exc()
    
    def _render_layer_direct(self, current_layer):
        """Direct rendering for simpler layers"""
        try:
            # Handle based on the selected layer
            if current_layer == "Elevation":
                self._render_elevation_layer()
            elif current_layer == "Temperature":
                self._render_temperature_layer()
            elif current_layer == "Humidity":
                self._render_humidity_layer()
            elif current_layer == "Cloud Cover":
                self._render_cloud_cover_layer()
            else:
                # Default to elevation if unknown layer
                self._render_elevation_layer()
        except Exception as e:
            print(f"Error in direct rendering: {e}")
            traceback.print_exc()
    
    def _render_wind_background(self):
        """Render just the background for the wind layer"""
        try:
            # Use altitude map as background for wind vectors
            display_data = self.map_altitude_to_color(self.sim.elevation)
            image = Image.fromarray(display_data.astype('uint8'))
            photo_img = ImageTk.PhotoImage(image)
            self.image_cache['wind_bg'] = photo_img
            
            # Update canvas with terrain background
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo_img)
        except Exception as e:
            print(f"Error rendering wind background: {e}")
            traceback.print_exc()
    
    def _render_elevation_layer(self):
        """Render the elevation layer"""
        # Generate grayscale elevation data
        rgb_data = self.map_to_grayscale(self.sim.elevation_normalized)
        img = Image.fromarray(rgb_data)
        photo_img = ImageTk.PhotoImage(image=img)
        self.image_cache['elevation'] = photo_img
        
        # Clear canvas and draw the image
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo_img, tags="map")
    
    def _render_temperature_layer(self):
        """Render the temperature layer"""
        # Generate the temperature visualization
        rgba_data = self.map_temperature_to_color(self.sim.temperature_celsius)
        img = Image.fromarray(rgba_data, 'RGBA')
        photo_img = ImageTk.PhotoImage(image=img)
        self.image_cache['temperature'] = photo_img
        
        # Clear canvas and draw the image
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo_img, tags="map")
        
        # Draw cloud overlay if enabled
        if hasattr(self.sim, 'show_clouds') and self.sim.show_clouds.get():
            # Get cloud RGBA data
            cloud_rgba = self.map_clouds_to_color()
            
            # If we have cloud data with non-zero alpha channel
            if np.any(cloud_rgba[:, :, 3] > 0):
                cloud_img = Image.fromarray(cloud_rgba, 'RGBA')
                cloud_photo = ImageTk.PhotoImage(image=cloud_img)
                self.image_cache['clouds'] = cloud_photo
                
                # Draw cloud overlay
                self.canvas.create_image(0, 0, anchor=tk.NW, image=cloud_photo, tags="clouds")
        
        # Draw wind vectors if wind overlay enabled
        if hasattr(self.sim, 'show_wind') and self.sim.show_wind.get():
            self.draw_wind_vectors()
    
    def _render_pressure_layer(self):
        """Render the pressure layer in stages to prevent UI blocking"""
        try:
            # If we have a cached image, use it immediately
            if self._pressure_image is not None:
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self._pressure_image, tags="map")
                return
                
            # Create a placeholder to show the user something while rendering
            self.canvas.delete("all")
            loading_text = self.canvas.create_text(
                self.canvas.winfo_width() // 2, 
                self.canvas.winfo_height() // 2,
                text="Generating pressure map...",
                fill="white",
                font=("Arial", 14)
            )
            
            # Update canvas to show loading message
            if hasattr(self.sim, 'root') and self.sim.root.winfo_exists():
                self.sim.root.update_idletasks()
                
            # Schedule the actual pressure generation on a short delay
            if hasattr(self.sim, 'root') and self.sim.root.winfo_exists():
                self.sim.root.after(20, lambda: self._generate_pressure_viz(loading_text))
        except Exception as e:
            print(f"Error starting pressure rendering: {e}")
            traceback.print_exc()
        
    def _generate_pressure_viz(self, loading_text):
        """Generate pressure visualization in a non-blocking way"""
        try:
            # Generate the pressure visualization
            rgba_data = self.map_pressure_to_color(self.sim.pressure)
            
            # Check if root still exists after the potentially long operation
            if not hasattr(self.sim, 'root') or not self.sim.root.winfo_exists():
                return
                
            # Update loading text to indicate progress
            self.canvas.itemconfigure(loading_text, text="Creating pressure image...")
            self.sim.root.update_idletasks()
            
            # Convert to image
            img = Image.fromarray(rgba_data, 'RGBA')
            
            # Schedule the final display step
            if hasattr(self.sim, 'root') and self.sim.root.winfo_exists():
                self.sim.root.after(10, lambda: self._finish_pressure_render(img, loading_text))
        except Exception as e:
            print(f"Error generating pressure visualization: {e}")
            traceback.print_exc()
            
            # Try to clear the loading text
            try:
                if hasattr(self.sim, 'root') and self.sim.root.winfo_exists():
                    self.canvas.delete(loading_text)
            except:
                pass
            
    def _finish_pressure_render(self, img, loading_text):
        """Final stage of pressure rendering to display the image"""
        try:
            # Convert to tkinter image
            self._pressure_image = ImageTk.PhotoImage(img)
            self.image_cache['pressure'] = self._pressure_image
            
            # Clear canvas and draw the image
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self._pressure_image, tags="map")
        except Exception as e:
            print(f"Error finishing pressure render: {e}")
            traceback.print_exc()
    
    def _render_wind_layer(self):
        """Render the wind layer"""
        # Use altitude map as background for wind vectors
        display_data = self.map_altitude_to_color(self.sim.elevation)
        image = Image.fromarray(display_data.astype('uint8'))
        photo_img = ImageTk.PhotoImage(image)
        self.image_cache['wind_bg'] = photo_img
        
        # Update canvas with terrain background
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo_img)
        
        # Draw wind vectors on top
        # Use a timer to avoid UI blocking during vector drawing
        self.sim.root.after(10, self.draw_wind_vectors)
    
    def _render_ocean_temperature_layer(self):
        """Render the ocean temperature layer"""
        # Create normalized ocean temperature data
        # Scale from -2°C to 30°C (typical ocean temperature range)
        min_ocean_temp = -2
        max_ocean_temp = 30
        normalized_temp = (self.sim.temperature_celsius - min_ocean_temp) / (max_ocean_temp - min_ocean_temp)
        np.clip(normalized_temp, 0, 1, out=normalized_temp)  # Ensure values are between 0-1
        
        # Generate the visualization
        rgb_data = self.map_ocean_temperature_to_color(normalized_temp)
        img = Image.fromarray(rgb_data)
        photo_img = ImageTk.PhotoImage(image=img)
        self.image_cache['ocean_temperature'] = photo_img
        
        # Clear canvas and draw the image
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo_img, tags="map")
        
        # Draw ocean currents if enabled
        if hasattr(self.sim, 'show_currents') and self.sim.show_currents.get():
            # Draw currents in a separate after callback to prevent UI blocking
            self.sim.root.after(10, self.draw_current_arrows)
    
    def _render_precipitation_layer(self):
        """Render the precipitation layer"""
        # If we have a cached image, use it
        if hasattr(self, '_cached_precip_image') and self._cached_precip_image is not None:
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self._cached_precip_image, tags="map")
            return
        
        # Otherwise generate the precipitation visualization
        if hasattr(self.sim, 'precipitation'):
            rgba_data = self.map_precipitation_to_color(self.sim.precipitation)
            img = Image.fromarray(rgba_data, 'RGBA')
            self._cached_precip_image = ImageTk.PhotoImage(img)
            self.image_cache['precipitation'] = self._cached_precip_image
            
            # Clear canvas and draw the image
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self._cached_precip_image, tags="map")
        else:
            # If no precipitation data available, show a blank canvas
            self.canvas.delete("all")
            self.canvas.create_text(
                self.canvas.winfo_width() // 2, 
                self.canvas.winfo_height() // 2,
                text="No precipitation data available",
                fill="white",
                font=("Arial", 14)
            )
    
    def _render_ocean_currents_layer(self):
        """Render the ocean currents layer"""
        # Use ocean temperature as background
        self._render_ocean_temperature_layer()
        
        # Draw ocean current vectors in a delayed callback
        self.sim.root.after(10, self.draw_current_arrows)
    
    def _render_humidity_layer(self):
        """Render the humidity layer"""
        # Generate the humidity visualization
        if hasattr(self.sim, 'humidity'):
            rgba_data = self.map_humidity_to_color(self.sim.humidity)
            img = Image.fromarray(rgba_data, 'RGBA')
            photo_img = ImageTk.PhotoImage(image=img)
            self.image_cache['humidity'] = photo_img
            
            # Clear canvas and draw the image
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo_img, tags="map")
        else:
            # If no humidity data available, show a blank canvas
            self.canvas.delete("all")
            self.canvas.create_text(
                self.canvas.winfo_width() // 2, 
                self.canvas.winfo_height() // 2,
                text="No humidity data available",
                fill="white",
                font=("Arial", 14)
            )
    
    def _render_cloud_cover_layer(self):
        """Render the cloud cover layer"""
        if hasattr(self.sim, 'cloud_cover'):
            rgba_data = self.map_cloud_cover_to_color(self.sim.cloud_cover)
            img = Image.fromarray(rgba_data, 'RGBA')
            photo_img = ImageTk.PhotoImage(image=img)
            self.image_cache['cloud_cover'] = photo_img
            
            # Clear canvas and draw the image
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo_img, tags="map")
        else:
            # If no cloud cover data available, show a blank canvas
            self.canvas.delete("all")
            self.canvas.create_text(
                self.canvas.winfo_width() // 2, 
                self.canvas.winfo_height() // 2,
                text="No cloud cover data available",
                fill="white",
                font=("Arial", 14)
            )
    
    def update_visualization_loop(self):
        """Continuous loop for updating visualization in separate thread"""
        try:
            print("Visualization thread started")
            
            # Track timing for performance analysis
            last_update_time = 0
            update_count = 0
            idle_count = 0
            
            # Run continuously until thread is explicitly stopped
            while True:
                try:
                    # Check if we should continue running
                    if hasattr(self.sim, 'visualization_active'):
                        if not self.sim.visualization_active.is_set():
                            break
                    else:
                        # If visualization_active doesn't exist, create it
                        print("Creating missing visualization_active attribute")
                        self.sim.visualization_active = threading.Event()
                        self.sim.visualization_active.set()
                    
                    # Wait for a visualization update event instead of polling
                    # This significantly reduces CPU usage and prevents UI thread blocking
                    update_available = False
                    if hasattr(self.sim, 'visualization_update_ready'):
                        # Wait efficiently with a timeout to allow checking stop flag
                        update_available = self.sim.visualization_update_ready.wait(0.05)  # 50ms timeout
                        
                        if update_available:
                            # Clear the event for the next update
                            self.sim.visualization_update_ready.clear()
                            # Reset idle counter when we get an update
                            idle_count = 0
                    else:
                        # Use a sleep if event not available (fallback)
                        time.sleep(0.05)
                    
                    # Only process updates at most 10 times per second to prevent UI thread overload
                    current_time = time.time()
                    time_since_last_update = current_time - last_update_time
                    
                    # Calculate if it's time for an update (max 10 FPS)
                    update_ready = update_available and time_since_last_update >= 0.1
                    
                    # Force an update periodically even if not signaled
                    # This ensures visualization continues even if signaling mechanism fails
                    force_update = idle_count > 50  # Force update after ~2.5 seconds of inactivity
                    if force_update and time_since_last_update >= 1.0:  # At most once per second
                        update_ready = True
                        idle_count = 0
                    else:
                        idle_count += 1
                    
                    # Check queue only when an update is signaled to reduce contention
                    if update_ready:
                        # Update last update time immediately to prevent race conditions
                        last_update_time = current_time
                        update_count += 1
                        
                        # Process the update through the UI thread
                        self._schedule_map_update()
                        
                        # Add a small sleep after scheduling to reduce thread contention
                        time.sleep(0.01)
                        
                except Exception as e:
                    print(f"Error in visualization loop iteration: {e}")
                    traceback.print_exc()
                    # Continue loop despite error
                    time.sleep(0.1)  # Short delay before retry
        
        except Exception as e:
            print(f"Error in visualization loop: {e}")
            traceback.print_exc()
    
    def _schedule_map_update(self):
        """Schedule a standard map update in the main thread"""
        if hasattr(self.sim, 'root') and self.sim.root.winfo_exists():
            # Cancel any existing scheduled update
            if hasattr(self, '_update_after_id'):
                try:
                    self.sim.root.after_cancel(self._update_after_id)
                except:
                    pass  # Ignore errors if the scheduled update no longer exists
            
            # Use a longer delay (25ms) to give the UI thread time to process events
            # This prevents UI freezing by allowing input events to be processed between visualization updates
            self._update_after_id = self.sim.root.after(25, self._deferred_update_visualization)
    
    def _deferred_update_visualization(self):
        """Deferred update method with safety checks to prevent blocking the UI thread"""
        try:
            # Reset the after ID since this update is now running
            if hasattr(self, '_update_after_id'):
                self._update_after_id = None
            
            # Check if update is still needed
            if not hasattr(self.sim, 'root') or not self.sim.root.winfo_exists():
                return
            
            # Process UI events before rendering to keep UI responsive
            self.sim.root.update_idletasks()
            
            # Update the visualization
            self.update_visualization()
            
            # Process UI events again after rendering
            self.sim.root.update_idletasks()
        except Exception as e:
            print(f"Error in deferred visualization update: {e}")
            traceback.print_exc()
    
    def _schedule_zoom_update(self, x, y):
        """Schedule a zoom view update in the main thread"""
        if not hasattr(self.sim, 'root') or not self.sim.root.winfo_exists():
            self._zoom_update_pending = False
            return
            
        try:
            # Create a synthetic event with the necessary coordinates
            class SyntheticEvent:
                def __init__(self, x, y):
                    self.x = x
                    self.y = y
                    
            event = SyntheticEvent(int(x), int(y))
            
            # Schedule the zoom update with a short timeout
            update_id = self.sim.root.after(10, lambda: self._update_zoom_view_debounced(event))
            
            # Store the after ID to allow cancellation if needed
            if not hasattr(self, '_zoom_update_ids'):
                self._zoom_update_ids = []
                
            # Remove any old update IDs to prevent memory leaks
            while len(self._zoom_update_ids) > 5:  # Keep only the 5 most recent
                old_id = self._zoom_update_ids.pop(0)
                try:
                    self.sim.root.after_cancel(old_id)
                except:
                    pass
                    
            # Add this update ID to the list
            self._zoom_update_ids.append(update_id)
                
        except Exception as e:
            print(f"Error in _schedule_zoom_update: {e}")
            traceback.print_exc()
            self._zoom_update_pending = False  # Reset pending flag on error
    
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
        """Update the zoom window with a magnified view around the mouse cursor"""
        # Skip if there's no zoom dialog
        if not hasattr(self.sim, 'zoom_dialog') or not self.sim.zoom_dialog or not self.sim.zoom_dialog.winfo_exists():
            return
            
        try:
            # Add debouncing to prevent too frequent zoom updates
            current_time = time.time() * 1000  # Convert to milliseconds
            time_since_last = current_time - self._last_zoom_update_time
            
            # If an update is already scheduled or it's too soon since the last update, skip
            if self._zoom_update_pending and time_since_last < 1000:  # Allow retrying after 1 second even if pending
                return
                
            # Reset stale pending updates that might have been forgotten
            if self._zoom_update_pending and time_since_last > 2000:  # If pending for more than 2 seconds
                #print("Resetting stale zoom update request")
                self._zoom_update_pending = False
                
            # If still a pending update, skip
            if self._zoom_update_pending or time_since_last < self._zoom_debounce_delay:
                return
            
            # Try the queue approach for prioritized updates
            success = False
            if hasattr(self.sim, 'visualization_queue') and hasattr(self.sim, 'VIZ_PRIORITY'):
                # Request a low-priority zoom update
                try:
                    # Create a copy of event x and y to avoid reference issues
                    update_request = {
                        "type": "ZOOM_UPDATE",
                        "priority": self.sim.VIZ_PRIORITY["LOW"],
                        "timestamp": time.time(),
                        "event_x": int(event.x),
                        "event_y": int(event.y)
                    }
                    
                    # Only add to queue if not full
                    if not self.sim.visualization_queue.full():
                        self.sim.visualization_queue.put_nowait(update_request)
                        self._zoom_update_pending = True
                        success = True
                except Exception as e:
                    print(f"Error queueing zoom update: {e}")
                    # Fall through to direct update
            
            # If queue approach failed, use direct update
            if not success:
                # Schedule the update with a small delay (direct approach if queue failed)
                self._zoom_update_pending = True
                try:
                    # Make a copy of the event to avoid reference issues
                    class SyntheticEvent:
                        def __init__(self, x, y):
                            self.x = x
                            self.y = y
                            
                    event_copy = SyntheticEvent(int(event.x), int(event.y))
                    self.root.after(10, lambda: self._update_zoom_view_debounced(event_copy))
                except Exception as e:
                    print(f"Error scheduling direct zoom update: {e}")
                    self._zoom_update_pending = False  # Reset flag on error
            
        except Exception as e:
            print(f"Error scheduling zoom update: {e}")
            traceback.print_exc()
            
            # Reset pending state to allow future updates
            self._zoom_update_pending = False
    
    def _update_zoom_view_debounced(self, event):
        """Actual implementation of zoom view update with debouncing"""
        try:
            # Mark that we're processing this update
            self._zoom_update_pending = False
            self._last_zoom_update_time = time.time() * 1000  # Convert to milliseconds
            
            # Get dialog and parameters
            zoom_dialog = self.sim.zoom_dialog
            if not zoom_dialog or not zoom_dialog.winfo_exists():
                return
                
            view_size = zoom_dialog.view_size
            zoom_factor = zoom_dialog.zoom_factor
            
            # Get coordinates
            x, y = event.x, event.y
            
            # Get current layer for visualization
            current_layer = self.sim.selected_layer.get()
            
            # Ensure x and y are within map bounds
            map_width = self.sim.map_width
            map_height = self.sim.map_height
            x = max(0, min(x, map_width - 1))
            y = max(0, min(y, map_height - 1))
            
            # Calculate view boundaries
            half_size = view_size // 2
            x_start = max(0, x - half_size)
            y_start = max(0, y - half_size)
            x_end = min(map_width, x_start + view_size)
            y_end = min(map_height, y_start + view_size)
            
            # Ensure we get exactly view_size pixels if possible
            if x_end - x_start < view_size and x_start == 0:
                x_end = min(map_width, view_size)
            elif x_end - x_start < view_size:
                x_start = max(0, x_end - view_size)
                
            if y_end - y_start < view_size and y_start == 0:
                y_end = min(map_height, view_size)
            elif y_end - y_start < view_size:
                y_start = max(0, y_end - view_size)
            
            # Get the current layer data from the appropriate source
            try:
                if current_layer == "Elevation":
                    img_data = self.map_elevation_to_color(self.sim.elevation)
                elif current_layer == "Temperature":
                    img_data = self.map_temperature_to_color(self.sim.temperature_celsius)
                elif current_layer == "Pressure":
                    img_data = self.map_pressure_to_color(self.sim.pressure)
                elif current_layer == "Wind":
                    img_data = self.map_altitude_to_color(self.sim.elevation)
                elif current_layer == "Ocean Temperature":
                    img_data = self.map_ocean_temperature_to_color(self.sim.temperature_celsius)
                elif current_layer == "Precipitation" and hasattr(self.sim, 'precipitation') and self.sim.precipitation is not None:
                    img_data = self.map_precipitation_to_color(self.sim.precipitation)
                elif current_layer == "Ocean Currents":
                    img_data = self.map_ocean_temperature_to_color(self.sim.temperature_celsius)
                elif current_layer == "Humidity" and hasattr(self.sim, 'humidity') and self.sim.humidity is not None:
                    img_data = self.map_humidity_to_color(self.sim.humidity)
                elif current_layer == "Cloud Cover" and hasattr(self.sim, 'cloud_cover') and self.sim.cloud_cover is not None:
                    img_data = self.map_cloud_cover_to_color(self.sim.cloud_cover)
                else:
                    # Default to elevation if layer not recognized
                    img_data = self.map_elevation_to_color(self.sim.elevation)
            except Exception as e:
                print(f"Error getting layer data: {e}")
                # Use a fallback - create a blank image
                img_data = np.zeros((view_size, view_size, 3), dtype=np.uint8)
                img_data.fill(200)  # Light gray
                
            # Check if zoom dialog still exists after potentially long img_data processing
            if not hasattr(self.sim, 'zoom_dialog') or not self.sim.zoom_dialog or not self.sim.zoom_dialog.winfo_exists():
                return
                
            # Extract the view region
            try:
                # Make sure we don't exceed bounds
                if y_start >= img_data.shape[0] or x_start >= img_data.shape[1]:
                    print("Warning: View region out of bounds")
                    return
                    
                # Get the maximum bounds we can extract
                y_end = min(y_end, img_data.shape[0])
                x_end = min(x_end, img_data.shape[1])
                
                view_data = img_data[y_start:y_end, x_start:x_end]
                
                # Verify dimensions - ensure we have at least a 1x1 pixel area
                if view_data.shape[0] == 0 or view_data.shape[1] == 0:
                    print("Warning: Empty zoom view region")
                    return
            except Exception as e:
                print(f"Error extracting view region: {e}, shape={img_data.shape}, bounds=({x_start}:{x_end}, {y_start}:{y_end})")
                return
                
            # Determine if we're dealing with RGB or RGBA data
            is_rgba = len(view_data.shape) > 2 and view_data.shape[2] == 4
            
            # Convert to a PIL image with the appropriate mode
            try:
                if is_rgba:
                    img = Image.fromarray(view_data.astype(np.uint8), 'RGBA')
                else:
                    img = Image.fromarray(view_data.astype(np.uint8), 'RGB')
            except Exception as e:
                print(f"Error creating image from array: {e}, shape={view_data.shape}, dtype={view_data.dtype}")
                # Try a fallback approach
                if len(view_data.shape) <= 2:
                    # If we have a 2D array, convert to RGB
                    rgb_data = np.stack([view_data]*3, axis=-1)
                    img = Image.fromarray(rgb_data.astype(np.uint8), 'RGB')
                else:
                    # Create a blank image as last resort
                    img = Image.new('RGB', (view_size, view_size), color=(200, 200, 200))
            
            # Check zoom dialog again before creating PhotoImage
            if not hasattr(self.sim, 'zoom_dialog') or not self.sim.zoom_dialog or not self.sim.zoom_dialog.winfo_exists():
                return
                
            # Scale up the image for zoom effect
            canvas_size = view_size * zoom_factor
            img = img.resize((canvas_size, canvas_size), Image.NEAREST)
            
            # Convert to PhotoImage for display
            try:
                photo = ImageTk.PhotoImage(img)
                
                # Store reference to prevent garbage collection
                zoom_dialog.photo = photo
                
                # Clear canvas and display the new image
                zoom_dialog.canvas.delete("all")
                zoom_dialog.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                
                # Add crosshairs again (as they were deleted)
                crosshair_size = zoom_dialog.crosshair_size
                center = canvas_size // 2
                
                # Draw the crosshairs
                # Main crosshair in red
                zoom_dialog.canvas.create_line(center - crosshair_size, center, 
                                      center + crosshair_size, center, 
                                      fill='red', width=2, tags='crosshair')
                zoom_dialog.canvas.create_line(center, center - crosshair_size, 
                                      center, center + crosshair_size, 
                                      fill='red', width=2, tags='crosshair')
                
                # White outline for contrast
                outline_offset = 1
                zoom_dialog.canvas.create_line(center - crosshair_size, center - outline_offset, 
                                      center + crosshair_size, center - outline_offset, 
                                      fill='white', width=1, tags='crosshair')
                zoom_dialog.canvas.create_line(center - crosshair_size, center + outline_offset, 
                                      center + crosshair_size, center + outline_offset, 
                                      fill='white', width=1, tags='crosshair')
                zoom_dialog.canvas.create_line(center - outline_offset, center - crosshair_size, 
                                      center - outline_offset, center + crosshair_size, 
                                      fill='white', width=1, tags='crosshair')
                zoom_dialog.canvas.create_line(center + outline_offset, center - crosshair_size, 
                                      center + outline_offset, center + crosshair_size, 
                                      fill='white', width=1, tags='crosshair')
                
                # Add wind vectors if on Wind layer
                if current_layer == "Wind" and hasattr(self.sim, 'u') and hasattr(self.sim, 'v'):
                    self._draw_zoom_wind_vectors(zoom_dialog, x_start, y_start, x_end, y_end)
                    
                # Add ocean current vectors if on Ocean Currents layer
                if current_layer == "Ocean Currents" and hasattr(self.sim, 'ocean_u') and hasattr(self.sim, 'ocean_v'):
                    self._draw_zoom_current_vectors(zoom_dialog, x_start, y_start, x_end, y_end)
            except Exception as e:
                print(f"Error rendering zoom view: {e}")
                traceback.print_exc()
                
        except Exception as e:
            print(f"Error in _update_zoom_view_debounced: {e}")
            traceback.print_exc()
            
            # Reset pending state to allow future updates
            self._zoom_update_pending = False
    
    def _draw_zoom_wind_vectors(self, zoom_dialog, x_start, y_start, x_end, y_end):
        """Draw wind vector overlay on zoom view"""
        try:
            # Get wind data
            u_data = self.sim.u[y_start:y_end, x_start:x_end]
            v_data = self.sim.v[y_start:y_end, x_start:x_end]
            
            # Skip if data is invalid
            if u_data.shape[0] == 0 or v_data.shape[0] == 0:
                return
                
            # Calculate grid spacing for vectors
            grid_size = 8  # Display a vector every 8 pixels
            zoom_factor = zoom_dialog.zoom_factor
            view_size = zoom_dialog.view_size
            
            # Scale factor for vector length
            scale = 0.2 * zoom_factor
            
            # Use just enough vectors to avoid clutter
            for y in range(0, view_size, grid_size):
                for x in range(0, view_size, grid_size):
                    if y < u_data.shape[0] and x < u_data.shape[1]:
                        # Get wind components
                        u = u_data[y, x]
                        v = v_data[y, x]
                        
                        # Skip negligible wind
                        if abs(u) < 0.5 and abs(v) < 0.5:
                            continue
                            
                        # Calculate vector length
                        speed = (u**2 + v**2)**0.5
                        
                        # Normalize components
                        if speed > 0:
                            u_norm = u / speed
                            v_norm = v / speed
                        else:
                            continue
                            
                        # Calculate scaled vector length
                        length = min(10, speed) * scale
                        
                        # Calculate vector endpoints
                        x1 = (x + 0.5) * zoom_factor
                        y1 = (y + 0.5) * zoom_factor
                        x2 = x1 + u_norm * length
                        y2 = y1 + v_norm * length
                        
                        # Draw the vector line
                        zoom_dialog.canvas.create_line(x1, y1, x2, y2, fill='cyan', width=1, arrow=tk.LAST)
                        
        except Exception as e:
            print(f"Error drawing zoom wind vectors: {e}")
            
    def _draw_zoom_current_vectors(self, zoom_dialog, x_start, y_start, x_end, y_end):
        """Draw ocean current vector overlay on zoom view"""
        try:
            # Get current data
            u_data = self.sim.ocean_u[y_start:y_end, x_start:x_end]
            v_data = self.sim.ocean_v[y_start:y_end, x_start:x_end]
            
            # Skip if data is invalid
            if u_data.shape[0] == 0 or v_data.shape[0] == 0:
                return
                
            # Calculate grid spacing for vectors
            grid_size = 8  # Display a vector every 8 pixels  
            zoom_factor = zoom_dialog.zoom_factor
            view_size = zoom_dialog.view_size
            
            # Scale factor for vector length
            scale = 0.5 * zoom_factor
            
            # Use just enough vectors to avoid clutter
            for y in range(0, view_size, grid_size):
                for x in range(0, view_size, grid_size):
                    if y < u_data.shape[0] and x < u_data.shape[1]:
                        # Check if this is ocean (only show currents in ocean)
                        if y+y_start < self.sim.elevation.shape[0] and x+x_start < self.sim.elevation.shape[1]:
                            if self.sim.elevation[y+y_start, x+x_start] > 0:
                                continue  # Skip land
                        
                        # Get current components
                        u = u_data[y, x]
                        v = v_data[y, x]
                        
                        # Skip negligible current
                        if abs(u) < 0.05 and abs(v) < 0.05:
                            continue
                            
                        # Calculate vector length
                        speed = (u**2 + v**2)**0.5
                        
                        # Normalize components
                        if speed > 0:
                            u_norm = u / speed
                            v_norm = v / speed
                        else:
                            continue
                            
                        # Calculate scaled vector length
                        length = min(5, speed * 10) * scale
                        
                        # Calculate vector endpoints
                        x1 = (x + 0.5) * zoom_factor
                        y1 = (y + 0.5) * zoom_factor
                        x2 = x1 + u_norm * length
                        y2 = y1 + v_norm * length
                        
                        # Draw the vector line
                        zoom_dialog.canvas.create_line(x1, y1, x2, y2, fill='yellow', width=1, arrow=tk.LAST)
                        
        except Exception as e:
            print(f"Error drawing zoom current vectors: {e}")
            traceback.print_exc() 
    
    def force_update(self):
        """Force an immediate update of the visualization in the current thread"""
        try:
            # Only proceed if root still exists
            if not hasattr(self.sim, 'root') or not self.sim.root.winfo_exists():
                return
                
            # Process UI events to ensure UI is responsive
            self.sim.root.update_idletasks()
            
            # Get the current selected layer
            current_layer = self.sim.selected_layer.get()
            
            # Clear the canvas
            self.canvas.delete("all")
            
            # Process main rendering based on layer
            if current_layer == "Elevation":
                self._render_elevation_layer()
            elif current_layer == "Temperature":
                self._render_temperature_layer()
            elif current_layer == "Pressure":
                self._render_pressure_layer()
            elif current_layer == "Wind":
                self._render_wind_layer()
            elif current_layer == "Ocean Temperature":
                self._render_ocean_temperature_layer()
            elif current_layer == "Precipitation":
                self._render_precipitation_layer()
            elif current_layer == "Ocean Currents":
                self._render_ocean_currents_layer()
            elif current_layer == "Humidity":
                self._render_humidity_layer()
            elif current_layer == "Cloud Cover":
                self._render_cloud_cover_layer()
            
            # Update mouse-over info if available
            if hasattr(self.sim, 'update_mouse_over'):
                self.sim.update_mouse_over()
                
            # Update zoom view if it exists
            if hasattr(self.sim, 'zoom_dialog') and self.sim.zoom_dialog and self.sim.zoom_dialog.winfo_exists():
                self.update_zoom_view(None)
                
            # Process UI events again after rendering
            self.sim.root.update_idletasks()
            
        except Exception as e:
            print(f"Error in force_update: {e}")
            traceback.print_exc()