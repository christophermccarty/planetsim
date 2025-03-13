import os
import sys
import time
import math
import json
import numpy as np
import tkinter as tk
import threading
import multiprocessing
import traceback
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from scipy.ndimage import gaussian_filter
import pickle
import random

from temperature import Temperature
from pressure import Pressure
from wind import Wind
from system_stats import SystemStats
from visualization import Visualization
from precipitation import Precipitation

from map_generation import MapGenerator


class SimulationApp:
    def __init__(self):
        """Initialize the simulation"""
        try:
            # Create the root window first
            self.root = tk.Tk()
            self.root.title("Climate Simulation - Planet Sim")
            
            # Initialize all variables
            self.map_size = 512
            self.map_width = self.map_size  # Set default width until image is loaded
            self.map_height = self.map_size # Set default height until image is loaded
            
            # Variable to track the selected layer - this needs to exist before setup_gui
            self.selected_layer = tk.StringVar(value="Elevation")
            
            # Mouse tracking
            self.last_mouse_x = 0
            self.last_mouse_y = 0
            
            # Time and simulation variables
            self.time_step = 0
            self.time_step_seconds = 60 * 30  # Simulation time step in seconds (30 minutes)
            self.Omega = 7.2921159e-5
            self.P0 = 101325
            self.desired_simulation_step_time = 0.1  # 0.1 seconds between simulation steps

            # Master switch for system stats printing is now in SystemStats class

            # Add energy budget tracking
            self.energy_budget = {
                'incoming': 0,
                'outgoing': 0,
                'greenhouse': 0,
                'albedo': 0,
                'cloud_effect': 0,
                'ocean_flux': 0,
                'ocean_heat_content': 0,
                'ocean_heat_change': 0,
                'evaporation': 0,
                'precipitation': 0,
                'humidity_transport': 0,
                'temperature_history': []  # Initialize temperature history list
            }
            
            # Adjustable climate parameters
            self.climate_params = {
                'solar_constant': 1361,      # Solar constant (W/m²)
                'albedo_land': 0.3,          # Land albedo
                'albedo_ocean': 0.06,        # Ocean albedo
                'emissivity': 0.95,          # Increased to allow more heat escape
                'climate_sensitivity': 3.0,   # Reduced for more stable response
                'greenhouse_strength': 2.5,   # Reduced greenhouse effect
                'heat_capacity_land': 0.8e6,    # Doubled land heat capacity for stability
                'heat_capacity_ocean': 4.2e6,   # Doubled ocean heat capacity for stability
                'ocean_heat_transfer': 0.3,   # Ocean heat transfer
                'atmospheric_heat_transfer': 2.0,  # Increased from 0.5 to 2.0 for smoother diffusion
                'baseline_land_temp': 15,   # Baseline land temperature
                'max_ocean_temp': 32.0,       # Maximum ocean temperature
                'min_ocean_temp': -2.0        # Minimum ocean temperature
            }
            
            # Seeds
            self.seed = 4200
            self.elevation_seed = self.seed
            
            # Simulation parameters
            self.global_octaves = 5
            self.global_frequency = 2.0
            self.global_lacunarity = 2.0
            self.global_persistence = 0.5
            
            # Atmospheric parameters
            # Greenhouse gas concentrations
            # Values based on approximate current atmospheric concentrations
            self.co2_concentration = 420  # ppm
            self.ch4_concentration = 1900  # ppb
            self.n2o_concentration = 330  # ppb

            # Radiative forcing coefficients (W/m² per ppm or ppb)
            self.radiative_forcing_co2 = 5.35 * np.log(self.co2_concentration / 280)  # Reference CO2 is 280 ppm
            self.radiative_forcing_ch4 = 0.036 * (self.ch4_concentration - 1750) / 1000  # Approximation
            self.radiative_forcing_n2o = 0.12 * (self.n2o_concentration - 270) / 1000  # Approximation

            # Total Radiative Forcing from Greenhouse Gases
            self.total_radiative_forcing = self.radiative_forcing_co2 + self.radiative_forcing_ch4 + self.radiative_forcing_n2o
            
            # Setup GUI first
            self.setup_gui()
            
            # Initialize modules after GUI and data arrays are set up
            self.system_stats = SystemStats(self)
            self.temperature = Temperature(self)
            self.pressure_system = Pressure(self)
            self.wind_system = Wind(self)
            self.precipitation_system = Precipitation(self)
            
            # Load initial data (this will set map dimensions)
            try:
                self.on_load("D:\dev\planetsim\images\GRAY_HR_SR_W_stretched.tif")
                print("Successfully loaded terrain data")
            except Exception as e:
                print(f"Error loading terrain data: {e}")
                # Initialize fallback empty arrays
                self.elevation = np.zeros((self.map_height, self.map_width), dtype=np.float32)
                self.elevation_normalized = np.zeros((self.map_height, self.map_width), dtype=np.float32)
                self.altitude = np.zeros((self.map_height, self.map_width), dtype=np.float32)
                
            # Initialize coordinate arrays
            latitudes = np.linspace(90, -90, self.map_height)
            longitudes = np.linspace(-180, 180, self.map_width)
            self.latitude, self.longitude = np.meshgrid(latitudes, longitudes, indexing='ij')
            self.latitudes_rad = np.deg2rad(self.latitude)
            
            # Initialize basic arrays if they don't already exist
            if not hasattr(self, 'humidity') or self.humidity is None:
                self.humidity = np.full((self.map_height, self.map_width), 0.5, dtype=np.float32)
            if not hasattr(self, 'temperature_celsius') or self.temperature_celsius is None:
                self.temperature_celsius = np.zeros((self.map_height, self.map_width), dtype=np.float32)
            if not hasattr(self, 'pressure') or self.pressure is None:
                self.pressure = np.zeros((self.map_height, self.map_width), dtype=np.float32)
            if not hasattr(self, 'u') or self.u is None:
                self.u = np.zeros((self.map_height, self.map_width), dtype=np.float32)
            if not hasattr(self, 'v') or self.v is None:
                self.v = np.zeros((self.map_height, self.map_width), dtype=np.float32)
            if not hasattr(self, 'ocean_u') or self.ocean_u is None:
                self.ocean_u = np.zeros((self.map_height, self.map_width), dtype=np.float32)
            if not hasattr(self, 'ocean_v') or self.ocean_v is None:
                self.ocean_v = np.zeros((self.map_height, self.map_width), dtype=np.float32)
            if not hasattr(self, 'ocean_temperature') or self.ocean_temperature is None:
                self.ocean_temperature = np.zeros((self.map_height, self.map_width), dtype=np.float32)
            
            # Initialize visualization only after canvas is created in setup_gui
            self.visualization = Visualization(self)
            
            # Now bind events that require visualization
            self.canvas.bind("<Motion>", self.visualization.update_zoom_view)
            
            # Initialize modules in the correct order with error handling
            try:
                print("Initializing temperature module...")
                self.temperature.initialize()
                print("Initializing pressure module...")
                self.pressure_system.initialize()
                print("Initializing wind module...")
                self.wind_system.initialize()
                print("Initializing precipitation module...")
                self.precipitation_system.initialize()
                print("Initializing ocean currents...")
                self.temperature.initialize_ocean_currents()
            except Exception as e:
                print(f"Error during module initialization: {e}")
                traceback.print_exc()
                
            # Initialize threading event for visualization control
            self.visualization_active = threading.Event()
            self.visualization_active.set()  # Start in active state

            # Start visualization in separate thread
            self.visualization_thread = threading.Thread(target=self.visualization.update_visualization_loop)
            self.visualization_thread.daemon = True
            self.visualization_thread.start()

            # Start simulation using Tkinter's after method instead of a thread
            self.root.after(100, self.simulate)
            
            # Create simulation control flags
            self.simulation_active = threading.Event()
            self.simulation_active.set()
            
            # Initialize other attributes
            self.humidity_effect_coefficient = 0.1

            # Flag for ocean diagnostics
            self.ocean_data_available = False
            
            # Optimization: Precompute and cache temperature-dependent values
            self._temp_cache = {}
            self._saturation_vapor_cache = None
            self._temp_range_for_cache = None
            
            # Optimization: Reusable arrays for calculations to minimize memory allocations
            self._reusable_array = None
            self._reusable_array2 = None
            
            # Optimization: Arrays for wind calculations
            self._u_geo = None
            self._v_geo = None
            self._u_new = None
            self._v_new = None
            
            # Optimization: Arrays for temperature calculations
            self._base_albedo = None
            self._solar_in = None
            self._longwave_out = None
            self._net_flux = None
            self._heat_capacity = None
            self._delta_T = None
            
            # Optimization: Arrays for cloud calculations
            self._relative_humidity = None
            self._cloud_coverage = None
            
            # Update the title of the existing window to be more descriptive
            self.root.title("Climate Simulation - Planet Sim")
            
            # Initialize the energy budget dictionary
            self.energy_budget = {
                'incoming': 0,
                'outgoing': 0,
                'greenhouse': 0,
                'albedo': 0,
                'cloud_effect': 0,
                'ocean_flux': 0,
                'ocean_heat_content': 0,
                'ocean_heat_change': 0,
                'evaporation': 0,
                'precipitation': 0,
                'humidity_transport': 0,
                'temperature_history': []  # Initialize temperature history list
            }
            
            # Flag to track if ocean data is available and calculated successfully
            self.ocean_data_available = False
         
            # Initialize wind
            self.wind_system.initialize()

            # Initialize system stats
            self.system_stats = SystemStats(self)

            # Initialize the visualization system after the canvas is created
            # self.canvas = tk.Canvas(self.visualization_frame, bg='black', width=self.map_width, height=self.map_height)
            # self.canvas.pack(fill=tk.BOTH, expand=True)
            # self.visualization = Visualization(self)

            # Create simulation control flags
            self.simulation_active = threading.Event()
            self.visualization_active = threading.Event()
            
            # Master switch for event detection
            self.simulation_running = False

        except Exception as e:
            print(f"Critical error in initialization: {e}")
            traceback.print_exc()
            try:
                messagebox.showerror("Initialization Error", f"Failed to initialize application: {str(e)}\n\nSee console for details.")
                # Attempt to clean up
                if hasattr(self, 'root') and self.root:
                    self.root.destroy()
            except:
                pass  # Just in case the messagebox itself fails
            sys.exit(1)

    def setup_gui(self):
        # Configure the root window to allow resizing
        self.root.resizable(True, True)
        
        # Main frame to contain all controls
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create control frame at top
        control_frame = tk.Frame(main_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Add buttons for file operations
        button_frame = tk.Frame(control_frame)
        button_frame.pack(side=tk.LEFT, padx=5, pady=5)
        
        # File buttons
        new_button = tk.Button(button_frame, text="New", command=self.on_new)
        new_button.pack(side=tk.LEFT, padx=2)
        
        load_button = tk.Button(button_frame, text="Load", command=lambda: self.on_load(None))
        load_button.pack(side=tk.LEFT, padx=2)
        
        # Layer selector
        checkbox_frame = tk.Frame(control_frame)
        checkbox_frame.pack(side=tk.LEFT, padx=10)
        
        # Make radiobuttons for layer selection
        tk.Label(checkbox_frame, text="Layer:").pack(side=tk.LEFT)
        
        layers = [
            "Elevation", "Temperature", "Pressure", "Wind", 
            "Precipitation", "Ocean Temperature"
        ]
        
        for layer in layers:
            rb = tk.Radiobutton(checkbox_frame, text=layer, variable=self.selected_layer, value=layer,
                          command=lambda layer=layer: self.on_layer_change(layer))
            rb.pack(side=tk.LEFT)
        
        # Status label
        self.status_label = tk.Label(control_frame, text="Ready", anchor=tk.W)
        self.status_label.pack(side=tk.RIGHT, padx=5)
        
        # Create frame for visualization
        self.visualization_frame = tk.Frame(main_frame)
        self.visualization_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=False)
        
        # Create canvas with fixed dimensions
        canvas_width = self.map_width if self.map_width is not None else self.map_size
        canvas_height = self.map_height if self.map_height is not None else self.map_size
        
        self.canvas = tk.Canvas(self.visualization_frame, bg='black', 
                                width=canvas_width, height=canvas_height)
        self.canvas.pack(fill=tk.NONE, expand=False)  # Don't allow canvas to resize
        
        # Create mouse-over info label
        self.mouse_over_label = tk.Label(main_frame, text="Mouse over data will appear here", 
                                       anchor=tk.W, justify=tk.LEFT)
        self.mouse_over_label.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=3)
        
        # Create stats display frame
        self.stats_frame = tk.Frame(main_frame, height=100)
        self.stats_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Create zoom dialog
        self.zoom_dialog = ZoomDialog(self.root)
        self.zoom_dialog.withdraw()  # Hide initially
        
        # Bind mouse events that don't require visualization
        self.canvas.bind("<Leave>", lambda e: self.zoom_dialog.withdraw())
        self.canvas.bind("<Enter>", lambda e: self.zoom_dialog.deiconify())
        
        # Motion binding will be added after visualization is created
        
        # Add cleanup on window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Set initial window size
        self._resize_window_to_map()

    def on_layer_change(self, layer):
        """Handle layer change event"""
        try:
            if hasattr(self, 'visualization'):
                self.visualization.update_map()
        except Exception as e:
            print(f"Error changing layer: {e}")
            traceback.print_exc()

    def on_closing(self):
        """Handle window closing"""
        self.cleanup()
        self.root.destroy()

    def on_new(self):
        # [Implement as in the original code...]
        try:
            # Generate new random terrain data and then start simulation
            self.on_generate()

            # Reset to default view
            self.selected_layer.set("Elevation")
            if hasattr(self, 'visualization'):
                self.visualization.update_map()
        except Exception as e:
            print(f"Error in on_new: {e}")
            traceback.print_exc()

    def on_load(self, file_path=None):
        """Load and process elevation image"""
        if file_path:
            try:
                elevation_image = Image.open(file_path)
                self.map_width, self.map_height = elevation_image.size
                
                # Resize window to match the new map dimensions
                self._resize_window_to_map()
                
                # Calculate grid spacing
                earth_circumference = 40075000  # meters
                self.grid_spacing_y = earth_circumference / self.map_height
                self.grid_spacing_x = earth_circumference / self.map_width

                # Convert image to numpy array
                img_array = np.array(elevation_image)
                img_array = img_array.astype(int)
                if len(img_array.shape) > 2:
                    img_array = img_array[:, :, 0]

                # Store raw pixel values
                self.altitude = img_array.copy()

                # Initialize elevation array
                self.elevation = np.zeros_like(img_array, dtype=float)

                # Create masks using pixel value 68 as sea level
                water_mask = img_array <= 68
                land_mask = img_array > 68

                # Calculate land elevations using modified logarithmic scale
                if np.any(land_mask):
                    max_elevation = 8848  # meters (Mount Everest)
                    land_pixels = img_array[land_mask]
                    
                    # Convert pixel values to 0-1 range for land (68 to 254)
                    normalized_pixels = (land_pixels - 68) / (254 - 68)
                    
                    # Apply combined logarithmic and power scaling for elevation distribution
                    log_factor = 0.5  # reduced for even gentler curve
                    power_factor = 2.5  # increased for more emphasis on lower elevations
                    
                    elevation_values = max_elevation * (
                        (np.log(1 + normalized_pixels * log_factor) / np.log(1 + log_factor)) ** power_factor
                    )
                    
                    self.elevation[land_mask] = elevation_values

                # Water tiles are 0m elevation
                self.elevation[water_mask] = 0.0

                # Normalize elevation for display
                elevation_min = self.elevation.min()
                elevation_max = self.elevation.max()
                if elevation_max > elevation_min:
                    self.elevation_normalized = (self.elevation - elevation_min) / (elevation_max - elevation_min)
                else:
                    self.elevation_normalized = np.zeros_like(self.elevation)
            except Exception as e:
                print(f"Error with on_load: {e}")
                raise e

    def _resize_window_to_map(self):
        """Resize the window and canvas to match the map dimensions"""
        try:
            if hasattr(self, 'canvas') and self.map_width is not None and self.map_height is not None:
                # Configure canvas size explicitly
                self.canvas.config(width=self.map_width, height=self.map_height)
                
                # Calculate window size with space for controls
                window_width = self.map_width
                window_height = self.map_height + 120  # Add space for controls
                
                # Set window size directly
                self.root.geometry(f"{window_width}x{window_height}")
                
                # Configure grid weights to ensure canvas gets proper space
                self.root.update_idletasks()
                print(f"Resized window to {window_width}x{window_height} (map: {self.map_width}x{self.map_height})")
        except Exception as e:
            print(f"Error resizing window: {e}")
            traceback.print_exc()

    def on_generate(self, elevation_data=None):
        # Cancel the previous simulation if running
        if hasattr(self, 'simulation_after_id'):
            self.root.after_cancel(self.simulation_after_id)

        # Reset time_step
        self.time_step = 0

        # Generate coordinate arrays
        x = np.linspace(0, 1, self.map_width, endpoint=False)
        y = np.linspace(0, 1, self.map_height, endpoint=False)
        x_coords, y_coords = np.meshgrid(x, y)

        # Grid spacing in meters
        earth_circumference = 40075000
        self.grid_spacing_x = earth_circumference / self.map_width
        self.grid_spacing_y = earth_circumference / (2 * self.map_height)

        # Initialize humidity
        self.humidity = np.full((self.map_height, self.map_width), 50, dtype=np.float32)

        # Convert latitude to radians
        self.latitudes_rad = np.deg2rad(self.latitude)

        # Use fixed values instead of sliders
        self.global_octaves = 6
        self.global_frequency = 2.0
        self.global_lacunarity = 2.0
        self.global_persistence = 0.5

        # Initialize temperature
        self.temperature.initialize()

        # Rest of the on_generate implementation...
        # (Continue with the elevation generation and temperature calculation code)

        # Resize window to match map dimensions
        self._resize_window_to_map()

    def simulate(self):
        """
        Main simulation loop - runs on its own thread.
        
        Called by a timer, processes one step of the simulation and then reschedules itself.
        Updates pressure, temperature, and other systems based on current state.
        """
        try:
            start_time = time.time()
            
            # Increment time step
            self.time_step += 1
            
            # Update physics fields in proper order
            self.precipitation_system.update()  # Update humidity first (depends on prev. temperature)
            
            # Synchronize humidity data with main simulation for compatibility
            self.humidity = self.precipitation_system.humidity
            self.precipitation = self.precipitation_system.precipitation
            self.cloud_cover = self.precipitation_system.cloud_cover
            
            self.temperature.update_land_ocean()      # Update temperatures (depends on humidity)
            self.temperature.update_ocean()  # Update ocean temps (uses updated surface temps)
            self.pressure_system.update()          # Update pressure (affected by temperature)
            self.wind_system.update()              # Finally update winds (affected by all others)
            
            # Clear the pressure image cache after updating pressure to force redraw
            if hasattr(self, '_pressure_image'):
                delattr(self, '_pressure_image')
            # Also clear the cached pressure visualization data to force a fresh redraw
            if hasattr(self, '_cached_pressure_viz'):
                delattr(self, '_cached_pressure_viz')
            if hasattr(self, '_cached_pressure_data'):
                delattr(self, '_cached_pressure_data')
            
            # Schedule GUI updates to happen in the main thread
            self.root.after(0, self.visualization.update_map)
            self.root.after(0, self.system_stats.print_stats)
            self.root.after(0, self.update_mouse_over)
            
            elapsed_time = time.time() - start_time
            # print(f"Cycle {self.time_step} completed in {elapsed_time:.4f} seconds")
            
            # Schedule next simulation step with a reduced or zero delay
            # Option 1: Set a smaller minimum delay
            min_delay = 1  # milliseconds
            delay = max(min_delay, int(elapsed_time * 1000))
            
            # Option 2: Remove min_delay to run as fast as possible
            # delay = max(0, int(elapsed_time * 1000))
            
            self.simulation_after_id = self.root.after(delay, self.simulate)
            
            # Mark simulation as active
            self.simulation_active.set()
            
            # Start visualization in another thread if not already running
            if not hasattr(self, 'visualization_active') or not self.visualization_active.is_set():
                self.visualization_active = threading.Event()
                self.visualization_active.set()
                self.visualization_thread = threading.Thread(target=self.visualization.update_visualization_loop)
                self.visualization_thread.daemon = True
                self.visualization_thread.start()
            
        except Exception as e:
            print(f"Error in simulation: {e}")
            traceback.print_exc()

    def update_mouse_over(self, event=None):
        try:
            if event is None:
                x, y = self.last_mouse_x, self.last_mouse_y
            else:
                x, y = event.x, event.y
                self.last_mouse_x, self.last_mouse_y = x, y

            if 0 <= x < self.map_width and 0 <= y < self.map_height:
                # Retrieve data at mouse position
                pressure_value = self.pressure[y, x]
                temperature_value = self.temperature_celsius[y, x]
                wind_speed_value = np.sqrt(self.u[y, x] ** 2 + self.v[y, x] ** 2)
                elevation_value = self.elevation[y, x]
                latitude_value = self.latitude[y, x]
                longitude_value = self.longitude[y, x]
                grayscale_value = self.altitude[y, x]
                
                # Get precipitation value if available
                precipitation_value = 0
                if hasattr(self, 'precipitation') and self.precipitation is not None:
                    precipitation_value = self.precipitation[y, x]
                
                # Calculate wind direction (direction wind is blowing TO)
                wind_direction_rad = np.arctan2(-self.u[y, x], -self.v[y, x])
                wind_direction_deg = (90 - np.degrees(wind_direction_rad)) % 360

                # Update label text
                self.mouse_over_label.config(text=(
                    f"Pressure: {pressure_value:.2f} Pa, "
                    f"Temperature: {temperature_value:.2f} °C, "
                    f"Wind Speed: {wind_speed_value:.2f} m/s, "
                    f"Wind Direction: {wind_direction_deg:.2f}°, "
                    f"Precipitation: {precipitation_value:.2f} mm/hr\n"
                    f"Latitude: {latitude_value:.2f}°, Longitude: {longitude_value:.2f}°, "
                    f"Elevation: {elevation_value:.2f} m, "
                    f"Y: {y}, X: {x}"
                ))
        except Exception as e:
            print(f"Error in update_mouse_over: {e}")
            # Don't reraise to prevent thread crashes

    def cleanup(self):
        """Clean up threads before closing"""
        self.visualization_active.clear()
        
        if hasattr(self, 'visualization_thread'):
            self.visualization_thread.join(timeout=1.0)

    def toggle_stats_display(self, enabled=None):
        """Toggle or set the system stats display"""
        if enabled is None:
            # Toggle current state
            self.system_stats.print_stats_enabled = not self.system_stats.print_stats_enabled
        else:
            # Set to specified state
            self.system_stats.print_stats_enabled = enabled


class ZoomDialog(tk.Toplevel):
    def __init__(self, parent, zoom_factor=4):
        super().__init__(parent)
        
        # Remove window decorations and make it stay on top
        self.overrideredirect(True)
        self.attributes('-topmost', True)
        
        # Set size for zoomed view (odd numbers ensure center pixel)
        self.view_size = 51  # Reduced from 101 to 51 pixels
        self.zoom_factor = zoom_factor
        
        # Calculate canvas size
        canvas_size = self.view_size * self.zoom_factor
        
        # Create canvas for zoomed view
        self.canvas = tk.Canvas(self, width=canvas_size, height=canvas_size)
        self.canvas.pack()
        
        # Create crosshair in the center
        self.crosshair_size = 10  # Increased size for better visibility
        center = canvas_size // 2
        
        # Create crosshair with multiple components for better visibility
        # Main crosshair in red
        self.canvas.create_line(center - self.crosshair_size, center, 
                              center + self.crosshair_size, center, 
                              fill='red', width=2, tags='crosshair')
        self.canvas.create_line(center, center - self.crosshair_size, 
                              center, center + self.crosshair_size, 
                              fill='red', width=2, tags='crosshair')
        
        # White outline for contrast
        outline_offset = 1
        self.canvas.create_line(center - self.crosshair_size, center - outline_offset, 
                              center + self.crosshair_size, center - outline_offset, 
                              fill='white', width=1, tags='crosshair')
        self.canvas.create_line(center - self.crosshair_size, center + outline_offset, 
                              center + self.crosshair_size, center + outline_offset, 
                              fill='white', width=1, tags='crosshair')
        self.canvas.create_line(center - outline_offset, center - self.crosshair_size, 
                              center - outline_offset, center + self.crosshair_size, 
                              fill='white', width=1, tags='crosshair')
        self.canvas.create_line(center + outline_offset, center - self.crosshair_size, 
                              center + outline_offset, center + self.crosshair_size, 
                              fill='white', width=1, tags='crosshair')


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        # Context already set, continue with current settings
        pass
        
    app = SimulationApp()
    app.root.mainloop()