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
            # Initialize all variables
            self.map_size = 512
            self.map_width = None  # Will be set when image is loaded
            self.map_height = None # Will be set when image is loaded
            
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
            
            # Mouse tracking
            self.last_mouse_x = 0
            self.last_mouse_y = 0
            
            # Seeds
            self.seed = 4200
            self.elevation_seed = self.seed
            
            # Simulation parameters
            self.global_octaves = 5
            self.global_frequency = 2.0
            self.global_lacunarity = 2.0
            self.global_persistence = 0.5
            
            # Initialize arrays
            self.altitude = None
            self.elevation = None
            self.elevation_normalized = None
            self.temperature_celsius = None
            self.previous_temperature_celsius = 0
            self.temperature_normalized = None
            self.pressure = None
            self.pressure_normalized = None

            self.u = None
            self.v = None
            self.ocean_u = None
            self.ocean_v = None

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
            
            # Multiprocessing setup
            self.num_processes = multiprocessing.cpu_count() - 20
            if self.num_processes < 1:
                self.num_processes = 1
            print(f'Number of CPU cores available: {self.num_processes}')
            
            # Create Tkinter window
            self.root = tk.Tk()
            self.root.title("Climate Simulation - Planet Sim")
            
            # Variable to track the selected layer
            self.selected_layer = tk.StringVar(value="Wind")
            
            # Load initial data (this will set map dimensions)
            # self.on_load("D:\\dev\\planetsim\\images\\1024x512_earth_8bit.png")
            self.on_load("D:\dev\planetsim\images\GRAY_HR_SR_W_stretched.tif")
            
            # Initialize modular systems
            self.temperature = Temperature(self)
            self.pressure_system = Pressure(self)
            self.wind_system = Wind(self)
            self.precipitation_system = Precipitation(self)
            self.system_stats = SystemStats(self)
            
            # Setup GUI first
            self.setup_gui()
            
            # Initialize visualization only after canvas is created in setup_gui
            self.visualization = Visualization(self)

            # Initialize coordinate arrays first
            latitudes = np.linspace(90, -90, self.map_height)
            longitudes = np.linspace(-180, 180, self.map_width)
            self.latitude, self.longitude = np.meshgrid(latitudes, longitudes, indexing='ij')
            self.latitudes_rad = np.deg2rad(self.latitude)

            # Initialize basic humidity field (will be updated later)
            self.humidity = np.full((self.map_height, self.map_width), 0.5, dtype=np.float32)

            # Initialize temperature array
            self.temperature_celsius = np.zeros((self.map_height, self.map_width), dtype=np.float32)
            
            # Initialize temperature module
            self.temperature.initialize()

            # Initialize temperature
            self.temperature.initialize()
            
            # Initialize pressure module
            self.pressure_system.initialize()
            
            # Initialize wind module
            self.wind_system.initialize()

            # Initialize humidity
            self.precipitation_system.initialize()

            # Initialize other attributes
            self.humidity_effect_coefficient = 0.1

            # Initialize wind via the wind system
            self.wind_system.initialize()

            # Initialize ocean currents
            self.temperature.initialize_ocean_currents()

            # Initialize threading event for visualization control
            self.visualization_active = threading.Event()
            self.visualization_active.set()  # Start in active state

            # Start visualization in separate thread
            self.visualization_thread = threading.Thread(target=self.visualization.update_visualization_loop)
            self.visualization_thread.daemon = True
            self.visualization_thread.start()

            # Start simulation using Tkinter's after method instead of a thread
            # This ensures all Tkinter interactions happen in the main thread
            self.root.after(100, self.simulate)

            # Trigger initial map update
            if hasattr(self, 'visualization'):
                self.visualization.update_map()
            
            # Add this near the end of your __init__ method
            # Flag for ocean diagnostics
            self.ocean_data_available = False
            
            # Remove duplicated initialization
            # self.energy_budget = {}
            
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
            
            # Create a Tkinter root window - ALREADY CREATED ABOVE, DON'T CREATE A SECOND ONE
            # self.root = tk.Tk()
            # self.root.title("Climate Simulation")
            
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
            print(f"Error in initialization: {e}")
            traceback.print_exc()

    def setup_gui(self):
        # Configure the root window to prevent resizing
        self.root.resizable(False, False)
        
        # Calculate window dimensions
        # Add buffer for UI elements: 60px for controls and 30px for mouse_over label
        ui_buffer = 90
        window_width = self.map_width
        window_height = self.map_height + ui_buffer
        
        # Set window geometry
        self.root.geometry(f'{window_width}x{window_height}')
        
        # Create the menu bar
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="New", command=self.on_new)
        file_menu.add_command(label="Open", command=self.on_load)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)

        # Create main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(expand=False, fill=tk.BOTH)

        # Initialize the canvas with exact map dimensions
        self.canvas = tk.Canvas(main_frame, 
                            width=self.map_width,
                            height=self.map_height)
        self.canvas.pack(expand=False)

        # Frame for controls
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill=tk.X)
        
        # Create a frame for radio buttons
        checkbox_frame = tk.Frame(self.root)
        checkbox_frame.pack(anchor='w')

        # Add radio buttons for each map layer horizontally
        tk.Radiobutton(checkbox_frame, text="Elevation", variable=self.selected_layer, value="Elevation",
                    command=lambda: self.visualization.update_map() if hasattr(self, 'visualization') else None).pack(side=tk.LEFT)
        tk.Radiobutton(checkbox_frame, text="Altitude", variable=self.selected_layer, value="Altitude", 
                    command=lambda: self.visualization.update_map() if hasattr(self, 'visualization') else None).pack(side=tk.LEFT)
        tk.Radiobutton(checkbox_frame, text="Temperature", variable=self.selected_layer, value="Temperature",
                    command=lambda: self.visualization.update_map() if hasattr(self, 'visualization') else None).pack(side=tk.LEFT)
        tk.Radiobutton(checkbox_frame, text="Wind", variable=self.selected_layer, value="Wind", 
                    command=lambda: self.visualization.update_map() if hasattr(self, 'visualization') else None).pack(side=tk.LEFT)
        tk.Radiobutton(checkbox_frame, text="Ocean Temperature", variable=self.selected_layer, value="Ocean Temperature",
                    command=lambda: self.visualization.update_map() if hasattr(self, 'visualization') else None).pack(side=tk.LEFT)
        tk.Radiobutton(checkbox_frame, text="Precipitation", variable=self.selected_layer, value="Precipitation",
                    command=lambda: self.visualization.update_map() if hasattr(self, 'visualization') else None).pack(side=tk.LEFT)

        # Add new radio button for pressure map
        tk.Radiobutton(
            control_frame,
            text="Pressure",
            variable=self.selected_layer,
            value="Pressure",
            command=lambda: self.visualization.update_map() if hasattr(self, 'visualization') else None
        ).pack(anchor=tk.W)

        # Label at the bottom to display the current values at the mouse position
        self.mouse_over_label = tk.Label(self.root, text="Pressure: --, Temperature: --, Wind Speed: --, Wind Direction: --")
        self.mouse_over_label.pack(anchor="w")

        # Bind the mouse motion event to the canvas
        self.canvas.bind("<Motion>", self.update_mouse_over)

        # Initialize zoom dialog
        self.zoom_dialog = ZoomDialog(self.root)
        self.zoom_dialog.withdraw()  # Hide initially
        
        # Bind mouse events
        self.canvas.bind("<Motion>", self.update_zoom_view)
        self.canvas.bind("<Leave>", lambda e: self.zoom_dialog.withdraw())
        self.canvas.bind("<Enter>", lambda e: self.zoom_dialog.deiconify())

        # Add cleanup on window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

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
            self.update_humidity()           # Update humidity first (depends on prev. temperature)
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

    def update_zoom_view(self, event):
        """Update the zoom window display when the user moves the mouse over the main map"""
        try:
            if not self.zoom_dialog or not hasattr(self, 'zoom_dialog') or not self.zoom_dialog.winfo_exists():
                return
                
            # Get cursor position
            x, y = event.x, event.y
            
            # Calculate the zoom window dimensions
            zoom_factor = self.zoom_dialog.zoom_factor
            # Use the zoom dialog's canvas width and height instead of undefined attributes
            zoom_canvas_width = self.zoom_dialog.canvas.winfo_width()
            zoom_canvas_height = self.zoom_dialog.canvas.winfo_height()
            half_width = int(zoom_canvas_width / (2 * zoom_factor))
            half_height = int(zoom_canvas_height / (2 * zoom_factor))
            
            # Calculate the region to extract, ensuring we don't go out of bounds
            x_start = max(0, x - half_width)
            x_end = min(self.map_width, x + half_width)
            y_start = max(0, y - half_height)
            y_end = min(self.map_height, y + half_height)
            
            # Extract the region from the appropriate data based on selected layer
            layer = self.selected_layer.get()
            
            if layer == "Elevation (Grayscale)":
                view_data = self.visualization.map_to_grayscale(self.elevation_normalized)[y_start:y_end, x_start:x_end]
            elif layer == "Terrain":
                view_data = self.visualization.map_altitude_to_color(self.elevation)[y_start:y_end, x_start:x_end]
            elif layer == "Temperature":
                view_data = self.visualization.map_temperature_to_color(self.temperature_celsius)[y_start:y_end, x_start:x_end]
            elif layer == "Pressure":
                view_data = self.visualization.map_pressure_to_color(self.pressure)[y_start:y_end, x_start:x_end]
            elif layer == "Rain Shadow":
                view_data = self.visualization.map_altitude_to_color(self.elevation)[y_start:y_end, x_start:x_end]
            elif layer == "Biomes":
                # Implement biome visualization using real biome data when available
                # For now, just show terrain
                if hasattr(self, 'biomes') and self.biomes is not None:
                    # Implement biome color mapping when available
                    view_data = self.visualization.map_altitude_to_color(self.elevation)[y_start:y_end, x_start:x_end]
                else:
                    view_data = self.visualization.map_altitude_to_color(self.elevation)[y_start:y_end, x_start:x_end]
            elif layer == "Ocean Temperature":
                # Normalize ocean temps to 0-1 range for better visualization
                ocean_temp_normalized = MapGenerator.normalize_data(self.ocean_temperature)
                view_data = self.visualization.map_ocean_temperature_to_color(ocean_temp_normalized)[y_start:y_end, x_start:x_end]
            elif layer == "Precipitation":
                view_data = self.visualization.map_precipitation_to_color(self.precipitation)[y_start:y_end, x_start:x_end]
            elif layer == "Ocean Currents":
                # For ocean currents, still use elevation as background
                view_data = self.visualization.map_altitude_to_color(self.elevation)[y_start:y_end, x_start:x_end]
            else:
                # Default to terrain
                view_data = self.visualization.map_altitude_to_color(self.elevation)[y_start:y_end, x_start:x_end]
            
            # Create zoomed image
            img = Image.fromarray(view_data)
            img = img.resize((self.zoom_dialog.view_size * self.zoom_dialog.zoom_factor,) * 2, 
                            Image.Resampling.NEAREST)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            self.zoom_dialog.canvas.delete("image")  # Only delete image, keep crosshair
            self.zoom_dialog.canvas.create_image(0, 0, anchor="nw", image=photo, tags="image")
            self.zoom_dialog.canvas.tag_raise('crosshair')  # Ensure crosshair stays on top
            self.zoom_dialog.image = photo  # Keep reference
            
            # Position dialog near cursor but not under it
            dialog_x = self.root.winfo_rootx() + event.x + 20
            dialog_y = self.root.winfo_rooty() + event.y + 20
            
            # Ensure dialog stays within screen bounds
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            dialog_width = self.zoom_dialog.view_size * self.zoom_dialog.zoom_factor
            dialog_height = dialog_width
            
            if dialog_x + dialog_width > screen_width:
                dialog_x = event.x - dialog_width - 20
            if dialog_y + dialog_height > screen_height:
                dialog_y = event.y - dialog_height - 20
            
            self.zoom_dialog.geometry(f"+{dialog_x}+{dialog_y}")
            
            # Update mouse over info
            self.update_mouse_over(event)
            
        except Exception as e:
            print(f"Error updating zoom view: {e}")
            print(f"Current layer: {self.selected_layer.get()}")


    def update_humidity(self):
        """
        Update precipitation system and synchronize humidity data with main simulation.
        This method is maintained for compatibility with code that expects humidity,
        precipitation, and cloud_cover attributes directly on the SimulationApp instance.
        """
        # Run the precipitation system update
        self.precipitation_system.update()
        
        # Synchronize data between the precipitation system and main simulation attributes
        self.humidity = self.precipitation_system.humidity
        self.precipitation = self.precipitation_system.precipitation
        self.cloud_cover = self.precipitation_system.cloud_cover

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