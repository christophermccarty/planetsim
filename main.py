import sys
import time
import numpy as np
import tkinter as tk
import threading
import multiprocessing
import traceback
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from scipy.ndimage import gaussian_filter
import random
import queue

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
            print("Starting SimulationApp initialization...")
            # Create the root window first
            self.root = tk.Tk()
            self.root.title("Climate Simulation - Planet Sim")
            print("Created root window")
            
            # Initialize all variables
            self.map_size = 512
            self.map_width = self.map_size  # Set default width until image is loaded
            self.map_height = self.map_size # Set default height until image is loaded
            
            # Add show_stats variable to control system stats display
            self.show_stats = True  # Default to showing stats
            
            # Variable to track the selected layer - this needs to exist before setup_gui
            self.selected_layer = tk.StringVar(value="Elevation")
            
            # Mouse tracking
            self.last_mouse_x = 0
            self.last_mouse_y = 0
            self.last_mouse_update_time = 0  # Track when we last updated mouse info
            self.mouse_position_changed = False  # Flag to track if mouse has moved since last update
            self.mouse_debounce_delay = 50  # Milliseconds to wait before updating after mouse movement
            
            # Time and simulation variables
            self.time_step = 0
            self.time_step_seconds = 60 * 60  # Simulation time step in seconds (1 hour)
            self.minimum_simulation_hours = 24  # Minimum number of hours to simulate per cycle
            self.target_update_rate = 1.0  # Target updates per second
            self.min_simulation_hours_per_update = 1.0  # Minimum hours to simulate per update
            self.max_simulation_hours_per_update = 24.0  # Maximum hours to simulate per update
            self.high_speed_mode = False  # Disable high-speed physics approximation by default
            self.physics_downsampling = 4  # Spatial downsampling factor for faster physics
            self._last_step_time = 0.1  # Initialize with a reasonable default value
            self.Omega = 7.2921159e-5
            self.P0 = 101325
            self.desired_simulation_step_time = 0.1  # 0.1 seconds between simulation steps

            # Create a queue for visualization updates
            self.visualization_queue = queue.Queue(maxsize=10)  # Limit queue size to prevent memory issues
            
            # Create a dedicated simulation thread event to control simulation
            self.simulation_thread_active = threading.Event()
            self.simulation_thread_active.set()
            
            # Initialize state flag
            self.simulation_running = False  # Start with simulation off
            
            # Define visualization update priority levels
            self.VIZ_PRIORITY = {
                "HIGH": 0,     # Critical updates (layer changes, initial display)
                "MEDIUM": 1,   # Important updates (simulation step completed)
                "LOW": 2       # Less important updates (mouse movements, small data changes)
            }

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
            print("Basic variables initialized")
            
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
            print("Climate parameters initialized")
            
            # Setup GUI first
            print("Setting up GUI...")
            self.setup_gui()
            print("GUI setup completed")
            
            # Initialize modules after GUI and data arrays are set up
            print("Initializing system modules...")
            self.system_stats = SystemStats(self)
            self.temperature = Temperature(self)
            self.pressure_system = Pressure(self)
            self.wind_system = Wind(self)
            self.precipitation_system = Precipitation(self)
            print("System modules initialized")
            
            # Load initial data (this will set map dimensions)
            print("Loading initial data...")
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
            print("Arrays initialized")
            
            # Initialize visualization only after canvas is created in setup_gui
            print("Initializing visualization...")
            self.visualization = Visualization(self)
            print("Visualization initialized")
            
            # Now bind events that require visualization
            self.canvas.bind("<Motion>", self.on_mouse_move)
            
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
                print("All modules initialized successfully")
            except Exception as e:
                print(f"Error initializing modules: {e}")
                traceback.print_exc()
                # Don't exit, let's try to continue with defaults
                
            # Initialize visualization thread
            print("Setting up visualization thread...")
            self.visualization_thread = threading.Thread(target=self.visualization.update_visualization_loop)
            self.visualization_thread.daemon = True
            self.visualization_thread.start()
            
            # Initialize the simulation thread but don't start it yet
            print("Setting up simulation thread...")
            self.simulation_thread = threading.Thread(target=self.simulation_worker)
            self.simulation_thread.daemon = True
            self.simulation_thread.start()
            
            # Create simulation control flags
            self.simulation_active = threading.Event()
            self.simulation_active.set()
            self.visualization_active = threading.Event()
            self.visualization_active.set()
            self.simulation_running = False  # Start with simulation off
            
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
            self.ocean_data_available = True
         
            # Initialize wind
            self.wind_system.initialize()

            # Initialize system stats
            self.system_stats = SystemStats(self)

            # Initialize the visualization system after the canvas is created
            # self.canvas = tk.Canvas(self.visualization_frame, bg='black', width=self.map_width, height=self.map_height)
            # self.canvas.pack(fill=tk.BOTH, expand=True)
            # self.visualization = Visualization(self)

            # Stability tracking - added for better monitoring
            self.stability_history = {
                'fixed_counts': [],  # Track how many values we've had to fix
                'severe_counts': [],  # Track severe instability occurrences
                'last_reset_time': 0,  # Track when we last reset the simulation
                'stable_cycles': 0,    # Count consecutive stable cycles
                'unstable_cycles': 0   # Count consecutive unstable cycles
            }
            
            print("SimulationApp initialization complete")

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
        self.root.geometry(f"{self.map_width}x{self.map_height+120}")  # Extra height for controls
        self.root.resizable(True, True)
        self.root.configure(bg='#1E1E1E')
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Create a main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create a frame for the top controls
        self.controls_frame = ttk.Frame(self.main_frame)
        self.controls_frame.pack(fill=tk.X, side=tk.TOP, pady=(0, 10))
        
        # Create a frame for the elevation map
        self.map_frame = ttk.Frame(self.main_frame)
        self.map_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add the canvas for the map display
        self.canvas = tk.Canvas(self.map_frame, width=self.map_width, height=self.map_height, bg='black', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Create buttons for file operations
        file_frame = ttk.Frame(self.controls_frame)
        file_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        # New and load buttons
        self.new_button = ttk.Button(file_frame, text="New", command=self.on_new)
        self.new_button.pack(side=tk.LEFT, padx=2)
        
        self.load_button = ttk.Button(file_frame, text="Load", command=lambda: self.on_load())
        self.load_button.pack(side=tk.LEFT, padx=2)
        
        # Add layer selection combo
        layer_frame = ttk.Frame(self.controls_frame)
        layer_frame.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(layer_frame, text="Layer:").pack(side=tk.LEFT)
        layers = ["Elevation", "Temperature", "Pressure", "Wind", "Ocean Temperature", "Precipitation", "Ocean Currents", "Humidity", "Cloud Cover"]
        self.layer_combo = ttk.Combobox(layer_frame, textvariable=self.selected_layer, values=layers, width=15, state="readonly")
        self.layer_combo.current(0)
        self.layer_combo.pack(side=tk.LEFT, padx=5)
        self.layer_combo.bind("<<ComboboxSelected>>", lambda e: self.on_layer_change(self.selected_layer.get()))
        
        # Add simulation control frame
        sim_frame = ttk.Frame(self.controls_frame)
        sim_frame.pack(side=tk.RIGHT, padx=10)
        
        # Simulate button
        self.sim_button = ttk.Button(sim_frame, text="Start", command=self.on_generate)
        self.sim_button.pack(side=tk.RIGHT, padx=2)
        
        # Add performance optimization frame
        perf_frame = ttk.Frame(self.controls_frame)
        perf_frame.pack(side=tk.RIGHT, padx=10)
        
        # High-speed approximation toggle
        self.high_speed_var = tk.BooleanVar(value=self.high_speed_mode)
        self.high_speed_check = ttk.Checkbutton(
            perf_frame, 
            text="High-Speed Mode", 
            variable=self.high_speed_var,
            command=self.toggle_high_speed_mode
        )
        self.high_speed_check.pack(side=tk.RIGHT, padx=5)
        
        # Stats display toggle
        self.stats_var = tk.BooleanVar(value=True)  # Use True as default value
        self.stats_check = ttk.Checkbutton(
            perf_frame, 
            text="Show Stats", 
            variable=self.stats_var,
            command=lambda: self.toggle_stats_display(self.stats_var.get())
        )
        self.stats_check.pack(side=tk.RIGHT, padx=5)
        
        # Initialize zoom window state
        self.zoom_enabled = tk.BooleanVar(value=False)
        
        # Add zoom toggle button
        zoom_button = ttk.Checkbutton(
            perf_frame,
            text="Zoom Window",
            variable=self.zoom_enabled,
            command=self.toggle_zoom_window
        )
        zoom_button.pack(side=tk.RIGHT, padx=5)
        
        # Initialize zoom window as None
        self.zoom_dialog = None
        
        # Create bottom status bar frame
        self.status_frame = tk.Frame(self.root, height=30, bg="#111")
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Add mouse over information label
        self.mouse_over_label = tk.Label(
            self.status_frame, 
            text="Move mouse over map for cell information",
            anchor=tk.W,
            background="#111",
            foreground="#FFF",
            font=("Arial", 10),
            padx=10,
            pady=5,
            width=150  # Increased width to accommodate more information
        )
        self.mouse_over_label.pack(side=tk.LEFT, anchor=tk.W, fill=tk.X, expand=True)
        
        # Start periodic mouse-over updates
        self._start_mouse_info_updates()

    def _start_mouse_info_updates(self):
        """Start periodic updates for mouse-over information"""
        # Update mouse-over info every 200ms even without mouse movement
        self.update_mouse_over()  # Initial update
        self.root.after(500, self._periodic_mouse_update)  # Changed from 200ms to 500ms
    
    def _periodic_mouse_update(self):
        """Update mouse-over information periodically"""
        try:
            # Only update if significant time has passed or mouse has moved
            current_time = time.time() * 1000  # Convert to milliseconds
            time_since_update = current_time - self.last_mouse_update_time
            
            if self.mouse_position_changed or time_since_update > 1000:  # Force update after 1 second
                self.update_mouse_over()
                self.mouse_position_changed = False
            
            # Schedule the next update with variable frequency
            # More frequent checks if we're actively moving the mouse, less frequent otherwise
            if self.mouse_position_changed:
                next_update = 200  # More responsive when moving
            else:
                next_update = 500  # Less frequent when idle
                
            self.root.after(next_update, self._periodic_mouse_update)
        except Exception as e:
            print(f"Error in periodic mouse update: {e}")
            # Try to reschedule despite error
            self.root.after(1000, self._periodic_mouse_update)

    def on_layer_change(self, layer):
        """Handle layer change event"""
        try:
            if hasattr(self, 'visualization'):
                # Layer changes are high priority
                self.request_visualization_update(self.VIZ_PRIORITY["HIGH"])
        except Exception as e:
            print(f"Error changing layer: {e}")
            traceback.print_exc()
    
    def request_visualization_update(self, priority=None):
        """Request a visualization update via the queue with optional priority"""
        try:
            if hasattr(self, 'visualization_queue'):
                # Default to MEDIUM priority if none specified
                if priority is None:
                    priority = self.VIZ_PRIORITY["MEDIUM"]
                
                # Create update request with priority
                update_request = {
                    "type": "UPDATE",
                    "priority": priority,
                    "timestamp": time.time()
                }
                
                if not self.visualization_queue.full():
                    self.visualization_queue.put_nowait(update_request)
                else:
                    # If queue is full, try to remove a lower priority item
                    try:
                        # Check if we can remove a lower priority item
                        queue_items = []
                        found_lower_priority = False
                        
                        # Get all items from queue temporarily
                        while not self.visualization_queue.empty():
                            item = self.visualization_queue.get_nowait()
                            queue_items.append(item)
                            
                        # Look for a lower priority item to replace
                        for i, item in enumerate(queue_items):
                            # If item has no priority field or has lower priority
                            if not isinstance(item, dict) or "priority" not in item or item["priority"] > priority:
                                # Replace with our higher priority item
                                queue_items[i] = update_request
                                found_lower_priority = True
                                break
                        
                        # If we couldn't find a lower priority item and this is HIGH priority
                        if not found_lower_priority and priority == self.VIZ_PRIORITY["HIGH"]:
                            # Replace the oldest item
                            oldest_index = 0
                            oldest_time = float('inf')
                            
                            for i, item in enumerate(queue_items):
                                if isinstance(item, dict) and "timestamp" in item and item["timestamp"] < oldest_time:
                                    oldest_time = item["timestamp"]
                                    oldest_index = i
                            
                            # Replace the oldest item with our high priority update
                            queue_items[oldest_index] = update_request
                        
                        # Put items back in the queue
                        for item in queue_items:
                            self.visualization_queue.put_nowait(item)
                            
                    except Exception as e:
                        print(f"Error managing visualization queue: {e}")
                        # In case of error, discard all and add the new one
                        while not self.visualization_queue.empty():
                            try:
                                self.visualization_queue.get_nowait()
                            except:
                                pass
                        self.visualization_queue.put_nowait(update_request)
        except Exception as e:
            print(f"Error requesting visualization update: {e}")
            # Fall back to direct update in case of queue errors
            if hasattr(self, 'visualization'):
                self.root.after(0, self.visualization.update_map)

    def on_closing(self):
        """Handle window closing"""
        self.cleanup()
        self.root.destroy()
        
    def toggle_zoom_window(self):
        """Toggle the zoom window on/off"""
        if self.zoom_enabled.get():
            # Create zoom window if it doesn't exist
            if not hasattr(self, 'zoom_dialog') or self.zoom_dialog is None or not self.zoom_dialog.winfo_exists():
                self.zoom_dialog = ZoomDialog(self.root)
                # Position the zoom dialog in the bottom right initially
                self.zoom_dialog.geometry(f"+{self.root.winfo_rootx() + self.map_width - 250}+{self.root.winfo_rooty() + self.map_height - 250}")
                # Request a visualization update to show the zoom view - medium priority
                self.request_visualization_update(self.VIZ_PRIORITY["MEDIUM"])
        else:
            # Destroy zoom window if it exists
            if hasattr(self, 'zoom_dialog') and self.zoom_dialog and self.zoom_dialog.winfo_exists():
                self.zoom_dialog.destroy()
                self.zoom_dialog = None

    def on_new(self):
        try:
            # Generate new random terrain data and then start simulation
            self.on_generate()

            # Reset to default view
            self.selected_layer.set("Elevation")
            if hasattr(self, 'visualization'):
                # High priority for initial display
                self.request_visualization_update(self.VIZ_PRIORITY["HIGH"])
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
        """Handle generating a simulation or starting a new simulation"""
        
        # Don't restart if already running
        if hasattr(self, 'simulation_running') and self.simulation_running:
            print("Simulation is already running")
            return
        
        try:
            print("on_generate() method called")
            
            # If elevation_data is provided, use it directly
            if elevation_data is not None:
                self.elevation = elevation_data
                print("Using provided elevation data")
            
            # Initialize key systems
            print("Initializing systems...")
            
            # Initialize temperature
            if not hasattr(self, 'temperature_system') or self.temperature_system is None:
                from temperature import Temperature
                self.temperature_system = Temperature(self)
                
            self.temperature_system.initialize()
            
            # Initialize pressure
            if not hasattr(self, 'pressure_system') or self.pressure_system is None:
                from pressure import Pressure
                self.pressure_system = Pressure(self)
            
            print("Initializing pressure...")
            self.pressure_system.initialize()
            
            # Initialize winds
            if not hasattr(self, 'wind_system') or self.wind_system is None:
                from wind import Wind
                self.wind_system = Wind(self)
                
            print("Initializing winds...")
            self.wind_system.initialize()
            
            # Initialize precipitation
            if not hasattr(self, 'precipitation_system') or self.precipitation_system is None:
                from precipitation import Precipitation
                self.precipitation_system = Precipitation(self)
                
            print("Initializing precipitation...")
            self.precipitation_system.initialize()
            
            # Initialize ocean currents
            if not hasattr(self, 'temperature_system') or self.temperature_system is None:
                from temperature import Temperature
                self.temperature_system = Temperature(self)
                
            print("Initializing ocean currents...")
            self.temperature_system.initialize_ocean_currents()
            
            # Initialize energy budget if not already done
            if not hasattr(self, 'energy_budget'):
                self.energy_budget = {}
            
            # Setup system stats if not already done
            if not hasattr(self, 'system_stats') or self.system_stats is None:
                from system_stats import SystemStats
                self.system_stats = SystemStats(self)
                # Initialize the system stats with the current show_stats setting
                self.system_stats.print_stats_enabled = self.show_stats
            
            # Ensure stats display is enabled if checkbox is checked
            if hasattr(self, 'show_stats_var') and self.show_stats_var.get():
                self.show_stats = True
                self.system_stats.print_stats_enabled = True
                # Force an immediate stats display update
                self.system_stats.force_next_update = True
                self.system_stats.print_stats()
            else:
                # Make sure system_stats respects the current setting
                self.system_stats.print_stats_enabled = self.show_stats
            
            # Configure UI elements for a running simulation
            print("Configuring UI elements...")
            
            # Disable generate/new/load during simulation
            if hasattr(self, 'generate_button'):
                self.generate_button.config(state=tk.DISABLED)
            if hasattr(self, 'new_button'):
                self.new_button.config(state=tk.DISABLED)
            if hasattr(self, 'load_button'):
                self.load_button.config(state=tk.DISABLED)
                
            # Change sim_button to Stop button instead of looking for stop_button
            if hasattr(self, 'sim_button'):
                self.sim_button.config(text="Stop", command=self.stop_simulation)
            else:
                print("Warning: sim_button not found!")
                
            # Mark simulation as running
            self.simulation_running = True
            print("Setting simulation_running = True")
            
            # Signal the simulation thread to process
            if hasattr(self, 'simulation_thread_active'):
                self.simulation_thread_active.set()
            
            print("Simulation started")
            
            # Schedule first simulation step
            print("Scheduling first simulation step")
            self.time_step = 0
            
            if hasattr(self, 'root'):
                self.root.after(10, self.simulate)
                print("First simulation step scheduled")
            
            # Force an immediate stats display update if stats are enabled
            if self.show_stats:
                self.system_stats.print_stats()
        except Exception as e:
            print(f"Error in on_generate: {e}")
            traceback.print_exc()

    def stop_simulation(self):
        """Stop the running simulation"""
        # Set flag to stop simulation
        self.simulation_running = False
        
        # Change button text back to "Start"
        self.sim_button.config(text="Start", command=self.on_generate)
        
        print("Simulation stopped")

    def simulation_worker(self):
        """Background thread for running simulation computations"""
        print("Simulation worker thread started")
        
        while True:
            try:
                # Check if we should stop the thread
                if not self.simulation_thread_active.is_set():
                    print("Simulation thread stopping (flag cleared)")
                    break
                
                # Check if simulation is running
                if not self.simulation_running:
                    # Sleep briefly and check again
                    time.sleep(0.1)
                    continue
                
                # Get start time for this cycle
                cycle_start = time.time()
                
                # Update current step
                self.time_step += 1
                
                # Determine simulation time based on time since last update
                current_time = time.time()
                
                # If we're just starting or _last_step_time doesn't exist yet, use a safe default
                if not hasattr(self, '_last_step_time') or self._last_step_time <= 0:
                    self._last_step_time = 0.1  # Use a reasonable default value
                    
                # Calculate simulation speed
                target_seconds_per_update = 1.0 / self.target_update_rate
                
                # Default steps_needed value (will be updated for standard mode)
                steps_needed = 1
                
                # Check for mode transition to handle it gracefully
                transitioning = hasattr(self, '_transitioning_speed_mode') and self._transitioning_speed_mode
                
                # Check if we should use high-speed approximation or standard simulation
                if self.high_speed_mode:
                    # During transition to high-speed, use a more conservative approach
                    if transitioning:
                        approximation_factor = 2  # Start with a lower factor during transition
                    else:
                        approximation_factor = 6  # Normal high-speed operation
                    
                    # In high-speed mode, we run with a fixed approximation factor
                    # This gives more consistent visual updates but less precise physics
                    target_hours = self.time_step_seconds * approximation_factor / 3600
                    self._run_approximated_simulation(target_hours)
                    self.simulation_speed = target_hours
                    
                    # For timing calculations in high-speed mode
                    steps_needed = approximation_factor
                else:
                    # In standard mode, we adjust simulation rate based on actual performance
                    # Calculate number of steps to run to hit our target update rate
                    simulation_fps = 1.0 / max(self._last_step_time, 0.001)
                    
                    # Calculate how many steps needed to maintain target rate
                    steps_needed = max(1, int(self.target_update_rate / simulation_fps))
                    
                    # Limit the maximum steps to prevent excessive lag during slowdowns
                    steps_needed = min(steps_needed, 24)  # Cap at 24 steps (1 sim day)
                    
                    # Run standard simulation with calculated steps
                    self._run_standard_simulation(steps_needed)
                    
                    # Calculate simulation speed in simulated hours per real time second 
                    self.simulation_speed = steps_needed * self.time_step_seconds / 3600
                
                # Request visualization update
                try:
                    # Use non-blocking put_nowait to prevent simulation thread from waiting
                    if hasattr(self, 'visualization_queue'):
                        if not self.visualization_queue.full():
                            # Simple string message for compatibility with both versions
                            self.visualization_queue.put_nowait("UPDATE")
                except:
                    # Queue is full or error, skip this update
                    pass
                
                # Schedule UI updates in the main thread using after()
                # This prevents UI blocking and ensures UI updates happen in the UI thread
                if hasattr(self, 'root') and self.root.winfo_exists():
                    self.root.after(0, self.update_ui_elements)
                
                # Measure total elapsed time
                elapsed_time = time.time() - cycle_start
                
                # Calculate sleep time to maintain target rate
                sleep_time = max(0, target_seconds_per_update - elapsed_time)
                
                # Sleep for calculated time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # Update _last_step_time for next cycle's timing calculation
                self._last_step_time = elapsed_time / max(1, steps_needed)
                
            except Exception as e:
                print(f"Error in simulation worker: {e}")
                traceback.print_exc()
                # Sleep briefly to prevent CPU spinning on errors
                time.sleep(0.5)
    
    def update_ui_elements(self):
        """Update UI elements in the main thread"""
        try:
            # Update stats display state from checkbox
            if hasattr(self, 'stats_var'):
                # Get current setting from checkbox
                current_stats_state = self.stats_var.get()
                
                # Update the show_stats value
                self.show_stats = current_stats_state
                
                if hasattr(self, 'system_stats'):
                    self.system_stats.print_stats_enabled = self.show_stats
            
            # Force periodic stats updates
            if hasattr(self, 'system_stats') and self.time_step % 5 == 0:
                self.system_stats.force_next_update = True
                self.system_stats.print_stats()
                
        except Exception as e:
            print(f"Error updating UI elements: {e}")
    

    def _run_standard_simulation(self, steps_needed):
        """Run standard simulation for the specified number of steps"""
        # Track time for performance measurement
        step_start_time = time.time()
        
        # Run multiple simulation steps to reach the target time
        for _ in range(steps_needed):
            # Increment time step
            self.time_step += 1
            
            # Update physics fields in proper order
            self.precipitation_system.update()
            self.humidity = self.precipitation_system.humidity
            self.precipitation = self.precipitation_system.precipitation
            self.cloud_cover = self.precipitation_system.cloud_cover
            
            self.temperature.update_land_ocean()
            self.temperature.update_ocean()
            self.pressure_system.update()
            self.wind_system.update()
        
        # Measure the average time per step
        step_end_time = time.time()
        total_step_time = step_end_time - step_start_time
        measured_step_time = total_step_time / steps_needed
        
        # Use exponential moving average for smoother timing adjustments
        if not hasattr(self, '_last_step_time'):
            self._last_step_time = measured_step_time
        else:
            # EMA with alpha=0.3 gives reasonable stability while allowing adjustments
            alpha = 0.3
            self._last_step_time = alpha * measured_step_time + (1 - alpha) * self._last_step_time
    
    def _run_approximated_simulation(self, target_hours):
        """Run an approximated high-speed simulation to achieve the target hours"""
        step_start_time = time.time()
        
        # Calculate total simulation seconds
        total_seconds = target_hours * 3600
        
        # Ensure all critical attributes exist to prevent errors
        if not hasattr(self, 'cloud_cover') or self.cloud_cover is None:
            print("Initializing missing cloud_cover attribute")
            self.cloud_cover = np.zeros((self.map_height, self.map_width), dtype=np.float32)
            
        if not hasattr(self, 'humidity') or self.humidity is None:
            print("Initializing missing humidity attribute")
            self.humidity = np.ones((self.map_height, self.map_width), dtype=np.float32) * 0.5
            
        if not hasattr(self, 'precipitation') or self.precipitation is None:
            print("Initializing missing precipitation attribute") 
            self.precipitation = np.zeros((self.map_height, self.map_width), dtype=np.float32)
        
        # Initialize the temperature field if missing
        if not hasattr(self, 'temperature_celsius') or self.temperature_celsius is None:
            print("Initializing temperature field")
            self.temperature_celsius = np.zeros((self.map_height, self.map_width), dtype=np.float32)
            # Set reasonable defaults based on latitude
            lat_factor = np.abs(self.latitude) / 90.0
            self.temperature_celsius = 30.0 - 40.0 * lat_factor
            
        # For better stability, use a conservative approximation factor
        approximation_factor = 2  # Reduced from 3 to 2 for better stability
        approx_time_step = self.time_step_seconds * approximation_factor
        steps_needed = max(1, int(total_seconds / approx_time_step))
        
        print(f"Running simplified physics with {steps_needed} steps at {approx_time_step/60:.1f} min/step")
        
        # Pre-check for existing instabilities before starting high-speed
        fix_count = self._check_stability_vectorized()
        if fix_count > 0:
            print(f"Fixed {fix_count} issues before starting high-speed simulation")
        
        # Only update essential components in high-speed mode
        # Temperature is most visually important, followed by precipitation
        essential_updates = {
            'temperature': True,   # Always update temperature
            'precipitation': 3,    # Every 3rd step
            'pressure': 4,         # Every 4th step (reduced frequency)
            'wind': 4,             # Every 4th step
            'ocean': 8             # Every 8th step (reduced frequency)
        }
        
        # Run all steps at once with minimal stability checking
        for step in range(steps_needed):
            # Increment time step counter
            self.time_step += 1
            
            # Apply scaled time step
            temp_time_step = self.time_step_seconds
            self.time_step_seconds = approx_time_step
            
            try:
                # 1. Always update temperature (most important for visualization)
                # Use simplified temperature update that's more stable at large time steps
                self._update_temperature_simplified()
                
                # 2. Selective system updates based on schedule
                if step % essential_updates['precipitation'] == 0:
                    self._update_precipitation_simplified()
                    
                if step % essential_updates['pressure'] == 0:
                    self._update_pressure_simplified()
                    
                if step % essential_updates['wind'] == 0:
                    self._update_wind_simplified()
                    
                if step % essential_updates['ocean'] == 0:
                    self._update_ocean_simplified()
                
                # 3. Only do stability check occasionally to save performance
                if step % 6 == 0:  # Less frequent checks (every 6th step)
                    fix_count = self._check_stability_vectorized()
                    if fix_count > 100:
                        print(f"Step {step}: Fixed {fix_count} stability issues")
                    
            except Exception as e:
                print(f"Error in high-speed step {step}: {e}")
                traceback.print_exc()
                # Reset time step and skip this iteration
                self.time_step_seconds = temp_time_step
                continue
                
            # Restore original time step
            self.time_step_seconds = temp_time_step
            
        # Final stability cleanup
        fix_count = self._check_stability_vectorized()
        if fix_count > 0:
            print(f"Final stability check: fixed {fix_count} issues")
        
        # Update timing stats for high-speed mode
        step_end_time = time.time()
        total_step_time = step_end_time - step_start_time
        
        # Calculate timing metrics
        high_speed_step_time = total_step_time / steps_needed if steps_needed > 0 else 0
        self._last_step_time = high_speed_step_time / approximation_factor
        
        print(f"Completed {steps_needed} steps in {total_step_time:.2f}s ({high_speed_step_time:.4f}s per step)")
        print(f"Effective simulation speed: {3600/high_speed_step_time*approximation_factor:.1f}x real-time")
    
    def _update_temperature_simplified(self):
        """Simplified temperature update for high-speed mode with better stability"""
        try:
            # Basic temperature update with simpler physics
            # Get masks for land and ocean
            is_land = self.elevation > 0
            is_ocean = ~is_land
            
            # 1. Apply solar heating based on latitude - keep 2D shape
            cos_lat = np.cos(np.radians(self.latitude))
            solar_factor = np.clip(cos_lat, 0, 1) * 0.1
            
            # 2. Apply basic greenhouse effect (scalar)
            greenhouse_factor = 0.05
            
            # 3. Apply cooling based on temperature difference from equilibrium
            # Instead of directly using masked arrays which can cause shape issues,
            # create full-sized arrays and then apply masks to update specific regions
            
            # Create equilibrium temperature arrays (same shape as original grid)
            land_equil_full = 15.0 - 30.0 * (np.abs(self.latitude) / 90.0)
            ocean_equil_full = 20.0 - 25.0 * (np.abs(self.latitude) / 90.0)
            
            # Only update if attributes exist
            if hasattr(self, 'temperature_celsius'):
                # Create arrays to hold temperature changes
                delta_temp = np.zeros_like(self.temperature_celsius)
                
                # Calculate temperature changes for land
                if np.any(is_land):
                    cooling_rate_land = 0.05
                    # Calculate relaxation for land regions
                    land_relaxation = (land_equil_full[is_land] - self.temperature_celsius[is_land]) * cooling_rate_land
                    # Add heating factors
                    delta_temp[is_land] = land_relaxation + solar_factor[is_land] + greenhouse_factor
                
                # Calculate temperature changes for ocean
                if np.any(is_ocean):
                    cooling_rate_ocean = 0.015  # Slower for ocean (0.05 * 0.3)
                    # Calculate relaxation for ocean regions
                    ocean_relaxation = (ocean_equil_full[is_ocean] - self.temperature_celsius[is_ocean]) * cooling_rate_ocean
                    # Add heating factors
                    delta_temp[is_ocean] = ocean_relaxation + solar_factor[is_ocean] + greenhouse_factor
                
                # Apply all changes at once
                self.temperature_celsius += delta_temp
                
                # Apply simple smoothing for stability
                self.temperature_celsius = gaussian_filter(self.temperature_celsius, sigma=0.5)
                
                # Enforce temperature bounds
                self.temperature_celsius = np.clip(self.temperature_celsius, -60, 60)
                
        except Exception as e:
            print(f"Error in simplified temperature update: {e}")
            traceback.print_exc()
    
    def _update_precipitation_simplified(self):
        """Simplified precipitation update for high-speed mode with better stability"""
        try:
            # Ensure all required attributes exist to prevent errors
            if not hasattr(self, 'humidity'):
                print("Initializing missing humidity attribute")
                self.humidity = np.ones((self.map_height, self.map_width), dtype=np.float32) * 0.5
                
            if not hasattr(self, 'precipitation'):
                print("Initializing missing precipitation attribute")
                self.precipitation = np.zeros((self.map_height, self.map_width), dtype=np.float32)
                
            if not hasattr(self, 'cloud_cover'):
                print("Initializing missing cloud_cover attribute")
                self.cloud_cover = np.zeros((self.map_height, self.map_width), dtype=np.float32)
                
            # Simple relaxation model
            # 1. Update humidity based on temperature and terrain
            is_ocean = self.elevation <= 0
            
            # Ocean evaporation adds humidity
            if np.any(is_ocean):
                ocean_humidity = 0.7 + 0.2 * np.cos(np.radians(self.latitude[is_ocean]))
                # Relaxation toward target
                self.humidity[is_ocean] += (ocean_humidity - self.humidity[is_ocean]) * 0.1
            
            # Land has lower equilibrium humidity
            if np.any(~is_ocean):
                land_humidity = 0.5 + 0.3 * np.cos(np.radians(self.latitude[~is_ocean]))
                self.humidity[~is_ocean] += (land_humidity - self.humidity[~is_ocean]) * 0.05
            
            # 2. Cloud formation based on humidity
            cloud_threshold = 0.6
            self.cloud_cover = np.clip((self.humidity - cloud_threshold) * 2.0, 0, 1)
            
            # 3. Precipitation based on clouds and stability
            self.precipitation = np.zeros_like(self.cloud_cover)
            rain_mask = self.cloud_cover > 0.4
            if np.any(rain_mask):
                self.precipitation[rain_mask] = (self.cloud_cover[rain_mask] - 0.4) * 0.05
            
            # Apply simple smoothing for stability
            self.humidity = gaussian_filter(self.humidity, sigma=1.0)
            self.cloud_cover = gaussian_filter(self.cloud_cover, sigma=1.0)
            self.precipitation = gaussian_filter(self.precipitation, sigma=1.0)
            
            # Ensure bounds
            self.humidity = np.clip(self.humidity, 0.1, 1.0)
            self.cloud_cover = np.clip(self.cloud_cover, 0, 1.0)
            self.precipitation = np.clip(self.precipitation, 0, 0.1)
                
        except Exception as e:
            print(f"Error in simplified precipitation update: {e}")
            traceback.print_exc()
    
    def _update_pressure_simplified(self):
        """Simplified pressure update for high-speed mode with better stability"""
        try:
            if hasattr(self, 'pressure') and hasattr(self, 'temperature_celsius'):
                # 1. Base pressure based on elevation (higher = lower pressure)
                elevation_factor = np.exp(-np.clip(self.elevation, 0, 2000) / 8000)
                
                # 2. Temperature effect (hotter = lower pressure)
                temp_factor = 1.0 - (self.temperature_celsius - 15) / 100
                
                # 3. Latitude effect (high/low pressure bands)
                lat_rad = np.abs(np.radians(self.latitude))
                lat_factor = 1.0 + 0.02 * np.cos(lat_rad * 6)
                
                # Calculate target pressure
                base_pressure = 101325.0  # 1013.25 hPa
                target_pressure = base_pressure * elevation_factor * temp_factor * lat_factor
                
                # Apply relaxation toward target pressure
                relaxation_rate = 0.1
                self.pressure += (target_pressure - self.pressure) * relaxation_rate
                
                # Apply simple smoothing for stability
                self.pressure = gaussian_filter(self.pressure, sigma=1.0)
                
                # Ensure realistic pressure range
                self.pressure = np.clip(self.pressure, 87000, 108000)
                
        except Exception as e:
            print(f"Error in simplified pressure update: {e}")
    
    def _update_wind_simplified(self):
        """Simplified wind update for high-speed mode with better stability"""
        try:
            if hasattr(self, 'u') and hasattr(self, 'v') and hasattr(self, 'pressure'):
                # Calculate pressure gradients
                dy, dx = np.gradient(self.pressure)
                
                # Convert to wind components (negative gradient = wind direction)
                # Scale by grid spacing
                target_u = -dx / self.grid_spacing_x * 0.0001
                target_v = -dy / self.grid_spacing_y * 0.0001
                
                # Apply Coriolis effect (simplified)
                coriolis_factor = np.sin(np.radians(self.latitude)) * 0.0001
                target_u_coriolis = target_u - coriolis_factor * target_v
                target_v_coriolis = target_v + coriolis_factor * target_u
                
                # Update wind with relaxation
                relaxation_rate = 0.2
                self.u += (target_u_coriolis - self.u) * relaxation_rate
                self.v += (target_v_coriolis - self.v) * relaxation_rate
                
                # Apply simple smoothing
                self.u = gaussian_filter(self.u, sigma=1.0)
                self.v = gaussian_filter(self.v, sigma=1.0)
                
                # Clip to realistic values
                max_wind = 30.0
                wind_mag = np.sqrt(self.u**2 + self.v**2)
                wind_correction = np.where(
                    wind_mag > max_wind,
                    max_wind / wind_mag,
                    1.0
                )
                self.u *= wind_correction
                self.v *= wind_correction
                
        except Exception as e:
            print(f"Error in simplified wind update: {e}")


    def _update_ocean_simplified(self):
        """Simplified ocean temperature update for high-speed mode with better stability"""
        try:
            if hasattr(self, 'temperature_celsius') and hasattr(self, 'elevation'):
                # Only process ocean cells
                is_ocean = self.elevation <= 0
                if np.any(is_ocean):
                    # 1. Ocean has high thermal inertia - changes more slowly
                    # Base ocean temperature on latitude
                    lat = np.abs(self.latitude[is_ocean])
                    target_temp = 25.0 - 25.0 * (lat / 90.0)
                    
                    # Apply slower relaxation for ocean
                    relaxation_rate = 0.02
                    self.temperature_celsius[is_ocean] += (target_temp - self.temperature_celsius[is_ocean]) * relaxation_rate
                    
                    # 2. Apply simplified ocean currents (just diffusion)
                    # Extract ocean temperature
                    ocean_temp = np.zeros_like(self.temperature_celsius)
                    ocean_temp[is_ocean] = self.temperature_celsius[is_ocean]
                    
                    # Apply diffusion
                    ocean_temp = gaussian_filter(ocean_temp, sigma=1.5) 
                    
                    # Update only ocean cells
                    self.temperature_celsius[is_ocean] = ocean_temp[is_ocean]
                    
                    # Ensure realistic ocean temperature bounds
                    self.temperature_celsius[is_ocean] = np.clip(self.temperature_celsius[is_ocean], -2, 30)
                
        except Exception as e:
            print(f"Error in simplified ocean update: {e}")

    def _check_stability_vectorized(self):
        """Vectorized stability check that's much faster than the original loop-based version"""
        try:
            fixed_count = 0
            severe_count = 0
            
            # 1. Check temperature field (most common source of instability)
            if hasattr(self, 'temperature_celsius'):
                # Create a mask for all problematic values
                temp_invalid = (
                    np.isnan(self.temperature_celsius) | 
                    np.isinf(self.temperature_celsius) | 
                    (self.temperature_celsius < -100) | 
                    (self.temperature_celsius > 100)
                )
                
                if np.any(temp_invalid):
                    # Count fixes
                    fixed_count += np.sum(temp_invalid)
                    severe_count += np.sum(np.isnan(self.temperature_celsius) | np.isinf(self.temperature_celsius))
                    
                    # Create default values
                    is_ocean = self.elevation <= 0
                    default_temp = np.zeros_like(self.temperature_celsius)
                    
                    # Create latitude-based defaults
                    y_indices = np.arange(self.map_height)
                    latitude_factor = np.abs(90 - y_indices * 180 / self.map_height) / 90
                    lat_factor_expanded = np.tile(latitude_factor[:, np.newaxis], (1, self.map_width))
                    
                    # Set ocean defaults
                    default_temp[is_ocean] = 15.0 + np.random.uniform(-3, 3, size=np.sum(is_ocean))
                    
                    # Set land defaults
                    default_temp[~is_ocean] = 30 * lat_factor_expanded[~is_ocean] - 10 + np.random.uniform(-5, 5, size=np.sum(~is_ocean))
                    
                    # Apply fixes
                    self.temperature_celsius[temp_invalid] = default_temp[temp_invalid]
            
            # 2. Check pressure field
            if hasattr(self, 'pressure'):
                # Create mask for invalid pressure
                pressure_invalid = (
                    np.isnan(self.pressure) | 
                    np.isinf(self.pressure) | 
                    (self.pressure < 87000) | 
                    (self.pressure > 108000)
                )
                
                if np.any(pressure_invalid):
                    # Count fixes
                    fixed_count += np.sum(pressure_invalid)
                    severe_count += np.sum(np.isnan(self.pressure) | np.isinf(self.pressure))
                    
                    # Set to standard pressure with small random variation
                    self.pressure[pressure_invalid] = 101325.0 + np.random.uniform(-500, 500, size=np.sum(pressure_invalid))
            
            # 3. Check wind components
            if hasattr(self, 'u') and hasattr(self, 'v'):
                # Create masks for invalid wind components
                u_invalid = (
                    np.isnan(self.u) | 
                    np.isinf(self.u) | 
                    (self.u < -50) | 
                    (self.u > 50)
                )
                
                v_invalid = (
                    np.isnan(self.v) | 
                    np.isinf(self.v) | 
                    (self.v < -50) | 
                    (self.v > 50)
                )
                
                # Fix u component
                if np.any(u_invalid):
                    fixed_count += np.sum(u_invalid)
                    severe_count += np.sum(np.isnan(self.u) | np.isinf(self.u))
                    self.u[u_invalid] = 0.0
                
                # Fix v component
                if np.any(v_invalid):
                    fixed_count += np.sum(v_invalid)
                    severe_count += np.sum(np.isnan(self.v) | np.isinf(self.v))
                    self.v[v_invalid] = 0.0
            
            # 4. Check precipitation and humidity
            if hasattr(self, 'precipitation'):
                precip_invalid = (
                    np.isnan(self.precipitation) | 
                    np.isinf(self.precipitation) | 
                    (self.precipitation < 0) | 
                    (self.precipitation > 1)
                )
                
                if np.any(precip_invalid):
                    fixed_count += np.sum(precip_invalid)
                    severe_count += np.sum(np.isnan(self.precipitation) | np.isinf(self.precipitation))
                    self.precipitation[precip_invalid] = 0.0
            
            if hasattr(self, 'humidity'):
                humidity_invalid = (
                    np.isnan(self.humidity) | 
                    np.isinf(self.humidity) | 
                    (self.humidity < 0) | 
                    (self.humidity > 1)
                )
                
                if np.any(humidity_invalid):
                    fixed_count += np.sum(humidity_invalid)
                    severe_count += np.sum(np.isnan(self.humidity) | np.isinf(self.humidity))
                    self.humidity[humidity_invalid] = 0.5
            
            # 5. Check cloud cover
            if hasattr(self, 'cloud_cover'):
                cloud_invalid = (
                    np.isnan(self.cloud_cover) | 
                    np.isinf(self.cloud_cover) | 
                    (self.cloud_cover < 0) | 
                    (self.cloud_cover > 1)
                )
                
                if np.any(cloud_invalid):
                    fixed_count += np.sum(cloud_invalid)
                    severe_count += np.sum(np.isnan(self.cloud_cover) | np.isinf(self.cloud_cover))
                    self.cloud_cover[cloud_invalid] = 0.0
            
            # Take emergency actions if severe issues detected
            if severe_count > 100 or fixed_count > 50000:
                print(f"STABILITY: Fixed {fixed_count} issues, including {severe_count} severe problems")
                if self.high_speed_mode and fixed_count > 500000:  # Increased from 100,000 to 500,000
                    print("EMERGENCY: Disabling high-speed mode due to severe instability")
                    self.high_speed_mode = False
                    if hasattr(self, 'high_speed_var'):
                        self.high_speed_var.set(False)
                    
            return fixed_count
                        
        except Exception as e:
            print(f"Error in stability check: {e}")
            return 0

    def update_mouse_over(self, event=None):
        """Update mouse-over information display"""
        try:
            # Track when this update happens
            self.last_mouse_update_time = time.time() * 1000  # Convert to milliseconds
            self.mouse_position_changed = False  # Reset change flag
            
            # Get mouse coordinates
            if event is not None:
                # Update from event if available
                x, y = event.x, event.y
                
                # Store last mouse position for reference
                self.last_mouse_x = x
                self.last_mouse_y = y
                
                # Also store in zoom dialog if it exists
                if hasattr(self, 'zoom_dialog') and self.zoom_dialog and self.zoom_dialog.winfo_exists():
                    self.zoom_dialog.last_main_x = x
                    self.zoom_dialog.last_main_y = y
            elif hasattr(self, 'last_mouse_x') and hasattr(self, 'last_mouse_y'):
                # Use stored position if no event
                x, y = self.last_mouse_x, self.last_mouse_y
            else:
                # No position data available
                if hasattr(self, 'mouse_over_label') and self.mouse_over_label:
                    self.mouse_over_label.config(text="Move mouse over map for cell information")
                return
            
            # Skip if out of bounds
            if x < 0 or y < 0 or x >= self.map_width or y >= self.map_height:
                if hasattr(self, 'mouse_over_label') and self.mouse_over_label:
                    self.mouse_over_label.config(text="Move mouse over map for cell information")
                return
                
            # Extract values from the map data at this location
            display_parts = [f"Position: ({x}, {y})"]
            
            # Determine if ocean or land
            is_ocean = False
            if hasattr(self, 'elevation') and self.elevation is not None:
                elevation_value = self.elevation[y, x]
                display_parts.append(f"Elevation: {elevation_value:.1f}m")
                
                is_ocean = elevation_value <= 0
                if is_ocean:
                    display_parts.append("(Ocean)")
                else:
                    display_parts.append("(Land)")
            
            # Always show temperature data (depends on land vs ocean)
            if hasattr(self, 'temperature_celsius'):
                # For land, show air temperature
                if not is_ocean:
                    temp = self.temperature_celsius[y, x]
                    display_parts.append(f"Temp: {temp:.1f}°C")
                # For ocean, show ocean temperature from temperature_celsius (not ocean_temperature)
                else:
                    # Ocean temperature is stored in the same temperature_celsius array
                    ocean_temp = self.temperature_celsius[y, x]
                    display_parts.append(f"Ocean Temp: {ocean_temp:.1f}°C")
                    
                    # Show ocean layer temperatures if available
                    if hasattr(self, 'ocean_layers') and self.ocean_layers is not None:
                        # Show surface layer temperature explicitly
                        surface_temp = self.ocean_layers[0][y, x]
                        display_parts.append(f"Surface Ocean: {surface_temp:.1f}°C")
                        
                        # Optionally show deeper layer temps
                        if len(self.ocean_layers) > 1:
                            mixed_temp = self.ocean_layers[1][y, x]
                            display_parts.append(f"Mixed Layer: {mixed_temp:.1f}°C")
            
            # Always show pressure if available
            if hasattr(self, 'pressure'):
                pressure = self.pressure[y, x]
                display_parts.append(f"Pressure: {pressure/100:.1f} hPa")
                
            # Always show wind if available
            if hasattr(self, 'u') and hasattr(self, 'v'):
                u = self.u[y, x]
                v = self.v[y, x]
                speed = np.sqrt(u*u + v*v)
                if hasattr(self, 'wind_system') and hasattr(self.wind_system, 'calculate_direction'):
                    direction = self.wind_system.calculate_direction(u, v)
                    display_parts.append(f"Wind: {speed:.1f} m/s {direction:.0f}°")
                else:
                    display_parts.append(f"Wind: {speed:.1f} m/s")
                
            # Always show precipitation if available
            if hasattr(self, 'precipitation'):
                precip = self.precipitation[y, x]
                if precip > 0.001:  # Only show if significant
                    display_parts.append(f"Rain: {precip:.2f} mm/h")
                
            # Always show humidity if available
            if hasattr(self, 'humidity'):
                humidity = self.humidity[y, x] * 100
                display_parts.append(f"Humidity: {humidity:.0f}%")
                
            # Always show cloud cover if available
            if hasattr(self, 'cloud_cover'):
                clouds = self.cloud_cover[y, x] * 100
                if clouds > 1.0:  # Only show if significant
                    display_parts.append(f"Clouds: {clouds:.0f}%")
            
            # Join all parts with pipe separator
            display_str = " | ".join(display_parts)
            
            # Update mouse over label if it exists
            if hasattr(self, 'mouse_over_label') and self.mouse_over_label:
                self.mouse_over_label.config(text=display_str)
                
        except Exception as e:
            print(f"Error in update_mouse_over: {e}")
            # Don't reraise to prevent thread crashes

    def cleanup(self):
        """Clean up threads before closing"""
        # Stop the simulation thread
        self.simulation_thread_active.clear()
        self.simulation_running = False
        
        # Stop the visualization thread
        self.visualization_active.clear()
        
        # Wait for threads to terminate
        if hasattr(self, 'visualization_thread'):
            self.visualization_thread.join(timeout=1.0)
            
        if hasattr(self, 'simulation_thread'):
            self.simulation_thread.join(timeout=1.0)

    def toggle_stats_display(self, enabled=None):
        """Toggle or set the system statistics display"""
        if enabled is None:
            # Toggle current state
            self.show_stats = not self.show_stats
            self.system_stats.print_stats_enabled = self.show_stats
        else:
            # Set to specified state
            self.show_stats = enabled
            self.system_stats.print_stats_enabled = enabled
            
        # Force an update when enabled
        if self.show_stats and hasattr(self, 'system_stats'):
            # Set flag to force update on next cycle regardless of timing
            self.system_stats.force_next_update = True
            
            # Force update by directly calling print_stats
            self.system_stats.print_stats()
            
            # Reset the last update step to ensure next cycle updates
            self.system_stats.last_update_step = 0
            
            # Also update the checkbox if it exists to match the current state
            if hasattr(self, 'stats_var'):
                self.stats_var.set(self.show_stats)

    def toggle_high_speed_mode(self):
        """Toggle or set the high-speed approximation mode"""
        old_value = self.high_speed_mode
        new_value = self.high_speed_var.get()
        
        # If no change, do nothing to avoid unnecessary work
        if old_value == new_value:
            return
            
        self.high_speed_mode = new_value
        mode_text = "enabled" if self.high_speed_mode else "disabled"
        print(f"High-speed mode {mode_text}")
        
        # Use a short-term flag to indicate we're transitioning modes
        # This allows the simulation to handle the transition more smoothly
        self._transitioning_speed_mode = True
        
        # For enabling high-speed mode, show a temporary status message
        # instead of blocking with a MessageBox
        if self.high_speed_mode:
            if hasattr(self, 'system_label') and self.system_label:
                current_text = self.system_label.cget("text")
                self.system_label.config(text=f"High-speed mode enabled - optimizing simulation...")
                
                # Schedule restoration of normal status text after a short delay
                self.root.after(2000, lambda: self.system_label.config(text=current_text))
        
        # Reset timing data to force recalculation but don't delete it
        # This avoids allocation/deallocation overhead
        if hasattr(self, '_last_step_time'):
            self._last_step_time = 0.1  # Reset to initial value
        
        # Schedule stability check to run after UI update completes
        # This prevents UI freezing during the check
        self.root.after(100, self._perform_mode_transition_tasks)
    
    def _perform_mode_transition_tasks(self):
        """Perform tasks needed when transitioning between simulation modes"""
        try:
            # Perform a lightweight stability check when changing modes
            if hasattr(self, '_check_stability_vectorized'):
                # Use a more targeted stability check during mode transitions
                self._check_stability_for_mode_transition()
            
            # Clear the transition flag
            self._transitioning_speed_mode = False
        except Exception as e:
            print(f"Error during mode transition: {e}")
            traceback.print_exc()
            self._transitioning_speed_mode = False
    
    def _check_stability_for_mode_transition(self):
        """Perform a faster targeted stability check during mode transitions"""
        # Only check temperature and pressure - the most critical fields
        # This is much faster than the full stability check
        fixed_count = 0
        
        # Check temperature field using vectorized operations
        if hasattr(self, 'temperature_celsius'):
            temp = self.temperature_celsius
            if temp is not None:
                # Find all temperature values that are outside valid range or NaN/Inf
                invalid_temp = np.isnan(temp) | np.isinf(temp) | (temp < -100) | (temp > 100)
                fixed_count += np.sum(invalid_temp)
                
                # Fix invalid temperature values
                if np.any(invalid_temp):
                    # Create replacement values based on latitude
                    height, width = temp.shape
                    y_indices, x_indices = np.where(invalid_temp)
                    
                    # Create default values for fixes
                    is_ocean = self.elevation[y_indices, x_indices] <= 0
                    
                    # Apply ocean defaults (ocean temperature is more stable)
                    ocean_fixes = np.where(is_ocean)[0]
                    if len(ocean_fixes) > 0:
                        # Use moderate ocean temperature with small variations
                        ocean_default = 15.0 + np.random.uniform(-3, 3, size=len(ocean_fixes))
                        ocean_idx = y_indices[ocean_fixes], x_indices[ocean_fixes]
                        temp[ocean_idx] = ocean_default
                    
                    # Apply land defaults (more variable by latitude)
                    land_fixes = np.where(~is_ocean)[0]
                    if len(land_fixes) > 0:
                        # Calculate latitude-based temperatures
                        land_idx = y_indices[land_fixes], x_indices[land_fixes]
                        latitude_factor = abs(90 - land_idx[0] * 180 / height) / 90
                        land_default = 30 * latitude_factor - 10 + np.random.uniform(-5, 5, size=len(land_fixes))
                        temp[land_idx] = land_default
        
        # Check pressure field using vectorized operations
        if hasattr(self, 'pressure'):
            pressure = self.pressure
            if pressure is not None:
                # Find all pressure values that are outside valid range or NaN/Inf
                invalid_pressure = np.isnan(pressure) | np.isinf(pressure) | (pressure < 87000) | (pressure > 108000)
                fixed_count += np.sum(invalid_pressure)
                
                # Fix invalid pressure values
                if np.any(invalid_pressure):
                    # Set to standard pressure with small variations
                    pressure[invalid_pressure] = 1013.0 + np.random.uniform(-10, 10)
        
        # Only print message if fixes were substantial
        if fixed_count > 100:
            print(f"Mode transition - fixed {fixed_count} values for stability")
            
        return fixed_count

                    
    def _emergency_stability_reset(self):
        """Perform emergency reset of simulation state to recover stability"""
        # Disable high-speed mode
        if self.high_speed_mode:
            self.high_speed_mode = False
            if hasattr(self, 'high_speed_var'):
                self.high_speed_var.set(False)
                
        # Reduce time step
        self.time_step_seconds = min(self.time_step_seconds, 600)  # Max 10 minutes
        
        # Reset temperature field with smooth, physically reasonable values
        if hasattr(self, 'temperature') and hasattr(self.temperature, 'temperature'):
            shape = self.temperature.temperature.shape
            new_temp = np.zeros(shape)
            
            # Helper function to safely get terrain information
            def is_ocean(x, y):
                try:
                    return self.elevation[y, x] <= 0
                except (IndexError, TypeError):
                    return False
                    
            # Generate realistic temperature field
            for y in range(shape[0]):
                for x in range(shape[1]):
                    # Calculate latitude factor (0 at poles, 1 at equator)
                    latitude_factor = 1.0 - abs(y - shape[0]//2) / (shape[0]//2)
                    
                    # Base temperature varies with latitude (warm at equator, cold at poles)
                    base_temp = 30 * latitude_factor - 15
                    
                    # Add some noise
                    noise = np.random.normal(0, 2)
                    
                    # Oceans have more moderate temperatures
                    if is_ocean(x, y):
                        # Ocean temperature varies less with latitude
                        ocean_temp = 25 * latitude_factor - 5
                        new_temp[y, x] = ocean_temp + noise * 0.5
                    else:
                        # Land has more extreme temperatures
                        new_temp[y, x] = base_temp + noise
            
            # Apply final smoothing
            new_temp = gaussian_filter(new_temp, sigma=1.0)
            
            # Replace temperature field
            self.temperature.temperature = new_temp.astype(np.float32)
            self.temperature_celsius = new_temp.astype(np.float32)
        
        # Reset wind to gentle global patterns
        if hasattr(self, 'wind_system'):
            if hasattr(self.wind_system, 'u') and self.wind_system.u is not None:
                shape = self.wind_system.u.shape
                # Create simple wind patterns based on latitude
                for y in range(shape[0]):
                    # Calculate latitude (-1 to 1, where 0 is equator)
                    lat = 2 * (y / shape[0] - 0.5)
                    
                    # Create simple zonal wind pattern (east-west)
                    # Trade winds, westerlies, and polar easterlies
                    if abs(lat) < 0.3:  # Trade winds (easterly near equator)
                        self.wind_system.u[y, :] = -5.0
                    elif abs(lat) < 0.7:  # Westerlies (mid-latitudes)
                        self.wind_system.u[y, :] = 8.0
                    else:  # Polar easterlies (high latitudes)
                        self.wind_system.u[y, :] = -3.0
                    
                    # Add some random noise
                    self.wind_system.u[y, :] += np.random.normal(0, 1, shape[1])
                
                # Apply smoothing
                self.wind_system.u = gaussian_filter(self.wind_system.u, sigma=1.0)
        
        print("Emergency stability reset completed")

    def on_mouse_move(self, event):
        """Handle mouse movement by updating both cell info and zoom view"""
        # Mark that mouse position has changed
        current_x, current_y = event.x, event.y
        
        # Only consider it a change if position is actually different
        if current_x != self.last_mouse_x or current_y != self.last_mouse_y:
            self.mouse_position_changed = True
            
            # Store last position for reference
            self.last_mouse_x = current_x
            self.last_mouse_y = current_y
            
            # Also store in zoom dialog if it exists
            if hasattr(self, 'zoom_dialog') and self.zoom_dialog and self.zoom_dialog.winfo_exists():
                self.zoom_dialog.last_main_x = current_x
                self.zoom_dialog.last_main_y = current_y
        
            # Debounced update - update immediately if it's been a while
            current_time = time.time() * 1000  # Convert to milliseconds
            time_since_update = current_time - self.last_mouse_update_time
            
            if time_since_update > self.mouse_debounce_delay:
                # Enough time has passed, update immediately
                self.update_mouse_over(event)
            else:
                # Otherwise, let the periodic update handle it
                pass
        
        # Always update zoom view for responsiveness, but with low priority
        if hasattr(self, 'visualization'):
            self.visualization.update_zoom_view(event)

    def update_zoom_view(self, event):
        """Update the zoom view if it exists"""
        if hasattr(self, 'visualization') and hasattr(self, 'zoom_dialog') and self.zoom_dialog and self.zoom_dialog.winfo_exists():
            self.visualization.update_zoom_view(event)


class ZoomDialog(tk.Toplevel):
    def __init__(self, parent, zoom_factor=4):
        super().__init__(parent)
        
        # Store reference to parent for event handling
        self.parent = parent
        
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
                              
        # Bind mouse events to update zoom window when mouse moves within it
        self.canvas.bind("<Motion>", self.on_mouse_move)
        
        # Track original mouse position on main canvas
        self.last_main_x = 0
        self.last_main_y = 0
        
    def on_mouse_move(self, event):
        """Handle mouse movement in the zoom window"""
        try:
            # Calculate the relative position in the zoom view
            zoom_x = event.x // self.zoom_factor
            zoom_y = event.y // self.zoom_factor
            
            # Calculate the corresponding position on the main map
            center_offset = self.view_size // 2
            map_x = self.last_main_x - center_offset + zoom_x
            map_y = self.last_main_y - center_offset + zoom_y
            
            # Ensure coordinates are within map bounds
            map_x = max(0, min(map_x, self.parent.map_width - 1))
            map_y = max(0, min(map_y, self.parent.map_height - 1))
            
            # Create a synthetic event to pass to the parent's update_mouse_over
            class SyntheticEvent:
                pass
                
            synthetic_event = SyntheticEvent()
            synthetic_event.x = map_x
            synthetic_event.y = map_y
            
            # Update parent's mouse position display
            self.parent.update_mouse_over(synthetic_event)
            
            # Also update the zoom view with the new coordinates
            if hasattr(self.parent, 'visualization'):
                self.parent.visualization.update_zoom_view(synthetic_event)
                
        except Exception as e:
            print(f"Error in on_mouse_move: {e}")


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        # Context already set, continue with current settings
        pass
        
    print("Creating SimulationApp instance...")
    app = SimulationApp()
    print("Starting mainloop...")
    try:
        app.root.mainloop()
        print("mainloop exited")
    except Exception as e:
        print(f"Error in mainloop: {e}")
        traceback.print_exc()
    
    print("Program ended - sleeping to prevent immediate exit...")
    try:
        # Wait for a while to see output
        import time
        time.sleep(5)
    except KeyboardInterrupt:
        print("Sleep interrupted by user")