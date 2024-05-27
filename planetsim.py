import json
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tkinter as tk
from tqdm import tqdm
from PIL import Image, ImageTk, ImageDraw
from tkinter import Menu
import multiprocessing
from generation import (generate_terrain, generate_craters, scale_to_grayscale, calculate_temperature,
                        calculate_latitudes, model_wind)


class PlanetSim:
    def __init__(self):
        self.current_terrain = None
        self.temperatures = None
        self.TILE_SIZE = 20
        self.canvas_size = 500
        self.terrain_size = 500
        self.TILE_SIZE = self.canvas_size / self.terrain_size
        self.window = tk.Tk()
        self.window.title("Planet Simulator")
        self.canvas = tk.Canvas(self.window, width=self.canvas_size, height=self.canvas_size)
        self.info_label = tk.Label(self.window)
        self.info_label.pack()
        self.current_ocean_map = None
        self.original_image = None
        self.latitudes = None

        self.min_elevation = -420
        self.max_elevation = 8848
        self.ocean_level = 0

        # Set the day of the year and time of day
        self.day_of_year = 80  # March 21st
        self.time_of_day = 12  # Noon

        self.ALBEDO_WATER = 0.06
        self.ALBEDO_LAND = 0.3

        # Create the zoom canvas
        self.zoom_canvas = tk.Canvas(self.window, width=200, height=200)  # Change the size to the desired size
        self.zoom_canvas.place(x=0, y=0)  # Initial position
        self.canvas.bind("<Button-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.zoom_canvas.bind("<B1-Motion>", self.on_mouse_move)

        # Initialize the zoom photo image
        self.zoom_photo_image = None


    def create_sidebar(self, frame_update_function):
        sidebar = tk.Frame(self.window, width=200, bg='gray')

        params = {
            "width": 500,
            "height": 500,
            "scale": 1.0,
            "octaves": 1,
            "persistence": 0.7,
            "lacunarity": 2.0,
            "max_amp": 0,
            "amp": 1,
            "freq": 3,
            "num_small_craters": 15,
            "num_large_craters": 3,
            "max_small_radius": 10,
            "max_large_radius": 40,
            "max_depth": 0.3
        }

        terrain_params = {
            "scale": 200.0,
            "octaves": 2,
            "persistence": 0.7,
            "lacunarity": 2.0,
            "strength": 0.5,
            "scale_factor": 4,
            "num_small_craters": 500,
            "num_large_craters": 10,
            "max_small_radius": 30,
            "max_large_radius": 120,
            "max_depth": 0.3
        }

        entries = {}
        for param, value in params.items():
            frame = tk.Frame(sidebar, bg='gray')
            frame.pack(fill='x', padx=5, pady=5)
            label = tk.Label(frame, text=param.capitalize(), bg='gray')
            label.pack(side='left')
            entry = tk.Entry(frame)
            entry.pack(side='right', expand=True, anchor='e')  # Right justify the entry field
            entry.insert(0, str(value))
            entries[param] = entry

        self.terrain_entries = {}
        for param, value in terrain_params.items():
            frame = tk.Frame(sidebar, bg='gray')
            frame.pack(fill='x', padx=5, pady=5)
            label = tk.Label(frame, text=param.capitalize(), bg='gray')
            label.pack(side='left')
            entry = tk.Entry(frame)
            entry.pack(side='right', expand=True)
            entry.insert(0, str(value))
            self.terrain_entries[param] = entry
            frame.pack_forget()  # Hide the terrain entries initially

        def on_generate_clicked(self):
            self.save_settings(entries)
            self.params = self.load_settings()  # Load the updated settings
            frame_update_function(self.params)

        generate_button = tk.Button(sidebar, text="Generate Terrain", command=lambda: on_generate_clicked(self))
        generate_button.pack(pady=5)
        exit_button = tk.Button(sidebar, text="Exit", command=lambda: self.exit_program(self.window, entries))
        exit_button.pack(pady=5)

        return sidebar, entries

    def generate_random_terrain(self):
        # Generate the terrain
        print("Generating terrain...")
        self.current_terrain = generate_terrain(
            int(self.params['width']), int(self.params['height']), scale=self.params['scale'],
            octaves=int(self.params['octaves']), persistence=self.params['persistence'],
            lacunarity=self.params['lacunarity']
        )
        if self.current_terrain is None:
            print("Error: generate_terrain returned None")
            return

        # Add craters to the terrain
        print("Generating craters...")
        self.current_terrain = generate_craters(
            self.current_terrain, num_small_craters=int(self.params['num_small_craters']),
            num_large_craters=int(self.params['num_large_craters']),
            max_small_radius=self.params['max_small_radius'], max_large_radius=self.params['max_large_radius'],
            max_depth=self.params['max_depth']
        )
        if self.current_terrain is None:
            print("Error: generate_craters returned None")
            return

        self.sidebar.pack(fill='y', side='left', padx=5, pady=5)  # Pack the sidebar here
        for frame in self.terrain_entries.values():
            frame.pack()  # Show the terrain entries

        # Update the terrain display
        self.update_terrain_display(self.current_terrain)

    @staticmethod
    def load_terrain_chunk(image_data, min_elevation, max_elevation, start_row, end_row):
        elevation_matrix = np.zeros((end_row - start_row, image_data.shape[1]), dtype=np.float32)
        scale_factor = (max_elevation - min_elevation) / (image_data.max() - image_data.min() + 1)
        for i in range(start_row, end_row):
            for j in range(image_data.shape[1]):
                elevation = (image_data[i][j] - image_data.min()) * scale_factor + min_elevation
                if -2 < elevation < 1:
                    elevation = 1
                elevation = np.around(elevation, 1)
                elevation_matrix[i - start_row][j] = elevation
        return elevation_matrix

    def load_terrain(self, image_path, min_elevation, max_elevation):
        with rasterio.open(image_path) as src:
            image_data = src.read(1)

        # Define the chunk size for multiprocessing
        chunk_size = image_data.shape[0] // multiprocessing.cpu_count()

        # Get the number of cores for multiprocessing
        num_cores = multiprocessing.cpu_count() - 1

        # Create arguments for the worker processes
        args = [
            (image_data, min_elevation, max_elevation, i * chunk_size, (i + 1) * chunk_size)
            for i in range(num_cores)]

        with multiprocessing.Pool(num_cores) as pool:
            with tqdm(total=len(args), desc='Loading terrain') as pbar:
                results = pool.starmap(self.load_terrain_chunk, args)
                pbar.update(len(args))

        # Combine the chunks back into a single terrain array
        elevation_matrix = np.concatenate(results)

        return elevation_matrix

    def update_terrain_display(self, terrain, ocean_map=None):
        try:
            # Get the 'terrain' colormap
            terrain_cmap = plt.get_cmap('terrain')

            # Create a custom colormap that excludes the bottom 25% of color values
            custom_cmap = mcolors.LinearSegmentedColormap.from_list(
                'custom_cmap',
                terrain_cmap(np.linspace(0.25, 1, 256))
            )

            # Check if the program is using the random_terrain or elevation_matrix
            if np.array_equal(terrain, self.current_terrain):
                # Use the min and max values of self.current_terrain
                norm = plt.Normalize(vmin=terrain.min(), vmax=terrain.max())
            else:
                # Normalize the terrain values to the range of the colormap
                norm = plt.Normalize(vmin=self.min_elevation, vmax=self.max_elevation)

            # Calculate the colors for the terrain using vectorized operations
            terrain_colors = terrain_cmap(norm(terrain))
            terrain_colors = np.array(terrain_colors)  # Convert terrain_colors to a numpy array
            terrain_colors[terrain > 0] = custom_cmap(norm(terrain[terrain > 0]))

            # Store the terrain colors
            self.terrain_colors = terrain_colors

            # Store the grayscale version of the terrain
            self.original_image = (terrain_colors[:, :, :3] * 255).astype(np.uint8)

            # If ocean_map is provided, set the color of ocean tiles to blue
            if ocean_map is not None:
                # Resize ocean_map to match terrain_colors using PIL.Image.resize
                ocean_map_image = Image.fromarray(ocean_map)
                ocean_map_resized = ocean_map_image.resize((terrain_colors.shape[1], terrain_colors.shape[0]),
                                                           Image.NEAREST)
                ocean_map = np.array(ocean_map_resized)

                # Only update the colors of the tiles that are classified as ocean tiles
                terrain_colors[np.where(ocean_map)] = [0, 0, 1, 1]  # Blue color for ocean tiles

            # Convert the colors from RGBA to RGB and scale to [0, 255]
            colors = (terrain_colors[:, :, :3] * 255).astype(np.uint8)

            # Create an image from the colors array
            with tqdm(total=1, desc='Creating image') as pbar:
                image = Image.fromarray(colors)
                pbar.update()

            # Convert the PIL.Image.Image object to a tkinter.PhotoImage object
            with tqdm(total=1, desc='Converting to PhotoImage') as pbar:
                photo_image = ImageTk.PhotoImage(image)
                pbar.update()

            image = Image.fromarray(colors)
            image = image.resize((terrain.shape[1], terrain.shape[0]))  # Resize the image to match the terrain size
            photo_image = ImageTk.PhotoImage(image)

            # Clear the canvas
            self.canvas.delete('all')

            # Resize the canvas to match the size of the terrain
            self.canvas.config(width=terrain.shape[1], height=terrain.shape[0])
            self.canvas.update()

            # Draw the PhotoImage on the canvas
            with tqdm(total=1, desc='Drawing PhotoImage') as pbar:
                self.canvas.create_image(0, 0, image=photo_image, anchor='nw')
                pbar.update()

            # Keep a reference to the PhotoImage to prevent it from being garbage collected
            self.photo_image = photo_image

            print("Canvas updated")

        except Exception as e:
            print(f"An error occurred: {e}")


    @staticmethod
    def classify_ocean(elevation_matrix, ocean_level=394.66, neighbor_threshold=4):
        # Initialize the ocean map
        ocean_map = np.zeros_like(elevation_matrix, dtype=bool)

        # Classify ocean tiles
        for i in range(elevation_matrix.shape[0]):
            for j in range(elevation_matrix.shape[1]):
                # Mark tiles as ocean if their elevation is at or below the ocean level
                ocean_map[i][j] = elevation_matrix[i][j] < ocean_level

        # Create a copy of the ocean map to modify
        new_ocean_map = ocean_map.copy()

        # Iterate over the ocean map again
        for i in range(1, ocean_map.shape[0] - 1):
            for j in range(1, ocean_map.shape[1] - 1):
                # If the current tile is not an ocean tile
                if not ocean_map[i][j]:
                    # Count the number of neighboring ocean tiles
                    neighbor_count = np.sum(ocean_map[i - 1:i + 2, j - 1:j + 2]) - ocean_map[i][j]

                    # If the number of neighboring ocean tiles is greater than or equal to the threshold
                    if neighbor_count >= neighbor_threshold:
                        # Mark the current tile as an ocean tile
                        new_ocean_map[i][j] = True

        return new_ocean_map

    def flood_fill(self, i, j, ocean_map, flood_fill_map):
        # Stack for the tiles to be checked
        stack = [(i, j)]

        while stack:
            i, j = stack.pop()
            if not flood_fill_map[i, j] and ocean_map[i, j]:
                flood_fill_map[i, j] = True
                if i > 0:
                    stack.append((i - 1, j))
                if i < ocean_map.shape[0] - 1:
                    stack.append((i + 1, j))
                if j > 0:
                    stack.append((i, j - 1))
                if j < ocean_map.shape[1] - 1:
                    stack.append((i, j + 1))

    def remove_inland_oceans(self, ocean_map):
        flood_fill_map = np.zeros_like(ocean_map, dtype=bool)
        for i in range(ocean_map.shape[0]):
            for j in range(ocean_map.shape[1]):
                if i == 0 or i == ocean_map.shape[0] - 1 or j == 0 or j == ocean_map.shape[1] - 1:
                    if ocean_map[i, j]:
                        self.flood_fill(i, j, ocean_map, flood_fill_map)

        return np.logical_and(ocean_map, flood_fill_map)

    def load_water_mask(self, image_path):
        # Load the image
        img = Image.open(image_path)
        # Convert the image to grayscale
        img = img.convert('L')
        # Convert the image to a numpy array
        img_array = np.array(img)
        # Classify the tiles as ocean or not
        ocean_map = img_array == 255

        return ocean_map

    def update_frame(self, params):
        # Generate the terrain
        self.current_terrain = generate_terrain(
            int(params['width']), int(params['height']), scale=params['scale'],
            octaves=int(params['octaves']), persistence=params['persistence'], lacunarity=params['lacunarity']
        )
        if self.current_terrain is None:
            print("Error: generate_terrain returned None")
            return

        # Add craters to the terrain
        self.current_terrain = generate_craters(
            self.current_terrain, num_small_craters=int(params['num_small_craters']),
            num_large_craters=int(params['num_large_craters']),
            max_small_radius=params['max_small_radius'], max_large_radius=params['max_large_radius'],
            max_depth=params['max_depth']
        )
        if self.current_terrain is None:
            print("Error: generate_craters returned None")
            return

        # Ensure self.current_terrain is a 2D array
        if len(self.current_terrain.shape) != 2:
            # Reshape self.current_terrain into a 2D array
            self.current_terrain = self.current_terrain.reshape(
                (self.current_terrain.shape[0], self.current_terrain.shape[1]))

        # Update the terrain display
        self.update_terrain_display(self.current_terrain)

    def create_image(self, terrain):
        terrain_scaled = scale_to_grayscale(terrain).astype('uint8')
        img = Image.fromarray(terrain_scaled, 'L')
        return img

    def load_settings(self):
        try:
            with open('settings.json', 'r') as f:
                settings = json.load(f)
        except FileNotFoundError:
            print("Settings file not found. Using default settings.")
            settings = {}

        # List of all keys that should be present in the settings
        keys = ['width', 'height', 'scale', 'octaves', 'persistence', 'lacunarity', 'num_small_craters',
                'num_large_craters', 'max_small_radius', 'max_large_radius', 'max_depth']

        # Default values for the keys
        default_values = [500, 500, 1.0, 1, 0.7, 2.0, 15, 3, 10, 40, 0.3]

        # Ensure all necessary keys are present in settings
        for i, key in enumerate(keys):
            if key not in settings:
                print(f"Key '{key}' not found in settings. Using default value.")
                settings[key] = default_values[i]
            elif self.entries[key].winfo_exists():
                self.entries[key].delete(0, tk.END)
                self.entries[key].insert(0, str(settings[key]))

        return settings

    def view_terrain(self):
        if self.current_terrain is not None and self.current_ocean_map is not None:
            self.update_terrain_display(self.current_terrain, self.current_ocean_map)
            print("Terrain viewed")
        else:
            print("No terrain map available.")

    def view_temperature(self):
        # Ensure self.temperatures is not None and not a scalar
        if self.temperatures is None or self.temperatures.size == 1:
            print("Error: No valid temperature data available.")
            return

        # Calculate min_temp and max_temp
        min_temp = np.min(self.temperatures)
        max_temp = np.max(self.temperatures)

        # Ensure min_temp and max_temp are not None
        if min_temp is None or max_temp is None:
            print("Error: Could not calculate min_temp or max_temp.")
            return

        epsilon = 1e-7
        # Normalize the temperatures to the range [0, 1]
        normalized_temperatures = (self.temperatures - min_temp) / (max_temp - min_temp + epsilon)

        # Create a color map from blue (cooler) to red (hotter)
        cmap = plt.get_cmap('coolwarm')

        # Apply the colormap to the normalized temperatures
        img_data = cmap(normalized_temperatures)

        # Convert the data to 8-bit RGBA values
        img_data = (img_data * 255).astype(np.uint8)
        # Create a PIL image from the data
        temp_img = Image.fromarray(img_data, 'RGBA')

        # Create a PIL image from the terrain colors
        terrain_img = Image.fromarray((self.terrain_colors * 255).astype(np.uint8), 'RGBA')

        # Adjust the alpha channel of the temperature image to control its transparency
        temp_img = temp_img.convert("RGBA")
        datas = temp_img.getdata()

        newData = []
        for item in datas:
            # change all white (also shades of whites)
            # pixels to yellow
            newData.append((item[0], item[1], item[2], 200))

        temp_img.putdata(newData)

        # Paste the temperature image onto the terrain image using the alpha channel of the temperature image as the mask
        terrain_img.paste(temp_img, (0, 0), temp_img)

        # Convert the PIL image to a Tkinter PhotoImage
        photo = ImageTk.PhotoImage(terrain_img)

        # Draw the PhotoImage onto the canvas
        self.canvas.create_image(0, 0, image=photo, anchor='nw')

        # Keep a reference to the PhotoImage to prevent it from being garbage collected
        self.photo_image = photo

    def visualize_wind_model(self):
        # Create a new image with the same dimensions as the terrain
        wind_image = Image.new('RGBA', (self.current_terrain.shape[1], self.current_terrain.shape[0]))

        # Create a Draw object
        draw = ImageDraw.Draw(wind_image)

        # Call the model_wind function to get wind_speed and wind_direction
        wind_speed, wind_direction = model_wind(self.current_terrain, self.latitudes)

        # Create a grid of x and y coordinates with a step size
        step_size = 35
        y, x = np.mgrid[0:wind_speed.shape[0]:step_size, 0:wind_speed.shape[1]:step_size]

        # Calculate the u and v components of the wind speed at the sampled points
        u = wind_speed[::step_size, ::step_size] * np.cos(wind_direction[::step_size, ::step_size])
        v = wind_speed[::step_size, ::step_size] * np.sin(wind_direction[::step_size, ::step_size])

        # Convert the u and v components to canvas coordinates
        scale_factor = 0.02  # Adjust this factor to control the size of wind vectors
        u_canvas = u * scale_factor
        v_canvas = v * scale_factor

        # Calculate the start and end points for the arrows
        start_x = x
        start_y = y
        end_x = start_x + u_canvas
        end_y = start_y + v_canvas

        # Calculate the logarithm of the wind speed
        wind_speed_log = np.log(wind_speed[::step_size, ::step_size])

        # Normalize the logarithmic wind speed to a range of [0, 1]
        wind_speed_log_norm = (wind_speed_log - wind_speed_log.min()) / (wind_speed_log.max() - wind_speed_log.min())

        # Define the minimum and maximum arrow sizes
        min_arrow_size = 1
        max_arrow_size = 15

        # Calculate the arrow sizes based on the normalized logarithmic wind speed
        arrow_sizes = min_arrow_size + (max_arrow_size - min_arrow_size) * wind_speed_log_norm

        # Define the minimum and maximum tail widths
        min_tail_width = 1
        max_tail_width = 3

        # Calculate the tail widths based on the normalized logarithmic wind speed
        tail_widths = min_tail_width + (max_tail_width - min_tail_width) * wind_speed_log_norm

        # Draw the wind vectors as arrows on the canvas
        for sx, sy, ex, ey, arrow_size, tail_width in zip(start_x.flatten(), start_y.flatten(), end_x.flatten(),
                                                          end_y.flatten(), arrow_sizes.flatten(),
                                                          tail_widths.flatten()):
            self.canvas.create_line(sx, sy, ex, ey, fill='white', width=tail_width, arrow=tk.LAST)
            angle = np.arctan2(ey - sy, ex - sx)
            arrow_coords = [
                (ex, ey),
                (ex - arrow_size * np.cos(angle - np.pi / 6), ey - arrow_size * np.sin(angle - np.pi / 6)),
                (ex - arrow_size * np.cos(angle + np.pi / 6), ey - arrow_size * np.sin(angle + np.pi / 6))
            ]

        # Convert the PIL image to a tkinter PhotoImage
        wind_photo_image = ImageTk.PhotoImage(wind_image)

        # Overlay the wind vectors image on the canvas
        self.canvas.create_image(0, 0, image=wind_photo_image, anchor='nw')

        # Keep a reference to the PhotoImage to prevent it from being garbage collected
        self.wind_photo_image = wind_photo_image

    def on_mouse_move(self, event):
        # Check if self.current_terrain is None
        if self.current_terrain is None:
            print("Warning: current_terrain is None")
            return

        # Calculate the tile's index based on the mouse's position
        tile_x = int(event.x // self.TILE_SIZE)
        tile_y = int(event.y // self.TILE_SIZE)

        # Ensure tile_x and tile_y do not exceed the size of the terrain
        tile_x = min(tile_x, self.current_terrain.shape[1] - 1)
        tile_y = min(tile_y, self.current_terrain.shape[0] - 1)

        # Retrieve the tile's elevation and truncate to 1 decimal place
        elevation = "{:.1f}".format(self.current_terrain[tile_y][tile_x])

        # Calculate the tile's latitude and longitude
        latitude = round(90 - tile_y / self.current_terrain.shape[0] * 180, 2)  # Adjusted latitude calculation
        longitude = round(tile_x / self.current_terrain.shape[1] * 360 - 180, 2)

        # Calculate the altitude and truncate to 1 decimal place
        altitude = np.around(max(0, float(elevation) - 394.66), 1)

        # Retrieve the grayscale pixel value
        grayscale_terrain = scale_to_grayscale(self.current_terrain)
        grayscale_pixel_value = "{:.1f}".format(grayscale_terrain[tile_y, tile_x])

        # Convert self.temperatures to a 2D array
        self.temperatures = np.squeeze(self.temperatures)

        # Retrieve the tile's temperature and truncate to 1 decimal place
        if self.temperatures is not None and len(self.temperatures.shape) > 1 and tile_y < self.temperatures.shape[
            0] and tile_x < self.temperatures.shape[1]:
            temperature = "{:.1f}".format(self.temperatures[tile_y][tile_x])
        else:
            temperature = "N/A"

        # Convert the temperature from kelvin to celsius
        if temperature != "N/A":
            temperature = round(float(temperature) - 273.15, 2)

        # Update the window title with the tile's information
        self.window.title(
            f"Elevation: {elevation}, Altitude: {altitude}, Latitude: {latitude}, Longitude: {longitude}, "
            f"Grayscale Pixel Value: {grayscale_pixel_value}, Temperature: {temperature}")

    def on_mouse_press(self, event):
        # Check if the zoom checkbox is selected
        if self.zoom_var.get() == 1:
            # Update the zoom canvas
            self.update_zoom_canvas(event.x, event.y)

            # Calculate the position of zoom_canvas such that its center is at the mouse position
            zoom_canvas_x = event.x - self.zoom_canvas.winfo_width() / 2
            zoom_canvas_y = event.y - self.zoom_canvas.winfo_height() / 2

            # Set the position of zoom_canvas
            self.zoom_canvas.place(x=(zoom_canvas_x + 250), y=(zoom_canvas_y + 25))

            # Print the coordinates
            print(f"Zoom canvas center: ({event.x - self.zoom_canvas.winfo_width() / 2}, {event.y - self.zoom_canvas.winfo_height() / 2})")
            print(f"Mouse position: ({event.x}, {event.y})")
        else:
            # Hide the zoom canvas
            self.zoom_canvas.place_forget()

    def on_mouse_release(self, event):
        # Hide the zoom canvas
        self.zoom_canvas.place_forget()

    def update_zoom_canvas(self, x, y):
        # Define the size of the zoomed view
        zoom_size = 200  # Change this to the desired size

        # Define the zoom factor (how much closer the zoomed view should be)
        zoom_factor = 2

        # Calculate the size of the extracted part of the terrain
        extract_size = zoom_size // zoom_factor

        # Calculate the coordinates of the top-left corner of the extracted part
        extract_x = max(0, x - extract_size // 2)
        extract_y = max(0, y - extract_size // 2)

        # Ensure the extracted part does not exceed the size of the terrain
        extract_x = min(extract_x, self.current_terrain.shape[1] - extract_size)
        extract_y = min(extract_y, self.current_terrain.shape[0] - extract_size)

        # Extract the part of the terrain colors that should be displayed in the zoomed view
        extracted_terrain_colors = self.terrain_colors[extract_y:extract_y + extract_size,
                                   extract_x:extract_x + extract_size]

        # Convert the colors from RGBA to RGB and scale to [0, 255]
        colors = (extracted_terrain_colors[:, :, :3] * 255).astype(np.uint8)

        # Create an image from the colors array
        extracted_image = Image.fromarray(colors)

        # Resize the image to the size of the zoom box
        zoomed_image = extracted_image.resize((zoom_size, zoom_size), Image.NEAREST)

        # Convert the PIL image to a Tkinter PhotoImage
        photo_image = ImageTk.PhotoImage(zoomed_image)

        # Clear the zoom canvas
        self.zoom_canvas.delete('all')

        # Draw the PhotoImage on the zoom canvas
        self.zoom_canvas.create_image(0, 0, image=photo_image, anchor='nw')

        # Keep a reference to the PhotoImage to prevent it from being garbage collected
        self.zoom_photo_image = photo_image

    def save_settings(self, entries):
        settings = {param: float(entry.get()) for param, entry in entries.items() if entry.winfo_exists()}
        print("Settings to save:", settings)  # Debug print
        with open('settings.json', 'w') as f:
            json.dump(settings, f, indent=4)

    def exit_program(self, window, entries):
        print("Exit button clicked")  # Debug print
        self.save_settings(entries)
        print("Settings saved")  # Debug print
        window.destroy()

    def display_loaded_image(self):
        self.elevation_matrix = self.load_terrain(
            r"D:\dev\planetsim\images\16_bit_dem_small_1280.tif", -420, 8848)
        self.current_terrain = self.elevation_matrix
        self.latitude = np.linspace(-90, 90, self.current_terrain.shape[0])

        self.ocean_map = PlanetSim.classify_ocean(self.current_terrain)
        self.ocean_map = self.remove_inland_oceans(self.ocean_map)
        self.ocean_map = self.load_water_mask("D:\dev\planetsim\images\water_8k.png")

        # Replace multiprocessing with a list comprehension
        self.latitudes = calculate_latitudes(self.current_terrain)
        args = [(self.current_terrain, self.current_terrain.shape, lat) for lat in self.latitudes]
        self.temperatures = np.array([calculate_temperature(arg) for arg in args])

        # Call the update_terrain_display function with the loaded terrain image
        self.update_terrain_display(self.elevation_matrix, self.ocean_map)

        # Create a sidebar with checkboxes if it's not already displayed
        if not self.sidebar.winfo_ismapped():
            self.sidebar.pack(fill='y', side='left', padx=5, pady=5)

            # Create variables for the checkboxes
            self.ocean_var = tk.IntVar()
            self.zoom_var = tk.IntVar()

            # Create the checkboxes
            ocean_check = tk.Checkbutton(self.sidebar, text="Ocean", variable=self.ocean_var)
            zoom_check = tk.Checkbutton(self.sidebar, text="Zoom", variable=self.zoom_var)

            # Pack the checkboxes
            ocean_check.pack()
            zoom_check.pack()

    def main(self):
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.zoom_canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        self.zoom_canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        self.canvas.pack(side='right', expand=True)
        menubar = Menu(self.window)
        view_menu = Menu(menubar, tearoff=0)
        view_menu.add_command(label="Load Terrain Image", command=self.display_loaded_image)
        view_menu.add_command(label="Random Terrain Generation", command=self.generate_random_terrain)
        view_menu.add_command(label="Temperature Map",
                              command=lambda: self.view_temperature() if self.temperatures is not None else print(
                                  "Temperature data not available"))
        view_menu.add_command(label="Wind Map", command=self.visualize_wind_model)
        menubar.add_cascade(label="View", menu=view_menu)
        self.window.config(menu=menubar)
        self.sidebar, self.entries = self.create_sidebar(self.update_frame)
        self.params = self.load_settings()
        self.display_loaded_image()
        self.visualize_wind_model()
        self.window.mainloop()


if __name__ == "__main__":
    sim = PlanetSim()
    sim.main()