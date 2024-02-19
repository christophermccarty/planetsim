import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tkinter as tk
from tqdm import tqdm
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import Menu
from generation import (generate_terrain, overlay_large_features, add_crater_optimized, generate_craters,
                        apply_minimal_smoothing, scale_to_grayscale, calculate_temperature, apply_atmospheric_model,
                        simulate_oceans, flood_fill)


class PlanetSim:
    def __init__(self):
        self.current_terrain = None
        self.temperatures = None
        self.TILE_SIZE = 10
        self.canvas_size = 500
        self.terrain_size = 500
        self.TILE_SIZE = self.canvas_size / self.terrain_size
        self.window = tk.Tk()
        self.window.title("Planet Simulator")
        self.canvas = tk.Canvas(self.window, width=500, height=500)
        self.info_label = tk.Label(self.window)
        self.info_label.pack()
        self.current_ocean_map = None

    def create_sidebar(self, frame_update_function):
        sidebar = tk.Frame(self.window, width=200, bg='gray')

        params = {
            "width": 500,
            "height": 500,
            "sigma": 2
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
            entry.pack(side='right', expand=True)
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
            self.params = self.load_settings(entries)  # Load the updated settings
            frame_update_function(self.params)

        generate_button = tk.Button(sidebar, text="Generate Terrain", command=lambda: on_generate_clicked(self))
        generate_button.pack(pady=5)
        exit_button = tk.Button(sidebar, text="Exit", command=lambda: self.exit_program(self.window, entries))
        exit_button.pack(pady=5)

        return sidebar, entries

    def show_terrain_entries(self):
        self.sidebar.pack(fill='y', side='left', padx=5, pady=5)  # Pack the sidebar here
        for frame in self.terrain_entries.values():
            frame.pack()  # Show the terrain entries

        # Display the terrain map on the canvas
        if self.current_terrain is not None:
            ocean_map = simulate_oceans(self.current_terrain)
            self.update_terrain_display(self.current_terrain, ocean_map)
        else:
            print("No terrain map available.")

    def generate_terrain_from_menu(self):
        # Generate the terrain
        print("Generating terrain...")
        self.current_terrain = generate_terrain(
            int(self.params['width']), int(self.params['height']), scale=self.params['scale'],
            octaves=int(self.params['octaves']), persistence=self.params['persistence'], lacunarity=self.params['lacunarity']
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

        # Update the terrain display
        self.update_terrain_display(self.current_terrain)

    def load_image_as_terrain(self, image_path):
        # Load the image
        print("Loading image...")
        image = Image.open(image_path).convert('L')
        image_np = np.array(image)
        self.original_image = np.array(image)  # Store the original grayscale image

        # Set the canvas size to match the image size
        self.canvas_size = image_np.shape

        # Print a table of pixel values and their count
        unique, counts = np.unique(image_np, return_counts=True)
        print("Pixel values and their count:" + "\n" + str(np.asarray((unique, counts)).T) + "\n")

        # Subtract 45 from all pixel values in the image
        image_np_adjusted = image_np - 45

        # Set all terrain color values that are below pixel values of 45 to 0
        image_np_adjusted[image_np < 45] = 0

        # Normalize the adjusted pixel values to the range [0, max_height]
        print("Normalizing grayscale values...")
        min_height = 0
        max_height = 8848
        min_pixel_value = image_np_adjusted.min()
        max_pixel_value = image_np_adjusted.max()
        image_np_scaled = ((image_np_adjusted - min_pixel_value) / (max_pixel_value - min_pixel_value)) * max_height

        # Convert the scaled grayscale values to elevations
        self.current_terrain = image_np_scaled
        print("Loading terrain...")
        print("Elevation range:", self.current_terrain.min(), self.current_terrain.max())

        # Generate the ocean map
        ocean_map = simulate_oceans(self.current_terrain, min_height)

        # Display the terrain map on the canvas
        self.update_terrain_display(self.current_terrain, ocean_map)

        # Calculate the latitude for each row in the terrain
        latitudes = np.linspace(-90, 90, self.current_terrain.shape[0])

        # Generate the initial temperatures
        #self.temperatures = calculate_temperature(self.current_terrain, latitudes[:, np.newaxis], ocean_map)

        # Update the canvas's width and height to match the canvas size
        self.canvas.config(width=self.canvas_size[1], height=self.canvas_size[0])

    def update_terrain_display(self, terrain, ocean_map):
        # Create a colormap to map terrain heights to colors
        cmap = plt.get_cmap('terrain')

        # Create an empty array for the colors
        colors = np.zeros((terrain.shape[0], terrain.shape[1], 3), dtype=np.uint8)

        # Calculate the colors for the terrain
        for i in tqdm(range(terrain.shape[0]), desc='Generating terrain colors'):
            for j in range(terrain.shape[1]):
                # Check if the tile is an ocean tile
                if ocean_map[i, j] or terrain[i, j] < 0:
                    colors[i, j] = [0, 0, 139]  # Dark blue color for ocean tiles
                else:
                    # Get the color for this tile from the colormap
                    color = cmap(terrain[i, j] / 8848)  # use elevation value for colormap

                    # Convert the color from RGB to hexadecimal
                    colors[i, j] = [int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)]

        # Create an image from the colors array
        image = Image.fromarray(colors)

        # Create a PhotoImage from the image
        photo_image = ImageTk.PhotoImage(image)

        # Clear the canvas
        self.canvas.delete('all')

        # Draw the PhotoImage on the canvas
        self.canvas.create_image(0, 0, image=photo_image, anchor='nw')

        # Keep a reference to the PhotoImage to prevent it from being garbage collected
        self.photo_image = photo_image

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

        # Calculate the latitude for each row in the terrain
        latitudes = np.linspace(-90, 90, self.current_terrain.shape[0])

        # Calculate the temperature
        self.temperatures = calculate_temperature(self.current_terrain, latitudes[:, np.newaxis])
        if self.temperatures is None:
            print("Error: calculate_temperature returned None")
            return

        # Update the terrain display
        self.update_terrain_display(self.current_terrain)

        return self.temperatures

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

        # Ensure all necessary keys are present in settings
        for param, entry in self.entries.items():
            if param not in settings:
                print(f"Key '{param}' not found in settings. Using default value.")
                settings[param] = float(entry.get())
            elif entry.winfo_exists():
                entry.delete(0, tk.END)
                entry.insert(0, str(settings[param]))

        return settings

    def view_terrain(self):
        if self.current_terrain is not None and self.current_ocean_map is not None:
            self.update_terrain_display(self.current_terrain, self.current_ocean_map)
        else:
            print("No terrain map available.")

    def view_temperature(self, window, canvas):
        # Ensure self.temperatures is not None
        if self.temperatures is None:
            self.temperatures = self.update_frame(self.params)
            if self.temperatures is None:
                print("Error: Failed to generate temperatures")
                return

        # Create a color map from blue (cooler) to red (hotter)
        cmap = plt.get_cmap('coolwarm')

        # Clear the canvas
        canvas.delete('all')

        # Normalize the temperatures to the range [0, 1]
        normalized_temperatures = (self.temperatures - self.temperatures.min()) / (
                self.temperatures.max() - self.temperatures.min())

        # Convert the normalized temperatures to colors
        rgba_colors = cmap(normalized_temperatures)

        # Flatten the rgba_colors array
        flattened_rgba_colors = rgba_colors.reshape(-1, rgba_colors.shape[-1])

        # Convert the RGBA colors to hexadecimal color strings
        hex_colors = ['#%02x%02x%02x' % (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)) for color in
                      flattened_rgba_colors]

        # Reshape hex_colors to match the original shape of rgba_colors
        hex_colors = np.array(hex_colors).reshape(rgba_colors.shape[0], rgba_colors.shape[1])

        # Draw the tiles on the canvas
        for i in tqdm(range(self.current_terrain.shape[0]), desc='Drawing tiles'):
            for j in range(self.current_terrain.shape[1]):
                canvas.create_rectangle(j, i, j + 1, i + 1, fill=hex_colors[i][j], outline="")

    def on_mouse_move(self, event):
        # Calculate the tile's index based on the mouse's position
        tile_x = int(event.x // self.TILE_SIZE)
        tile_y = int(event.y // self.TILE_SIZE)

        # Ensure tile_x and tile_y do not exceed the size of the terrain
        tile_x = min(tile_x, self.current_terrain.shape[1] - 1)
        tile_y = min(tile_y, self.current_terrain.shape[0] - 1)

        # Retrieve the tile's temperature and elevation
        #temperature = round(self.temperatures[tile_y][tile_x], 2)
        elevation = round(self.current_terrain[tile_y][tile_x], 2)  # self.current_terrain already contains elevations

        # Calculate the tile's latitude and longitude
        latitude = round(tile_y / self.current_terrain.shape[0] * 180 - 90, 2)
        longitude = round(tile_x / self.current_terrain.shape[1] * 360 - 180, 2)

        # Retrieve the grayscale pixel value
        grayscale_pixel_value = self.original_image[tile_y][tile_x]

        # Update the window title with the tile's information
        # self.window.title(
        #     f"Temperature: {temperature}, Elevation: {elevation}, Latitude: {latitude}, Longitude: {longitude}, "
        #     f"Grayscale Pixel Value: {grayscale_pixel_value}")

        self.window.title(
            f"Elevation: {elevation}, Latitude: {latitude}, Longitude: {longitude}, "
            f"Grayscale Pixel Value: {grayscale_pixel_value}")

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

    def main(self):
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.pack(side='right', expand=True)
        # Create a menu bar
        menubar = Menu(self.window)
        # Create a view menu
        view_menu = Menu(menubar, tearoff=0)
        view_menu.add_command(label="Terrain", command=self.show_terrain_entries)
        view_menu.add_command(label="New Terrain", command=self.generate_terrain_from_menu)
        view_menu.add_command(label="Temperature", command=lambda: self.view_temperature(self.window, self.canvas))
        # Add the view menu to the menu bar
        menubar.add_cascade(label="View", menu=view_menu)
        # Add the menu bar to the window
        self.window.config(menu=menubar)
        self.sidebar, self.entries = self.create_sidebar(self.update_frame)  # Initialize self.entries and self.sidebar
        self.params = self.load_settings()  # Load settings from the JSON file
        # Load the image as the tilemap
        self.load_image_as_terrain(r"D:\dev\planetsim\images\16_bit_dem_small_earth_1280.png")
        # Display the terrain map on the canvas
        self.view_terrain()
        self.window.mainloop()


if __name__ == "__main__":
    sim = PlanetSim()
    sim.main()