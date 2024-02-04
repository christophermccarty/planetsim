import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tkinter as tk
from PIL import Image
from tkinter import Menu
from generation import (generate_terrain, overlay_large_features, add_crater_optimized, generate_craters,
                        apply_minimal_smoothing, scale_to_grayscale, calculate_temperature)


# Global variable to store the current terrain map
current_terrain = None


def create_sidebar(window, frame_update_function):
    sidebar = tk.Frame(window, width=200, bg='gray')
    sidebar.pack(fill='y', side='left', padx=5, pady=5)

    params = {
        "width": 500,
        "height": 500,
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
        "max_depth": 0.3,
        "sigma": 2
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

    def on_generate_clicked():
        save_settings(entries)
        params = load_settings(entries)  # Load the updated settings
        frame_update_function(window, canvas, params)

    generate_button = tk.Button(sidebar, text="Generate Terrain", command=on_generate_clicked)
    generate_button.pack(pady=5)
    exit_button = tk.Button(sidebar, text="Exit", command=lambda: exit_program(window, entries))
    exit_button.pack(pady=5)

    return sidebar, entries


def update_terrain_display(terrain, canvas):
    # Clear the canvas
    canvas.delete('all')

    # Create a colormap to map terrain heights to colors
    cmap = plt.get_cmap('terrain')

    # Normalize the terrain heights to the range [0, 1]
    terrain_normalized = (terrain - terrain.min()) / (terrain.max() - terrain.min())

    # Create a grid of rectangles on the canvas
    for i in range(terrain.shape[0]):
        for j in range(terrain.shape[1]):
            # Get the color for this tile from the colormap
            color = cmap(terrain_normalized[i, j])

            # Convert the color from RGB to hexadecimal
            color = '#%02x%02x%02x' % (int(color[0]*255), int(color[1]*255), int(color[2]*255))

            # Create a rectangle on the canvas for this tile
            canvas.create_rectangle(j, i, j+1, i+1, fill=color, outline="")

def update_frame(window, canvas, params):
    global current_terrain
    current_terrain = generate_terrain(
        int(params['width']), int(params['height']), scale=params['scale'],
        octaves=int(params['octaves']), persistence=params['persistence'], lacunarity=params['lacunarity']
    )
    if current_terrain is None:
        print("Error: generate_terrain returned None")
        return

    current_terrain = overlay_large_features(current_terrain, strength=params['strength'], scale_factor=params['scale_factor'])
    if current_terrain is None:
        print("Error: overlay_large_features returned None")
        return

    current_terrain = generate_craters(
        current_terrain, num_small_craters=int(params['num_small_craters']),
        num_large_craters=int(params['num_large_craters']),
        max_small_radius=params['max_small_radius'], max_large_radius=params['max_large_radius'],
        max_depth=params['max_depth']
    )
    if current_terrain is None:
        print("Error: generate_craters returned None")
        return

    current_terrain = apply_minimal_smoothing(current_terrain, sigma=params['sigma'])
    if current_terrain is None:
        print("Error: apply_minimal_smoothing returned None")
        return

    update_terrain_display(current_terrain, canvas)


def create_image(terrain):
    terrain_scaled = scale_to_grayscale(terrain).astype('uint8')
    img = Image.fromarray(terrain_scaled, 'L')
    return img


def save_settings(entries):
    settings = {param: float(entry.get()) for param, entry in entries.items() if entry.winfo_exists()}
    print("Settings to save:", settings)  # Debug print
    with open('settings.json', 'w') as f:
        json.dump(settings, f, indent=4)

def exit_program(window, entries):
    print("Exit button clicked")  # Debug print
    save_settings(entries)
    print("Settings saved")  # Debug print
    window.destroy()


def load_settings(entries):
    try:
        with open('settings.json', 'r') as f:
            settings = json.load(f)
    except FileNotFoundError:
        print("Settings file not found. Using default settings.")
        settings = {}

    # Ensure all necessary keys are present in settings
    for param, entry in entries.items():
        if param not in settings:
            print(f"Key '{param}' not found in settings. Using default value.")
            settings[param] = float(entry.get())
        elif entry.winfo_exists():
            entry.delete(0, tk.END)
            entry.insert(0, str(settings[param]))

    return settings


def view_terrain(window, canvas, entries):
    if current_terrain is not None:
        update_terrain_display(current_terrain, canvas)
    else:
        print("No terrain map available.")


def view_temperature(window, canvas, entries):
    params = load_settings(entries)  # Load the updated settings
    # Calculate the temperature of the planet
    base_temperature = calculate_temperature()

    # Create a color map from blue (cooler) to red (hotter)
    cmap = mcolors.LinearSegmentedColormap.from_list("temp_map", ["blue", "red"])

    # Clear the canvas
    canvas.delete('all')

    # Create a grid of rectangles on the canvas
    temperatures = []
    for i in range(1, int(params['height']) - 1):
        for j in range(1, int(params['width']) - 1):
            # Define the sun angle based on the y-coordinate
            sun_angle = np.pi / 2 * (1 - abs(i - params['height'] / 2) / (params['height'] / 2))  # 90 degrees at the equator, decreasing towards the poles

            # Calculate the gradient of the terrain at this location
            gradient_x = current_terrain[i, j+1] - current_terrain[i, j-1]
            gradient_y = current_terrain[i+1, j] - current_terrain[i-1, j]
            terrain_angle = np.arctan(np.sqrt(gradient_x**2 + gradient_y**2))

            # Calculate the angle at which the sun's rays strike the terrain
            strike_angle = max(0, sun_angle - terrain_angle)

            # Adjust the temperature based on the strike angle
            temperature = base_temperature * strike_angle / sun_angle
            temperatures.append(temperature)

    min_temp = min(temperatures)
    max_temp = max(temperatures)

    for i in range(1, int(params['height']) - 1):
        for j in range(1, int(params['width']) - 1):
            # Normalize the temperature to the range [0, 1]
            normalized_temperature = (temperatures[(i-1) * (int(params['width']) - 2) + (j-1)] - min_temp) / (max_temp - min_temp)

            # Get the color for this tile from the colormap
            color = cmap(normalized_temperature)

            # Convert the color from RGB to hexadecimal
            color = '#%02x%02x%02x' % (int(color[0]*255), int(color[1]*255), int(color[2]*255))

            # Create a rectangle on the canvas for this tile
            canvas.create_rectangle(j, i, j+1, i+1, fill=color, outline="")


def main():
    window = tk.Tk()
    window.title("Planet Simulator")
    # Create a menu bar
    menubar = Menu(window)
    # Create a view menu
    view_menu = Menu(menubar, tearoff=0)
    sidebar, entries = create_sidebar(window, update_frame)
    global canvas  # This global is necessary to prevent the canvas from being garbage collected
    canvas = tk.Canvas(window, width=500, height=500)  # Set initial size, it will be updated later
    canvas.pack(side='right', expand=True)
    view_menu.add_command(label="Terrain", command=lambda: view_terrain(window, canvas, entries))
    view_menu.add_command(label="Temperature", command=lambda: view_temperature(window, canvas, entries))
    # Add the view menu to the menu bar
    menubar.add_cascade(label="View", menu=view_menu)
    # Add the menu bar to the window
    window.config(menu=menubar)
    params = load_settings(entries)  # Load settings from the JSON file
    update_frame(window, canvas, params)  # Generate initial terrain after loading settings
    window.mainloop()


if __name__ == "__main__":
    main()