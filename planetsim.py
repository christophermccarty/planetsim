from generation5 import (regenerate_map_with_slider_values, generate_terrain, generate_temperature_map,
                         render_terrain_map, render_temperature_map)
import tkinter as tk
import cProfile, pstats, io


# # Cprofile code for later use
# # Place at top of code
# pr = cProfile.Profile()
# pr.enable()
# # Place at end of code
# pr.disable()
# s = io.StringIO()
# sortby = 'tottime'
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# ps.print_stats(10)
# print(s.getvalue())

tiles = []


def create_sliders(master):
    # Default values for sliders
    default_values = {
        "warp_factor": 0.2,
        "num_octaves": 20,
        "persistence": 0.4,
        "lacunarity": 1.9,
        "amplitude": 0.85,  # Default value for amplitude
        "frequency": 3.0,  # Default value for frequency
        "scale": 0.5  # Default value for scale
    }

    """Create sliders for the specified values and return a dictionary of them."""
    sliders = {}

    # Define properties for each slider
    slider_properties = {
        "warp_factor": {"from_": 0.1, "to": 1, "resolution": 0.01, "label": "Warp Factor"},
        "num_octaves": {"from_": 1, "to": 20, "resolution": 1, "label": "Number of Octaves"},
        "persistence": {"from_": 0.2, "to": 0.5, "resolution": 0.01, "label": "Persistence"},
        "lacunarity": {"from_": 1, "to": 3.0, "resolution": 0.01, "label": "Lacunarity"},
        "amplitude": {"from_": 0.1, "to": 2.0, "resolution": 0.01, "label": "Amplitude"},
        "frequency": {"from_": 1.0, "to": 5, "resolution": 0.1, "label": "Frequency"},
        "scale": {"from_": 0.01, "to": 2.0, "resolution": 0.01, "label": "Scale"}
    }

    for key, props in slider_properties.items():
        slider = tk.Scale(master, orient=tk.HORIZONTAL, **props)
        slider.pack(padx=10, pady=5, fill=tk.X, side=tk.TOP)
        sliders[key] = slider

        # Set the default value for the slider
        slider.set(default_values[key])

    return sliders


def open_terrain_window(root, canvas, window_width, window_height, tiles_count):
    terrain_window = tk.Toplevel(root)
    terrain_window.title('Terrain Settings')

    # Update the main window's geometry before calculating the position
    root.update_idletasks()

    x_position = root.winfo_x() + root.winfo_width() + 4
    y_position = root.winfo_y()
    terrain_window.geometry(f"+{x_position}+{y_position}")

    sliders = create_sliders(terrain_window)
    btn = tk.Button(terrain_window, text="Regenerate", command=lambda: regenerate_map_with_slider_values(
        sliders, canvas, window_width, window_height, tiles_count))
    btn.pack(pady=20, side=tk.BOTTOM)


def generate_new_terrain(canvas):
    # Parameters for terrain generation - these can be customized
    width = 1024
    height = 768
    tiles_count = 10000
    warp_factor = 0.2
    num_octaves = 20
    persistence = 0.4
    lacunarity = 1.9
    amplitude = 0.85
    frequency = 3.0
    scale = 0.5

    global tiles
    tiles.clear()
    # Generating the terrain and storing the tiles
    tiles = generate_terrain(canvas, width, height, tiles_count, warp_factor,
                        num_octaves, persistence, lacunarity, amplitude, frequency, scale)


def main():
    window_width = 1024
    window_height = 768
    tiles_count = 10000

    root = tk.Tk()
    root.title("Terrain Map")
    canvas = tk.Canvas(root, width=window_width, height=window_height)
    # Creating the menu bar
    menu_bar = tk.Menu(root)
    root.config(menu=menu_bar)

    # Adding the File menu
    file_menu = tk.Menu(menu_bar, tearoff=0)
    file_menu.add_command(label="New", command=lambda: generate_new_terrain(canvas))
    file_menu.add_command(label="Open")
    file_menu.add_command(label="Save")
    file_menu.add_command(label="Exit", command=root.quit)
    menu_bar.add_cascade(label="File", menu=file_menu)

    # Adding the View menu
    view_menu = tk.Menu(menu_bar, tearoff=0)
    view_menu.add_command(label="Terrain", command=lambda: (render_terrain_map(tiles, canvas, window_width,window_height)))
    view_menu.add_command(label="Temperature", command=lambda: (generate_temperature_map(tiles),
                                                                render_temperature_map(tiles, canvas, window_width,
                                                                                       window_height)))
    view_menu.add_command(label="Precipitation")
    view_menu.add_command(label="Biomes")
    view_menu.add_command(label="Plate Tectonics")
    view_menu.add_command(label="Wind")
    menu_bar.add_cascade(label="View", menu=view_menu)

    # Adding the Settings menu
    settings_menu = tk.Menu(menu_bar, tearoff=0)
    settings_menu.add_command(label="Preferences")
    settings_menu.add_command(label="Terrain", command=lambda: open_terrain_window(
        root, canvas, window_width, window_height, tiles_count))  # Wrapped with lambda
    menu_bar.add_cascade(label="Settings", menu=settings_menu)
    canvas.pack()

    root.mainloop()


if __name__ == "__main__":
    main()