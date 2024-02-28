import numpy as np
from opensimplex import OpenSimplex
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from scipy.optimize import fsolve
from scipy.interpolate import interp1d


# Constants for atmospheric composition
NITROGEN = 0.78084
OXYGEN = 0.20946
ARGON = 0.00934
CARBON_DIOXIDE = 0.000413
NEON = 0.00001818
HELIUM = 0.00000524
METHANE = 0.00000179
KRYPTON = 0.00000114
HYDROGEN = 0.00000055
NITROUS_OXIDE = 0.00000032
XENON = 0.000000009
OZONE = 0.000000004
NITROGEN_DIOXIDE = 0.000000002
IODINE = 0.000000001
CARBON_MONOXIDE = 0.000000001

# Constants for radiative forcing of greenhouse gases (W/m^2)
WATER_VAPOR_RF = 0.15  # Estimated value, actual value varies greatly
CARBON_DIOXIDE_RF = 1.82
METHANE_RF = 0.48
NITROUS_OXIDE_RF = 0.17
OZONE_RF = 0.35  # Tropospheric Ozone

# Constants for albedo
ALBEDO_LAND = 0.3  # Albedo of land
ALBEDO_WATER = 0.06  # Albedo of water


def generate_terrain(width, height, seed=None, scale=400.0, octaves=1, persistence=0.7, lacunarity=2.0):
    if seed is None:
        seed = np.random.randint(0, 100)
    simplex = OpenSimplex(seed)

    max_amp = 0
    amp = 1
    freq = 3

    terrain = np.zeros((height, width))

    # Wrap the outer loop with tqdm for a progress bar
    for o in tqdm(range(octaves), desc='Generating terrain'):
        for i in range(height):
            for j in range(width):
                terrain[i][j] += simplex.noise2(i / scale * freq, j / scale * freq) * amp
        max_amp += amp
        amp *= persistence
        freq *= lacunarity

    # Normalizing the terrain
    terrain = (terrain + max_amp) / (2 * max_amp)

    # Scale the normalized terrain values to the range [0, 20000]
    terrain *= 20000

    return terrain


def overlay_large_features(terrain, seed=None, strength=0.5, scale_factor=4):
    if seed is None:
        seed = np.random.randint(0, 100)
    simplex = OpenSimplex(seed)
    height, width = terrain.shape
    large_scale = max(width, height) / scale_factor

    for i in range(height):
        for j in range(width):
            terrain[i][j] += simplex.noise2(i / large_scale, j / large_scale) * strength

    return terrain


def add_crater_optimized(terrain, cx, cy, radius, depth):
    height, width = terrain.shape
    y, x = np.ogrid[-cx:height-cx, -cy:width-cy]
    mask = x**2 + y**2 <= radius**2

    # Apply crater effect within the masked area
    distance_from_center = np.sqrt(x**2 + y**2)[mask]
    delta = depth * (1 - distance_from_center / radius)
    terrain[mask] -= delta
    terrain[terrain < 0] = 0  # Ensure terrain height doesn't go below 0


def generate_craters(terrain, num_small_craters=15, num_large_craters=3, max_small_radius=10, max_large_radius=40,
                     max_depth=0.1):
    total_craters = num_small_craters + num_large_craters
    with tqdm(total=total_craters, desc='Adding craters') as pbar:
        # Add small craters
        for _ in range(num_small_craters):
            height, width = terrain.shape
            # Random center, radius, and depth for each crater
            cx, cy = np.random.randint(0, height), np.random.randint(0, width)
            radius = np.random.uniform(0, max_small_radius)
            depth = np.random.uniform(0, max_depth)
            add_crater_optimized(terrain, cx, cy, radius, depth)
            pbar.update()

        # Add large craters
        for _ in range(num_large_craters):
            height, width = terrain.shape
            # Random center, radius, and depth for each crater
            cx, cy = np.random.randint(0, height), np.random.randint(0, width)
            radius = np.random.uniform(0, max_large_radius)
            depth = np.random.uniform(0, max_depth)
            add_crater_optimized(terrain, cx, cy, radius, depth)
            pbar.update()

    return terrain


def apply_minimal_smoothing(terrain, sigma=1):
    return gaussian_filter(terrain, sigma=sigma)


def scale_to_grayscale(arr):
    min_val, max_val = arr.min(), arr.max()
    return (arr - min_val) / (max_val - min_val) * 255


# Create a function to calculate the latitude of each tile
def calculate_latitudes(terrain, step=10):
    # Constants
    R = 6371  # Radius of the Earth (km)

    # Calculate the latitude of each tile for a subset of points
    subset_indices = np.arange(0, terrain.shape[0], step)
    subset_latitudes = 90 - (subset_indices / terrain.shape[0]) * 180

    # Use interpolation to estimate the latitudes for the remaining points
    interpolate = interp1d(subset_indices, subset_latitudes, kind='linear', fill_value="extrapolate")
    latitudes_array = interpolate(np.arange(terrain.shape[0]))

    return latitudes_array


def calculate_temperature(args):
    # Eventually add back in atmospheric temperature generation with
    # atmospheric_temperatures = np.zeros_like(terrain)
    # Tatm_solution = f * Ts_solution
    # atmospheric_temperatures[i][j] = Tatm_solution
    # and return atmospheric_temperatures as well
    terrain, resized_shape, lat = args
    # Constants
    S0 = 1361  # Solar constant (W/m2)
    sigma = 5.67e-8  # Stefan-Boltzmann constant (W/m2/K4)
    A = 0.3  # Albedo
    G = 0.85  # Greenhouse factor
    epsilon = 0.8  # Atmospheric emissivity
    f = 0.7  # Factor to relate Tatm to Ts

    # Create a 2D array of latitudes
    latitudes_2d = np.tile(lat, (resized_shape[1], 1)).T

    # Linear model for temperature based on latitude
    Ts_solutions = 288 - 0.5 * np.abs(latitudes_2d)  # 288 is the average global temperature in K, 0.5 is a scaling factor

    return Ts_solutions