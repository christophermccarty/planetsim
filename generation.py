import numpy as np
from opensimplex import OpenSimplex
from tqdm import tqdm
from scipy.ndimage import gaussian_filter


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


def apply_atmospheric_model(temperatures):
    # Apply the greenhouse effect to each temperature in the map
    return [[calculate_greenhouse_effect(temp) for temp in row] for row in temperatures]


def calculate_greenhouse_effect(temperature, radiative_forcing):
    # The greenhouse effect increases the temperature based on the radiative forcing of greenhouse gases
    return temperature * (1 + radiative_forcing)


def calculate_solar_insolation(latitude, day_of_year, time_of_day):
    # Constants for solar insolation calculation
    SOLAR_CONSTANT = 1361  # Solar constant in W/m^2
    PLANET_TILT = 0  # Tilt of the Earth's axis in degrees

    # Calculate the declination angle
    declination_angle = np.radians(PLANET_TILT * np.cos((day_of_year + 10) / 365.25 * 2 * np.pi))

    # Calculate the hour angle
    hour_angle = np.radians((time_of_day - 12) * 15)

    # Calculate the solar zenith angle
    solar_zenith_angle = np.arccos(np.sin(np.radians(latitude)) * np.sin(declination_angle) +
                                    np.cos(np.radians(latitude)) * np.cos(declination_angle) * np.cos(hour_angle))

    # Calculate the solar insolation
    solar_insolation = SOLAR_CONSTANT * np.cos(solar_zenith_angle)

    return solar_insolation


def calculate_temperature(elevation, latitude, ocean_map, day_of_year, time_of_day):
    # Constants for solar radiation simulation
    BOND_ALBEDO = 0.306  # Albedo of the planet
    STEFAN_BOLTZMANN_CONSTANT = 5.670374419e-8  # Stefan-Boltzmann constant in W/m^2/K^4
    # Constants for sunlight absorption
    SUNLIGHT_ABSORPTION = 0.7

    # Initialize the temperature array
    temperature = np.zeros_like(elevation)

    # Iterate over the elevation array with a progress bar
    for i in tqdm(range(elevation.shape[0]), desc='Calculating temperatures'):
        for j in range(elevation.shape[1]):
            # Calculate the solar insolation
            solar_insolation = calculate_solar_insolation(latitude[i], day_of_year, time_of_day)

            # Adjust the solar insolation based on the albedo of the tile
            if ocean_map[i, j]:
                solar_insolation *= (1 - ALBEDO_WATER)
            else:
                solar_insolation *= (1 - ALBEDO_LAND)

            # Calculate the equilibrium temperature
            equilibrium_temperature = ((solar_insolation * (1 - BOND_ALBEDO)) / (4 * STEFAN_BOLTZMANN_CONSTANT)) ** 0.25

            # Adjust the temperature based on the elevation of the tile
            if elevation[i][j] <= 11000:  # For the troposphere
                temp = equilibrium_temperature - (0.0065 * elevation[i][j])  # Temperature decreases by 6.5 degrees per km of elevation
            else:  # For the stratosphere and above
                temp = equilibrium_temperature - (0.0065 * 11000)  # Temperature is constant

            temp = temp - 273.15  # Convert from Kelvin to Celsius

            # Adjust the temperature based on the greenhouse gas concentration and solar radiation
            radiative_forcing = WATER_VAPOR_RF + CARBON_DIOXIDE_RF + METHANE_RF + NITROUS_OXIDE_RF + OZONE_RF
            temp = calculate_greenhouse_effect(temp, radiative_forcing)

            # Check if the tile is an ocean tile
            if ocean_map[i, j]:
                # Adjust the temperature calculation for ocean tiles
                temp += 2  # Increase the temperature by 2 degrees for ocean tiles

            # Store the calculated temperature
            temperature[i][j] = temp

    return temperature

def flood_fill(i, j, ocean_map, flood_fill_map, terrain_shape):
    # Stack for the tiles to be checked
    stack = [(i, j)]

    while stack:
        i, j = stack.pop()

        if not flood_fill_map[i, j]:
            flood_fill_map[i, j] = True

            # Check the neighboring tiles
            if i > 0 and ocean_map[i - 1, j]:
                stack.append((i - 1, j))
            if j > 0 and ocean_map[i, j - 1]:
                stack.append((i, j - 1))
            if i < terrain_shape[0] - 1 and ocean_map[i + 1, j]:
                stack.append((i + 1, j))
            if j < terrain_shape[1] - 1 and ocean_map[i, j + 1]:
                stack.append((i, j + 1))
