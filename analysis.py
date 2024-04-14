import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Load the dataset of stars
stars_df = pd.read_csv('1712727135283O-result.csv')

# Load the dataset of stars known to host exoplanets
exoplanet_hosts_df = pd.read_csv('recognized_stars_with_gaia_ids.csv')

# Clean the Gaia_ID column in exoplanet_hosts_df to ensure it matches the format in stars_df
exoplanet_hosts_df['Gaia_ID'] = exoplanet_hosts_df['Gaia_ID'].str.split().str[-1]

# Convert Gaia_ID in exoplanet_hosts_df to integer if not already
exoplanet_hosts_df['Gaia_ID'] = exoplanet_hosts_df['Gaia_ID'].astype(str)

# Mark stars in stars_df that are known to host exoplanets
stars_df['has_exoplanet'] = stars_df['source_id'].astype(str).isin(exoplanet_hosts_df['Gaia_ID'])
filtered_stars = stars_df[stars_df['parallax'] > 5]
# Sky coverage analysis
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(filtered_stars['ra'], filtered_stars['dec'], s=1, alpha=0.5, label='All Stars parallax > 5')
plt.scatter(filtered_stars[filtered_stars['has_exoplanet']]['ra'], filtered_stars[filtered_stars['has_exoplanet']]['dec'], s=1, alpha=0.75, color='red', label='Exoplanet Hosts')
#plt.scatter(stars_df['ra'], stars_df['dec'], s=1, alpha=0.5, label='All Stars')
#plt.scatter(stars_df[stars_df['has_exoplanet']]['ra'], stars_df[stars_df['has_exoplanet']]['dec'], s=1, alpha=0.75, color='red', label='Exoplanet Hosts')
plt.xlabel('Right Ascension')
plt.ylabel('Declination')
plt.title('Sky Coverage')
plt.legend()

# Metallicity distribution
plt.subplot(1, 2, 2)
plt.hist(filtered_stars['metallicity'], bins=30, alpha=0.5, label='All Stars parallax>5', edgecolor='black')
#plt.hist(stars_df['metallicity'], bins=30, alpha=0.5, label='All Stars', edgecolor='black')
#plt.hist(stars_df[stars_df['has_exoplanet']]['metallicity'], bins=30, alpha=0.75, color='red', label='Exoplanet Hosts')
plt.xlabel('Metallicity [Fe/H]')
plt.ylabel('Count')
plt.title('Metallicity Distribution')
plt.legend()

plt.tight_layout()
plt.show()

# Note: Update 'path/to/your/' to the actual path of your CSV files


from mpl_toolkits.mplot3d import Axes3D

# Calculate distance in parsecs (pc) for simplicity; distance = 1 / parallax (in arcseconds)
stars_df['distance_pc'] = 1000 / stars_df['parallax']

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plotting only exoplanet hosts for clarity
exoplanet_hosts = stars_df[stars_df['has_exoplanet'] & (stars_df['parallax'] > 5)]
ax.scatter(exoplanet_hosts['ra'], exoplanet_hosts['dec'], exoplanet_hosts['distance_pc'], color='red', alpha=0.6, s=10)

ax.set_xlabel('Right Ascension')
ax.set_ylabel('Declination')
ax.set_zlabel('Distance (pc)')
ax.set_title('3D Distribution of Exoplanet-Hosting Giants')

plt.show()


import seaborn as sns

# Calculating correlation matrix for exoplanet hosts
corr_matrix = exoplanet_hosts[['teff', 'luminosity', 'logg', 'metallicity', 'distance_pc']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap for Exoplanet-Hosting Giants')
plt.show()

exoplanet_hosts['distance'] = 1000 / exoplanet_hosts['parallax']  # Parallax in milliarcseconds to distance in parsecs

# Plotting the histogram of distances for exoplanet-hosting stars
plt.figure(figsize=(10, 6))
plt.hist(exoplanet_hosts['distance'], bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.xlabel('Distance (parsecs)')
plt.ylabel('Number of Exoplanet-Hosting Stars')
plt.title('Distance Distribution of Exoplanet-Hosting Giants')
plt.show()


distance_filtered_hosts = exoplanet_hosts[(exoplanet_hosts['distance_pc'] >= 100) & (exoplanet_hosts['distance_pc'] <= 115)]

# Sky coverage plot for these stars
plt.figure(figsize=(10, 5))
plt.scatter(distance_filtered_hosts['ra'], distance_filtered_hosts['dec'], s=10, alpha=0.5)
plt.xlabel('Right Ascension')
plt.ylabel('Declination')
plt.title('Sky Coverage of Exoplanet Hosts (100-115 pc)')
plt.show()

# Metallicity histogram for these stars
plt.figure(figsize=(10, 5))
plt.hist(distance_filtered_hosts['metallicity'], bins=20, alpha=0.5, edgecolor='black')
plt.xlabel('Metallicity [Fe/H]')
plt.ylabel('Count')
plt.title('Metallicity Distribution of Exoplanet Hosts (100-115 pc)')
plt.show()

# HR diagram for these stars
plt.figure(figsize=(10, 5))
plt.scatter(distance_filtered_hosts['teff'], np.log10(distance_filtered_hosts['luminosity']), s=10, alpha=0.5)
plt.gca().invert_xaxis()  # HR diagrams typically have the higher temperatures on the left
plt.xlabel('Effective Temperature (K)')
plt.ylabel('log(L/L_sun)')
plt.title('HR Diagram of Exoplanet Hosts (100-115 pc)')
plt.show()
