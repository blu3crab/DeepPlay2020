__author__ = "blu3crab"
__license__ = "Apache License 2.0"
__version__ = "0.0.1"

import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
print("Setup Complete")

# Path of the file to read
ign_filepath = "../data/ign_scores.csv"

# Fill in the line below to read the file into a variable ign_data
ign_data = pd.read_csv(ign_filepath, index_col="Platform")

# Print the data
print(ign_data)

# Bar chart showing average score for racing games by platform
# Set the width and height of the figure
plt.figure(figsize=(12, 6))

# Add title
plt.title("average score for racing games by platform")

# Bar chart showing average
# sns.barplot(x=ign_data.index, y=ign_data['Racing'])
sns.barplot(y=ign_data.index, x=ign_data['Racing'])

# Add label for vertical axis
plt.ylabel("Platform")
plt.show()

# Heatmap showing average game score by platform and genre
# Set the width and height of the figure
plt.figure(figsize=(14, 7))

# Add title
plt.title("Average score by genre & platform")

# Heatmap showing average score by genre & paltform
sns.heatmap(data=ign_data, annot=True)

# Add label for horizontal axis
plt.xlabel("platform")
plt.show()

print("!adios amigos!")