__author__ = "blu3crab"
__license__ = "Apache License 2.0"
__version__ = "0.0.1"

import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
print("Setup Complete")

# Fill in the line below: Specify the path of the CSV file to read
my_filepath = "../../data/covid19-in-usa/us_states_covid19_daily.csv"

# Fill in the line below: Read the file into a variable my_data
my_data = pd.read_csv(my_filepath, index_col="date", parse_dates=True)

# total tested positive & negative by state
plt.figure(figsize=(12, 24))

# Add title
plt.title("total tested positive & negative")

# Bar chart showing average
# sns.barplot(x=ign_data.index, y=ign_data['Racing'])
sns.barplot(y=my_data.state, x=my_data['total'])

# Add label for vertical axis
plt.ylabel("state")
plt.show()

sns.scatterplot(x=my_data['total'], y=my_data['death'])

# Scatter plot w/ regression line showing the relationship between 'total' tested and 'death'
# positive correlation: as testing increases, deaths increase
sns.regplot(x=my_data['total'], y=my_data['death'])
plt.show()

# Scatter plot showing the relationship between 'pricepercent', 'winpercent', and 'chocolate'
sns.scatterplot(x=my_data['total'], y=my_data['hospitalized'], hue=my_data['death'])
plt.show()

print("Adios Amigos!")