__author__ = "blu3crab"
__license__ = "Apache License 2.0"
__version__ = "0.0.1"

import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
print("Setup Complete")

# Paths of the files to read
cancer_b_filepath = "../data/cancer_b.csv"
cancer_m_filepath = "../data/cancer_m.csv"

# Fill in the line below to read the (benign) file into a variable cancer_b_data
cancer_b_data = pd.read_csv(cancer_b_filepath, index_col="Id")

# Fill in the line below to read the (malignant) file into a variable cancer_m_data
cancer_m_data = pd.read_csv(cancer_m_filepath, index_col="Id")

cancer_b_data.head()
cancer_m_data.head()

# Histograms for benign and maligant tumors
sns.distplot(a=cancer_b_data['Area (mean)'], label="Benign", kde=False)
sns.distplot(a=cancer_m_data['Area (mean)'], label="Malignant", kde=False)

# Add title
plt.title("Histogram of Area by Benign and Malignant")

# Force legend to appear
plt.legend()
plt.show()

# KDE plots for benign and malignant tumors
# KDE Plot described as Kernel Density Estimate is used for visualizing
# the Probability Density of a continuous variable. It depicts the probability density at
# different values in a continuous variable.
sns.kdeplot(data=cancer_b_data['Radius (worst)'], label="Benign", shade=True)
sns.kdeplot(data=cancer_m_data['Radius (worst)'], label="Malignant", shade=True)

# Add title
plt.title("KDE Histogram of Radius (worst) by Benign and Malignant")

# Force legend to appear
plt.legend()
plt.show()

print("Adios Amigos!")
