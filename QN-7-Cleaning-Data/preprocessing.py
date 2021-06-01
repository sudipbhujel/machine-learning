# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Creating dataset
np.random.seed(10)
data = np.random.normal(100, 20, 200)
noise1 = np.random.randint(100, size=(10))
noise2 = np.random.randint(200, size=(10))

# Add noise to normal variable
data = np.concatenate([data, noise1, noise2])

# Histogram plot
plt.hist(data, bins=20)
plt.title("Histogram of Uncleaned Data")
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.savefig('histogram-uncleaned-data.jpg', dpi=500)
plt.show()

# Box plot of uncleaned data
plt.boxplot(data)
plt.title("Box plot of cleaned data")
plt.savefig('box-plot-uncleaned-data.jpg', dpi=500)
plt.show()

# Prepare Dataframe
df = pd.DataFrame(data)[0]

# Define maximum threshold
mx_threshold = df.quantile(0.95)
print(mx_threshold)
df[df > mx_threshold]

# Define minimum threshold
mn_threshold = df.quantile(0.05)
print(mn_threshold)
df[df < mn_threshold]

# Remove noise from data
cleaned_data = df[(df < mx_threshold) & (df > mn_threshold)]

# Histogram plot of Cleaned data
plt.hist(cleaned_data, bins=20)
plt.title("Histogram of Cleaned Data")
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.savefig('histogram-cleaned-data.jpg', dpi=500)
plt.show()

# Box plot of cleaned data
plt.boxplot(cleaned_data)
plt.title("Box plot of cleaned data")
plt.savefig('box-plot-cleaned-data.jpg', dpi=500)
plt.show()
