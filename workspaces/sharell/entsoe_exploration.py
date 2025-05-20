# Generate a concise summary of the dataset
def summary(df):
    print("Dataset Summary:")
    print(df.describe(include='all'))  # Summary statistics
    print("\nMissing Values (NA) per Column:")
    print(df.isna().sum())  # Count of missing values

df_hourly = df_hourly.loc['2022-04-01':'2025-04-01']
summary(df_hourly.loc['2022-04-01':'2025-04-01'])

# Find timestamps with NA values for each column
def list_na_timestamps(df):
    na_timestamps = {}
    for column in df.columns:
        na_timestamps[column] = df[df[column].isna()].index.tolist()
    return na_timestamps

# Generate the list of NA timestamps
na_timestamps = list_na_timestamps(df_hourly)

# Print the NA timestamps for each variable
for column, timestamps in na_timestamps.items():
    print(f"Variable: {column}")
    print(f"NA Timestamps: {timestamps[:50]}")  # Print first 10 timestamps for brevity
    print(f"Total NA Count: {len(timestamps)}\n")

    import pandas as pd
import matplotlib.pyplot as plt

# Load the existing CSV file
df = pd.read_csv('electricity_data_nl_2022_2025_hourly_flow.csv', index_col=0, parse_dates=True)
print(df.head())

df_filtered = df.loc['2022-04-01':'2025-04-01']

#df.index.name = 'date'

# Zorg ervoor dat de datumkolom wordt herkend als datetime
#df['date'] = pd.to_datetime(df['date'])

# Ensure the date column is recognized as datetime
df.index = pd.to_datetime(df.index)
# Ensure the index is recognized as datetime
df.index = pd.to_datetime(df.index)

# Create a boxplot of the Price column for the years 2022 and 2023
plt.figure(figsize=(5,3))
plt.boxplot(df_filtered['Price'].dropna(), vert=False)
plt.xlabel('Price')
plt.title('Boxplot of Price')
plt.show()

# Plot the load over the years 2022 and 2023
plt.figure(figsize=(10,5))
plt.plot(df_filtered.index, df_filtered['Price'], label='Price')
plt.xlabel('Year')
plt.ylabel('Price')
plt.title('Price')
plt.show()

df_filtered_price = df_filtered[['Price']].dropna()

# Extract the month and day from the index
df_filtered_price['Month'] = df_filtered_price.index.month
df_filtered_price['Day'] = df_filtered_price.index.day

# Plot the minimum and maximum price per day for each month
plt.figure(figsize=(15, 20))
for month in range(1, 13):
    plt.subplot(4, 3, month)
    month_data = df_filtered_price[df_filtered_price['Month'] == month]
    daily_min_price = month_data.groupby(month_data.index.day)['Price'].min()
    daily_max_price = month_data.groupby(month_data.index.day)['Price'].max()
    plt.plot(daily_min_price.index, daily_min_price, marker='o', linestyle='-', label='Min Price')
    plt.plot(daily_max_price.index, daily_max_price, marker='o', linestyle='-', label='Max Price')
    plt.fill_between(daily_min_price.index, daily_min_price, daily_max_price, color='gray', alpha=0.2)
    plt.xlabel('Day of the Month')
    plt.ylabel('Price')
    plt.title(f'Min and Max Price per Day in Month {month}')
    plt.xticks(range(1, 32))  # Ensure all days are shown on the x-axis
    plt.grid(True)
    plt.ylim(-400,1000)
    plt.legend()

plt.tight_layout()
plt.show()

df_filtered.head

plt.figure(figsize=(5,3))
plt.boxplot(df_filtered['Load'].dropna(), vert=False)
plt.xlabel('Load')
plt.title('Boxplot of Load')
plt.show()

plt.figure(figsize=(10,5))
plt.plot(df_filtered.index, df_filtered['Load'], label='Load')
plt.xlabel('Year')
plt.ylabel('Load')
plt.title('Consumption (Load)')
plt.legend()
plt.show()

# Create a boxplot of the Total_Flow column for the years 2022 and 2023
plt.figure(figsize=(5,3))
plt.boxplot(df_filtered['Total_Flow'].dropna(), vert=False)
plt.xlabel('Total_Flow')
plt.title('Boxplot of Total_Flow')
plt.show()

# Plot the load over the years 2022 and 2023
plt.figure(figsize=(10,5))
plt.plot(df_filtered.index, df_filtered['Total_Flow'], label='Total_Flow')
plt.xlabel('Year')
plt.ylabel('Total_Flow')
plt.title('Total_Flow Over the Years 2022 and 2023')
plt.legend()
plt.show()

plt.scatter(df['Price'], df['Load'], s = 0.1)
plt.title('Price vs Load')
plt.show()

plt.scatter(df['Price'], df['Total_Flow'], s = 0.1)
plt.title('Price vs Total flow')
plt.show()

import matplotlib.pyplot as plt
import pandas as pd

# Voeg een kolom toe voor de maand en het jaar
df = df_filtered.loc['2025-01-01':'2025-04-01']
df['year_month'] = df.index.to_period('W')  # Maand en jaar (bijv. 2022-01)

# Maak een lijst van unieke maanden
unique_months = df['year_month'].unique()

# Bereken het aantal unieke maanden
num_months = len(unique_months)

# Bepaal het aantal rijen en kolommen voor de subplots
rows = (num_months // 4) + (1 if num_months % 4 != 0 else 0)  # Dynamisch aantal rijen
cols = min(4, num_months)  # Maximaal 6 kolommen

# Maak een figuur met dynamisch aantal subplots
fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 5), sharex=False, sharey=True)
axes = axes.flatten()  # Maak de 2D-array van assen plat voor iteratie

# Itereer over de unieke maanden en maak scatter plots
for i, month in enumerate(unique_months):
    # Filter de data voor de huidige maand
    month_data = df[df['year_month'] == month]
    
    # Bereken de correlatie tussen Price en Load
    correlation = month_data['Price'].corr(month_data['Load'])
    
    # Maak een scatter plot voor de huidige maand
    axes[i].scatter(month_data['Price'], month_data['Load'], s=0.5, alpha=0.7)
    axes[i].set_title(f'{month} (Corr: {correlation:.2f})', fontsize=10)
    axes[i].set_xlabel('Price', fontsize=8)
    axes[i].set_ylabel('Load', fontsize=8)
    
    # Stel dynamische x-aslimieten in
    axes[i].set_xlim(month_data['Price'].min(), month_data['Price'].max())

# Verwijder lege subplots als er minder subplots nodig zijn dan gemaakt
for j in range(num_months, len(axes)):
    fig.delaxes(axes[j])

# Pas de layout aan
fig.suptitle('Price vs Load (Scatter Plots by Month)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

average_correlation = sum(correlations) / len(correlations)
print(f"De gemiddelde correlatie over alle maanden is: {average_correlation:.2f}")

import numpy as np

# Drop rows with missing values in 'Price' or 'Load' columns
df_filtered_pl = df_filtered[['Price', 'Load']].dropna()

# Calculate the correlation coefficient
r = np.corrcoef(df_filtered_pl['Price'], df_filtered_pl['Load'])
print("Correlation coefficient matrix:")
print(r)

# Create a boxplot of the Total_Flow column for the years 2022 and 2023
plt.figure(figsize=(5,3))
plt.boxplot(df_filtered['Total_Flow'].dropna(), vert=False)
plt.xlabel('Total_Flow')
plt.title('Boxplot of Total_Flow')
plt.show()

# Plot the load over the years 2022 and 2023
plt.figure(figsize=(10,5))
plt.plot(df_filtered.index, df_filtered['Total_Flow'], label='Total_Flow')
plt.xlabel('Year')
plt.ylabel('Total_Flow')
plt.title('Total_Flow')
plt.legend()
plt.show()

# Drop rows with missing values in 'Price' column
df_filtered_price = df_filtered[['Price']].dropna()

# Extract the hour and day of the week from the index
df_filtered_price['Hour'] = df_filtered_price.index.hour
df_filtered_price['DayOfWeek'] = df_filtered_price.index.dayofweek

# Define a color map for the days of the week
colors = plt.cm.get_cmap('tab10', 7)  # Use a colormap with 7 distinct colors

# Plot the average price per hour for each day of the week
plt.figure(figsize=(10, 5))
for day in range(7):
    day_data = df_filtered_price[df_filtered_price['DayOfWeek'] == day]
    average_price_per_hour = day_data.groupby('Hour')['Price'].mean()
    plt.plot(average_price_per_hour.index, average_price_per_hour, marker='o', linestyle='-', color=colors(day), label=f'Day {day}')

plt.xlabel('Hour of the Day')
plt.ylabel('Average Price')
plt.title('Average Price per Hour Over a Day')
plt.xticks(range(24))  # Ensure all hours are shown on the x-axis
plt.grid(True)
plt.legend(title='Day of the Week', labels=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.ylim(0, 250)
plt.show()

import matplotlib.pyplot as plt
import pandas as pd

# Drop rows with missing values in 'Load' column
df_filtered_load = df[['Load']].dropna()

# Ensure the index is in datetime format
df_filtered_load.index = pd.to_datetime(df_filtered_load.index)

# Extract the hour and day of the week from the index
df_filtered_load['Hour'] = df_filtered_load.index.hour
df_filtered_load['DayOfWeek'] = df_filtered_load.index.dayofweek

# Define a color map for the days of the week
colors = plt.cm.get_cmap('tab10', 7)  # Use a colormap with 7 distinct colors

# Plot the average load per hour for each day of the week
plt.figure(figsize=(10, 5))
for day in range(7):
    day_data = df_filtered_load[df_filtered_load['DayOfWeek'] == day]
    if not day_data.empty:  # Ensure there is data for the day
        average_load_per_hour = day_data.groupby('Hour')['Load'].mean()
        plt.plot(average_load_per_hour.index, average_load_per_hour, marker='o', linestyle='-', color=colors(day), label=f'Day {day}')

plt.xlabel('Hour of the Day')
plt.ylabel('Average Load')
plt.title('Average Load per Hour Over a Day')
plt.xticks(range(24))  # Ensure all hours are shown on the x-axis
plt.grid(True)
plt.legend(title='Day of the Week', labels=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.ylim(0, 20000)  # Adjust the y-axis limits as needed
plt.show()

# Drop rows with missing values in 'Total_Flow' column
df_filtered_flow = df_filtered[['Total_Flow']].dropna()

# Extract the hour and month from the index
df_filtered_flow['Hour'] = df_filtered_flow.index.hour
df_filtered_flow['Month'] = df_filtered_flow.index.month

# Plot the average Total_Flow per hour for each month
plt.figure(figsize=(15, 20))
for month in range(1, 13):
    plt.subplot(4, 3, month)
    month_data = df_filtered_flow[df_filtered_flow['Month'] == month]
    average_total_flow_per_hour = month_data.groupby('Hour')['Total_Flow'].mean()
    plt.plot(average_total_flow_per_hour.index, average_total_flow_per_hour, marker='o', linestyle='-', label=f'Month {month}')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Average Total_Flow')
    plt.title(f'Average Total_Flow per Hour in Month {month}')
    plt.xticks(range(24))  # Ensure all hours are shown on the x-axis
    plt.grid(True)
    plt.ylim(-3500,3500)
    plt.legend()

plt.tight_layout()
plt.show()

df_filtered['Flow_import'] = df_filtered['Flow_BE_to_NL'] + df_filtered['Flow_DE_to_NL'] + df_filtered['Flow_DK_to_NL'] + df_filtered['Flow_GB_to_NL'] + df_filtered['Flow_NO_to_NL']
df_filtered['Flow_export'] = df_filtered['Flow_NL_to_BE'] + df_filtered['Flow_NL_to_DE'] + df_filtered['Flow_NL_to_DK'] + df_filtered['Flow_NL_to_GB'] + df_filtered['Flow_NL_to_NO']

df_filtered['Hour'] = df_filtered.index.hour
df_filtered['Month'] = df_filtered.index.month

# Plot the average Flow_import and Flow_export per hour for each month
plt.figure(figsize=(15, 20))
for month in range(1, 13):
    plt.subplot(4, 3, month)
    month_data = df_filtered[df_filtered['Month'] == month]
    average_flow_import_per_hour = month_data.groupby('Hour')['Flow_import'].mean()
    average_flow_export_per_hour = month_data.groupby('Hour')['Flow_export'].mean()
    plt.plot(average_flow_import_per_hour.index, average_flow_import_per_hour, marker='o', linestyle='-', label='Flow_import')
    plt.plot(average_flow_export_per_hour.index, average_flow_export_per_hour, marker='o', linestyle='-', label='Flow_export')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Average Flow')
    plt.title(f'Average Flow per Hour in Month {month}')
    plt.xticks(range(24))  # Ensure all hours are shown on the x-axis
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()

import numpy as np

# Drop rows with missing values in 'Price' or 'Load' columns
df_filtered_ws = df_filtered[['Price', 'Total_Flow']].dropna()

# Calculate the correlation coefficient
r = np.corrcoef(df_filtered_ws['Price'], df_filtered_ws['Total_Flow'])
print("Correlation coefficient matrix:")
print(r)

# Extract the month from the index
df_filtered_price['Month'] = df_filtered_price.index.month

# Calculate the average price per month
average_price_per_month = df_filtered_price.groupby('Month')['Price'].mean()

# Plot the average price per month
plt.figure(figsize=(5,3))
plt.plot(average_price_per_month.index, average_price_per_month, marker='o', linestyle='-', label='Average Price')
plt.xlabel('Hour of the Day')
plt.ylabel('Average Price')
plt.title('Average Price per Month')
plt.xticks(range(12))  # Ensure all months are shown on the x-axis
plt.grid(True)
plt.legend()
plt.ylim(0,300)
plt.show()