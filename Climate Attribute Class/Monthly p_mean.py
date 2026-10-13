## import libraries
import pandas as pd
import platform
 
## Define your file paths
csv_path = ".../CAMELS_Afr_Climate_Daily_L1_p_mean_mswep.csv"
 
 
## Load the data
df = pd.read_csv(csv_path)
 
#Read the first 5 rows
df.head(5)
 
## Convert 'Date' column to datetime (Handling the MM/DD/YYYY format)
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].dt.strftime('%#m/%#d/%Y')
 
#again read the data after changing the date format
df.head(5)
 
##Resample to Month Start (summing the daily values)
 
#Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])
 
# Set Date as index
df.set_index('Date', inplace=True)
 
# Sum daily values for each month
monthly_data = df.resample('MS').sum()
 
# Display the result
print(monthly_data.head())
 
## Format date as M/YYYY
if platform.system() == 'Windows':
    monthly_data.index = monthly_data.index.strftime('%#m/%#d/%Y')
else:
    monthly_data.index = monthly_data.index.strftime('%-m/%-d/%Y')
 
output_path= ".../CAMELS_Afr_Climate_Monthly_L1_p_mean_mswep.csv"
# Save to CSV
monthly_data.to_csv(output_path)
 
print(f"Monthly file saved to: {output_path}")
