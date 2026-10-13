## import libraries
import pandas as pd
import platform
 
## Define your file paths
CSV_path = ".../CAMELS_Afr_Climate_Daily_L1_p_mean_mswep.csv"
 
## Load the data
df = pd.read_csv(CSV_path)
 
#Read the first 5 rows
df.head(5)
 
## Convert 'Date' column to datetime (Handling the MM/DD/YYYY format)
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].dt.strftime('%#m/%#d/%Y')
 
#again read the data after changing the date format
df.head(5)
 
##Resample to Annual Start (summing the daily values)
 
#Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])
 
# Set Date as index
df.set_index('Date', inplace=True)
 
# Sum daily values for each month
annual_percip = df.resample('YS').sum()
 
# Display the result
print(annual_percip.head())
 
## Transform the index to show ONLY the year (integer)
annual_percip.index = annual_percip.index.year
 
##Rename the index to "Year"
annual_percip.index.name = 'Year'
 
##Define your output path
output_path = ".../CAMELS_Afr_Climate_Annual_L1_p_mswep.csv"
 
##Save to CSV
annual_percip.to_csv(output_path)
print(f"Annual file saved to: {output_path}")