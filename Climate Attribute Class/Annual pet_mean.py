## import libraries
import pandas as pd
import platform
 
## Define your file paths
input_path = ".../CAMELS_Afr_Climate_Daily_L4_pet_mean.csv"
 
## Load the data
df = pd.read_csv(input_path)
 
#Read the first 5 rows
df.head(5)
 
## Convert 'Date' column to datetime (Handling the MM/DD/YYYY format)
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].dt.strftime('%#m/%#d/%Y')
 
#again read the data after changing the date format
df.head(5)
 
##Resample to annual frequency and sum the values
 
#Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])
 
# Set Date as index
df.set_index('Date', inplace=True)
 
# Sum daily values for each year
print("Calculating annual sums...")
annual_pet = df.resample('YE').sum() #'YE' stands for Year End (groups by calendar year)
 
# Display the result
print(annual_pet.head())
 
## Transform the index to show ONLY the year (integer)
annual_pet.index = annual_pet.index.year
 
##Rename the index to "Year"
annual_pet.index.name = 'Year'
 
##Define your output path
output_path = ".../CAMELS_Afr_Climate_Annual_L4_pet_mean.csv"
 
##Save to CSV
annual_pet.to_csv(output_path)
print(f"Success! Annual PET sum saved to: {output_path}")
print(f"Years processed: {annual_pet.index.min()} to {annual_pet.index.max()}")