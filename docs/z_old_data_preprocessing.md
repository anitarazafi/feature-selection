## Defining missing values for each column according to the dataset documentation
- define a generic rule for general string columns for what are considered as missing values 
- replace the missing values with NaN
- overview the missing values, no critical column values were missing

## Handle missing values:
- Keep missing values as NaN since the critical columns already pretty complete,
- We can conclude that the data is already cleaned up
- Save the data frame with standardized missing values to parquet file under data/processed folder

## Fix data types:
- convert ISO_TIME column data type to datetime, make sure all entries have valid timestamp
- convert numerical values to numeric data type
- standardize categorical values and quick check how many unique values are present for each column

## Handle duplicates

## Sort by time and identifiers

## Create derived features 

## Encode categorical variables
- We used one-hot encoding to transform the categorical variables

## Save cleaned dataset for modeling