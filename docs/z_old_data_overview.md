## IBTrACS dataset:
https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C01552 

## Documentation:
https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/doc/IBTrACS_version4r01_Technical_Details.pdf 

## Attributes: 
https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/doc/IBTrACS_v04r01_column_documentation.pdf

## Observations:
- The current project uses the Comma Separated Values (CSV) version of the dataset. 
At first, the subset last3years is used. 

- The dataset contains 174 columns. All columns are currently loaded as object (string) types, including variables that are expected to be numeric or temporal. Explicit type conversion will be required during the preprocessing stage. 

- ISO_TIME is the main temporal column.

- The dataset is mostly sampled at 3-hour intervals per storm, but some irregular gaps exist.
These irregularities will need handling during preprocessing, e.g., interpolation or filtering.

- column data types are "object" mostly