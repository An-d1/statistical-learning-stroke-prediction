library(dplyr)
library(here)
library(readr)

#using 'here' to improve code reproducibility
file_path <- here("data", "raw", "healthcare-dataset-stroke-data.csv")

df <- read.csv(file_path)

# - Data Cleaning Pipeline
df_clean <- df %>%
  # I drop the 'id' column as it has no predictive value
  select(-id) %>%
  
  # Handle 'bmi' missing values ("N/A")
  # First, I convert to numeric (this forces "N/A" strings to become actual NA values)
  # Then, replace those NAs with the median of the column
  mutate(
    bmi = as.numeric(na_if(bmi, "N/A")),
    bmi = if_else(is.na(bmi), median(bmi, na.rm = TRUE), bmi)
  ) %>%
  
  # There was only 1 gneder listed as other, so i removed the row to not break the scripts later
  filter(gender != "Other") %>%
  
  # Here I convert categorical character columns into Factors
  mutate(
    across(c(gender, ever_married, work_type, Residence_type, smoking_status), as.factor),
    
    # Also I convert binary integers to factors 
    # so algorithms treat them as categories, not continuous numbers
    across(c(hypertension, heart_disease, stroke), as.factor)
  )

#only for shoring, for working will be used the .rds dataset
write_csv(df_clean, here("data", "processed", "cleaned_stroke_dataset.csv"))

saveRDS(df_clean, here("data", "processed", "cleaned_stroke_dataset.rds"))
print("Data cleaning complete. Cleaned dataset saved to data/processed/")
