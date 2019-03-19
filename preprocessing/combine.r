# set work directory
# setwd("...")
getwd()

library(data.table)
library(plyr)

combine.csv.files <- function()
{
  input.files = c(
    "./data_preprocessed/061_HEK293T_class1.csv",
    "./data_preprocessed/065_HEK293T_class2.csv",
    "./data_preprocessed/066_HEK293T_class3.csv",
    "./data_preprocessed/067_HEK293T_class4.csv",
    "./data_preprocessed/068_HEK293T_class5.csv",
    "./data_preprocessed/073_HEK293T_class6.csv",
    "./data_preprocessed/074_HEK293T_class7.csv"
  )
  
  # read and add class column for every file
  data061 = fread(input.files[1], data.table=getOption("datatable.fread.datatable", FALSE))
  data061$class = "class1"
  
  data065 = fread(input.files[2], data.table=getOption("datatable.fread.datatable", FALSE))
  data065$class = "class2"
  
  data066 = fread(input.files[3], data.table=getOption("datatable.fread.datatable", FALSE))
  data066$class = "class3"
  
  data067 = fread(input.files[4], data.table=getOption("datatable.fread.datatable", FALSE))
  data067$class = "class4"
  
  data068 = fread(input.files[5], data.table=getOption("datatable.fread.datatable", FALSE))
  data068$class = "class5"
  
  data073 = fread(input.files[6], data.table=getOption("datatable.fread.datatable", FALSE))
  data073$class = "class6"
  
  data074 = fread(input.files[7], data.table=getOption("datatable.fread.datatable", FALSE))
  data074$class = "class7"
  
  # combine all of the dataframes into one
  combined = rbind.fill(data061, data065, data066, data067, data068, data073, data074)
  dim(combined)

  # eliminate columns with NA values
  combined = combined[, colSums(is.na(combined)) == 0]
  dim(combined)
  attributes(combined)

  gc()
  
  # save data frame into csv
  fwrite(combined, "./data_preprocessed/combined_data.csv")
}

combine.csv.files()
