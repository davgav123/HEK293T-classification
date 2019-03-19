# set work directory
# setwd("...")
getwd()

# we can also put everything inside a function...
library(data.table)
transpose.and.filter <- function(inPath, outPath)
{
  df = fread(inPath, header = F, data.table=getOption("datatable.fread.datatable", FALSE))
  df = transpose(df)
  
  # attributes(df)
  
  # remove index column
  df = df[, -1]
  
  # take attributes, hg names
  attr = df[1, ]
  attr = as.character(attr)
  
  # drop hg names from the table
  df = df[-1, ]
  
  # set new attribute names
  colnames(df) = attr
  # attributes(df)
  
  # delete columns with all rows
  df.filtered = df[, colSums(df != 0) != 0]
  print(dim(df.filtered))
  
  # save file
  fwrite(df.filtered, outPath)
  
  # call the garbage collector, just in case
  gc()
}

transpose.and.filter("./data/061_HEK293T_human_embryonic_kidney_csv.csv", 
                     "./data_preprocessed/061_HEK293T_human_embryonic_kidney_transposed_filtered.csv")

transpose.and.filter("./data/065_HEK293T_human_embryonic_kidney_csv.csv", 
                     "./data_preprocessed/065_HEK293T_human_embryonic_kidney_transposed_filtered.csv")

transpose.and.filter("./data/066_HEK293T_human_embryonic_kidney_csv.csv", 
                     "./data_preprocessed/066_HEK293T_human_embryonic_kidney_transposed_filtered.csv")

transpose.and.filter("./data/067_HEK293T_human_embryonic_kidney_csv.csv", 
                     "./data_preprocessed/067_HEK293T_human_embryonic_kidney_transposed_filtered.csv")

transpose.and.filter("./data/068_HEK293T_human_embryonic_kidney_csv.csv", 
                     "./data_preprocessed/068_HEK293T_human_embryonic_kidney_transposed_filtered.csv")

transpose.and.filter("./data/073_HEK293T-human_embryonic_kidney_matcsv.csv", 
                     "./data_preprocessed/073_HEK293T_human_embryonic_kidney_transposed_filtered.csv")

transpose.and.filter("./data/074_HEK293T-human_embryonic_kidney_csv.csv", 
                     "./data_preprocessed/074_HEK293T_human_embryonic_kidney_transposed_filtered.csv")
