
library(tidyverse)
library(destiny)
library(readxl)
library(Biobase)
library(rgl)


#########################################
# Import

path <- choose.dir()
inputpath <- paste0(path, '/per_sample_data/')
outpath <- paste0(path, '/stage2_data/')
data_list <- list.files(inputpath)

for (i in data_list) {
  raw_ct <- read.csv(paste0(inputpath, i), 
                     stringsAsFactors = FALSE)
  as_tibble(raw_ct)
  
  # Diffusion Map
  dm_data <- raw_ct[, 1:18]
  dm <- DiffusionMap(dm_data)
  dpt <- DPT(dm)
  as_tibble(as.data.frame(dpt))
  
  # A Data Frame of Pseudo Time and Age
  df <- as.data.frame(dpt)
  new_df <- as.data.frame(df$DPT1) 
  new_df$Age <- raw_ct$age
  head(new_df)
  
  dc1 <- dm$DC1
  dc2 <- dm$DC2
  
  dcs <- as.data.frame(dc1)
  dcs$dc2 <- dc2
  dcs$age <- raw_ct$age
  dcs$id <- raw_ct$subject.id
  dcs$dpt <- df$DPT187
  
  write.csv(dcs, paste0(outpath, i))
}