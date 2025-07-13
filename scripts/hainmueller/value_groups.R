library(tidyverse)
library(dplyr)
library(here)
library(cregg)
library(ggplot2)
library(viridis)

df_values <- read_csv(
  here("data", "data_values_ch_cn.csv"),
  show_col_types = FALSE
)

hist(df_values$lreco)
hist(df_values$galtan)
hist(df_values$socio_ecol)

df_values$lreco <- scale(df_values$lreco)
df_values$galtan <- scale(df_values$galtan)
df_values$socio_ecol <- scale(df_values$socio_ecol)
