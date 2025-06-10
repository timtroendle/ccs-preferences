library(dplyr)
library(tidyverse)
library(readr)
library(here)
library(cregg)
library(ggplot2)
library(viridis)

main_text_size <- 14

df_ch <- read_csv(
  here("data", "data_translated_ch.csv"),
  show_col_types = FALSE
) |>
  mutate(
    country = "switzerland"
  )

df_cn <- read_csv(
  here("data", "data_translated_cn.csv"),
  show_col_types = FALSE
) |>
  mutate(
    country = "china"
  )

df <- bind_rows(df_ch, df_cn) |>
  mutate(
    country = factor(country),
    framing = factor(framing),
    attr_vicinity = factor(attr_vicinity),
    attr_reason = factor(attr_reason),
    attr_source_purpose = factor(attr_source_purpose)
  )

# vicinity and reason

mm_location <- cj(
  df,
  supported ~ attr_reason + attr_vicinity,
  id = ~ id,
  estimate = "mm",
  by = ~ country + attr_source_purpose + framing
)

plot_location <- ggplot(
  mm_location,
  aes(
    x = estimate,
    y = level,
    colour = country,
    shape = framing
  )
) +
  geom_vline(
    xintercept = 0.5,
    linetype = 2,
    colour = "gray40",
    linewidth = .3
  ) +
  geom_point(
    size = 2.8,
    position = position_dodge(width = 0.5)
  ) +
  geom_errorbarh(
    aes(
      xmax = upper,
      xmin = lower
    ),
    height = .1,
    position = position_dodge(width = 0.5)
  ) +
  scale_color_viridis_d(end = 0.8) +
  xlim(0.2, 0.8) +
  labs(
    y = NULL,
    x = "Marginal means"
  ) +
  facet_wrap(~attr_source_purpose) +
  theme_classic() +
  theme(
    legend.position = "right",
    text = element_text(size = main_text_size),
    strip.background = element_rect(size = 0),
    strip.text.x = element_text(size = main_text_size, face = "bold"),
    plot.title = element_text(hjust = -0.23, size = main_text_size)
  )

plot_location

ggsave(
  plot = plot_location,
  here("output", "plot_source_location.png"),
  height = 7, width = 10
)
