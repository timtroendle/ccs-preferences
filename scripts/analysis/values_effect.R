library(tidyverse)
library(dplyr)
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

df_conjoint <- bind_rows(df_ch, df_cn) |>
  mutate(
    country = factor(country),
    framing = factor(framing),
    attr_vicinity = factor(attr_vicinity),
    attr_reason = factor(attr_reason),
    attr_source_purpose = factor(attr_source_purpose),
    attr_industry = factor(attr_industry),
    attr_costs = factor(attr_costs)
  )

df_values <- read_csv(
  here("data", "data_values_ch_cn.csv"),
  show_col_types = FALSE
) |>
  mutate(
    country = factor(country)
  )

df_values$lreco_bin <- cut(
  df_values$lreco,
  breaks = quantile(
    df_values$lreco,
    probs = c(0, 0.33, 0.67, 1),
    na.rm = TRUE
  ),
  labels = c("Left", "Mid", "Right"),
  include.lowest = TRUE
)

df_values$galtan_bin <- cut(
  df_values$galtan,
  breaks = quantile(
    df_values$galtan,
    probs = c(0, 0.5, 1),
    na.rm = TRUE
  ),
  labels = c("Alternative", "Conservative"),
  include.lowest = TRUE
)

df_values$ecol_bin <- cut(
  df_values$socio_ecol,
  breaks = quantile(
    df_values$socio_ecol,
    probs = c(0, 0.33, 0.67, 1),
    na.rm = TRUE
  ),
  labels = c("High concern", "Mid concern", "Low concern"),
  include.lowest = TRUE
)

df <- left_join(df_conjoint, df_values, by = c("id", "country")) |>
  mutate(
    lreco_bin = factor(lreco_bin),
    galtan_bin = factor(galtan_bin),
    ecol_bin = factor(ecol_bin)
  )

df <- df[!is.na(df$lreco_bin), ]
df <- df[!is.na(df$galtan_bin), ]
df <- df[!is.na(df$ecol_bin), ]

mm_lreco <- cj(
  df,
  supported ~ attr_source_purpose,
  id = ~ id,
  estimate = "mm",
  by = ~ lreco_bin + country + framing
)

mm_galtan <- cj(
  df,
  supported ~ attr_source_purpose,
  id = ~ id,
  estimate = "mm",
  by = ~ galtan_bin + country + framing
)

mm_ecol <- cj(
  df,
  supported ~ attr_source_purpose,
  id = ~ id,
  estimate = "mm",
  by = ~ ecol_bin + country + framing
)

# plot_values <- function(data, value_column, main_text_size = 14) {
#   ggplot(
#     data,
#     aes(
#       x = estimate,
#       y = level,
#       colour = country,
#       alpha = value_column
#     )
#   ) +
#     geom_vline(
#       xintercept = 0.5,
#       linetype = 2,
#       colour = "gray40",
#       linewidth = .3
#     ) +
#     geom_point(
#       size = 2.8,
#       position = position_dodge(width = 0.5)
#     ) +
#     geom_errorbarh(
#       aes(xmax = upper, xmin = lower),
#       height = .1,
#       position = position_dodge(width = 0.5)
#     ) +
#     scale_color_viridis_d(end = 0.8) +
#     xlim(0.15, 0.85) +
#     labs(
#       y = NULL,
#       x = "Marginal means"
#     ) +
#     facet_wrap(~framing) +
#     theme_classic() +
#     theme(
#       legend.position = "right",
#       text = element_text(size = main_text_size),
#       strip.background = element_rect(size = 0),
#       strip.text.x = element_text(size = main_text_size, face = "bold"),
#       plot.title = element_text(hjust = -0.23, size = main_text_size)
#     )
# }

plot_lreco <- ggplot(
  mm_lreco,
  aes(
    x = estimate,
    y = level,
    colour = country,
    alpha = lreco_bin
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
  scale_alpha_discrete(range = c(0.4, 1)) +
  xlim(0.15, 0.85) +
  labs(
    y = NULL,
    x = "Marginal means"
  ) +
  facet_wrap(~framing) +
  theme_classic() +
  theme(
    legend.position = "right",
    text = element_text(size = main_text_size),
    strip.background = element_rect(size = 0),
    strip.text.x = element_text(size = main_text_size, face = "bold"),
    plot.title = element_text(hjust = -0.23, size = main_text_size)
  )

plot_galtan <- ggplot(
  mm_galtan,
  aes(
    x = estimate,
    y = level,
    colour = country,
    alpha = galtan_bin
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
  scale_alpha_discrete(range = c(0.4, 1)) +
  xlim(0.15, 0.85) +
  labs(
    y = NULL,
    x = "Marginal means"
  ) +
  facet_wrap(~framing) +
  theme_classic() +
  theme(
    legend.position = "right",
    text = element_text(size = main_text_size),
    strip.background = element_rect(size = 0),
    strip.text.x = element_text(size = main_text_size, face = "bold"),
    plot.title = element_text(hjust = -0.23, size = main_text_size)
  )

plot_ecol <- ggplot(
  mm_ecol,
  aes(
    x = estimate,
    y = level,
    colour = country,
    alpha = ecol_bin
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
  scale_alpha_discrete(range = c(0.4, 1)) +
  xlim(0.15, 0.85) +
  labs(
    y = NULL,
    x = "Marginal means"
  ) +
  facet_wrap(~framing) +
  theme_classic() +
  theme(
    legend.position = "right",
    text = element_text(size = main_text_size),
    strip.background = element_rect(size = 0),
    strip.text.x = element_text(size = main_text_size, face = "bold"),
    plot.title = element_text(hjust = -0.23, size = main_text_size)
  )

plot_lreco
plot_galtan
plot_ecol

ggsave(
  plot = plot_lreco,
  here("output", "plot_lreco.png"),
  height = 6, width = 10
)

ggsave(
  plot = plot_galtan,
  here("output", "plot_galtan.png"),
  height = 5, width = 10
)

ggsave(
  plot = plot_ecol,
  here("output", "plot_ecol.png"),
  height = 6, width = 10
)
