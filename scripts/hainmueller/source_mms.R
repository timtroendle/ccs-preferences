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
    attr_source_purpose = factor(attr_source_purpose),
    attr_industry = factor(attr_industry),
    attr_costs = factor(attr_costs)
  )

# source framing

mm_source <- cj(
  df,
  chosen ~ attr_source_purpose,
  id = ~id,
  estimate = "mm",
  by = ~country + framing
)

plot_source_framing <- ggplot(
  mm_source,
  aes(
    x = estimate,
    y = level,
    colour = country,
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
  facet_wrap(~framing) +
  scale_color_viridis_d(end = 0.8) +
  # xlim(0.2, 0.8) +
  labs(
    y = NULL,
    x = "Marginal means"
  ) +
  theme_classic() +
  theme(
    legend.position = "right",
    text = element_text(size = main_text_size),
    strip.background = element_rect(size = 0),
    strip.text.x = element_text(size = main_text_size, face = "bold"),
    plot.title = element_text(hjust = -0.23, size = main_text_size)
  )

plot_source_framing

# industry and costs

mm_source_industry <- cj(
  df,
  chosen ~ attr_source_purpose,
  id = ~id,
  estimate = "mm",
  by = ~country + framing + attr_industry
)

mm_source_costs <- cj(
  df,
  chosen ~ attr_source_purpose,
  id = ~id,
  estimate = "mm",
  by = ~country + framing + attr_costs
)

plot_source_industry <- ggplot(
  mm_source_industry,
  aes(
    x = estimate,
    y = attr_industry,
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
  facet_wrap(~level) +
  scale_color_viridis_d(end = 0.8) +
  labs(
    y = NULL,
    x = "Marginal means"
  ) +
  theme_classic() +
  theme(
    legend.position = "right",
    text = element_text(size = main_text_size),
    strip.background = element_rect(size = 0),
    strip.text.x = element_text(size = main_text_size, face = "bold"),
    plot.title = element_text(hjust = -0.23, size = main_text_size)
  )

plot_source_costs <- ggplot(
  mm_source_costs,
  aes(
    x = estimate,
    y = attr_costs,
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
  facet_wrap(~level) +
  scale_color_viridis_d(end = 0.8) +
  labs(
    y = NULL,
    x = "Marginal means"
  ) +
  theme_classic() +
  theme(
    legend.position = "right",
    text = element_text(size = main_text_size),
    strip.background = element_rect(size = 0),
    strip.text.x = element_text(size = main_text_size, face = "bold"),
    plot.title = element_text(hjust = -0.23, size = main_text_size)
  )

plot_source_industry
plot_source_costs

ggsave(
  plot = plot_source_framing,
  here("output", "plot_source_framing.png"),
  height = 4, width = 10
)

ggsave(
  plot = plot_source_industry,
  here("output", "plot_source_industry.png"),
  height = 7, width = 10
)

ggsave(
  plot = plot_source_costs,
  here("output", "plot_source_costs.png"),
  height = 6, width = 10
)

