library(tidyverse)
library(ggtext)
library(ggstance)
library(kableExtra)
library(ggh4x)

asllrp <- read_csv("data/asllrp-collective-nmm-annotated.csv") %>%
  mutate(
    only_nmm = case_when(
      qm == 0 ~ 1,
      TRUE ~ only_nmm
    )
  ) %>% 
  select(utterance_id, phenomenon, good_sentence, bad_sentence, nll_diff, training_condition, inference_condition, phenomenon_family, only_nmm) %>%
  mutate(
    source = "ASLLRP"
  )

exp3 <- asllrp %>%
  # left_join(og_bleurt %>% select(utterance_id, training_condition, inference_condition)) %>%
  mutate(
    phenomenon = str_to_title(phenomenon) %>%
      str_replace("_", "-") %>%
      str_replace("Yes/No", "Y/N-Questions") %>%
      str_replace("-r", "-R"),
    phenomenon = case_when(
      phenomenon == "Classifier" ~ "Classifiers",
      phenomenon == "Number" ~ "Numbers",
      phenomenon == "Conditional" ~ "Conditionals",
      phenomenon == "Wh-Question" ~ "Wh-Questions",
      TRUE ~ phenomenon
    ),
    only_nmm = case_when(
      is.na(only_nmm) ~ 0,
      TRUE ~ only_nmm
    ),
    inference_condition = factor(inference_condition, levels = c("AC", "NE", "NM", "NF", "NFB", "NH", "NHM", "NHF", "NHB"))
  ) %>%
  mutate(
    # nll_diff_scaled = scale(nll_diff)
    only_nmm = case_when(
      phenomenon %in% c("Negation", "Wh-Questions") ~ 0,
      TRUE ~ only_nmm
    ),
    nll_diff_scaled = nll_diff
  ) %>%
  mutate(
    only_nmm = as.logical(only_nmm),
    phenomenon = case_when(
      only_nmm == TRUE ~ glue::glue("{phenomenon}<br><span style='font-size:12pt'>(<i>only non-manual cue usage</i>)</span>"),
      TRUE ~ phenomenon
    ),
    phenomenon = str_replace(phenomenon, "^Y/N-Questions$", "Polar Questions vs. Declaratives"),
    phenomenon = str_replace(phenomenon, "^Y/N-Questions\\<br\\>", "Polar Questions vs. Declaratives<br>"),
    phenomenon = str_replace(phenomenon, "Y/N-Questions-Reversed", "Declaratives vs. Polar Questions"),
    phenomenon = str_replace(phenomenon, "^Negation$", "Negation vs. Positive"),
    phenomenon = str_replace(phenomenon, "Negation-Reversed", "Positive vs. Negation"),
    phenomenon = factor(
      phenomenon,
      levels = c(
        "Numbers",
        "Fingerspelling",
        "Classifiers",
        "Wh-Questions",
        "Negation vs. Positive",
        "Positive vs. Negation",
        "Conditionals",
        "Conditionals<br><span style='font-size:12pt'>(<i>only non-manual cue usage</i>)</span>",
        "Declaratives vs. Polar Questions",
        "Polar Questions vs. Declaratives",
        "Polar Questions vs. Declaratives<br><span style='font-size:12pt'>(<i>only non-manual cue usage</i>)</span>"
      )
    )
  ) %>%
  filter(training_condition==inference_condition)

metrics_training <- exp3 %>%
  group_by(phenomenon, inference_condition, only_nmm) %>%
  summarize(
    n = n(),
    std = sd(nll_diff_scaled),
    cb = qt(0.05/2, n-1, lower.tail = FALSE) * std/sqrt(n),
    mean = mean(nll_diff_scaled),
    accuracy = mean(nll_diff > 0)
  ) %>%
  ungroup()

range <- seq(0.02, 1, length.out=8)
# range <- c(0.05, 0.20, 0.30, 0.40,0.50, 0.6, 0.7, 0.8, 0.9, 1.01)
# pal <- scales::seq_gradient_pal("#f7fcb9", "#41ab5d")(range)
# pal <- scales::seq_gradient_pal("#e9f7cb", "#1b7378")(range)
# pal <- c('#e9f7cb', '#d4e8c2', '#bfd8b8', '#aac9af', '#95baa5', '#80ac9c', '#6a9d93', '#548f8a', '#3b8181', '#1b7378')
pal <- c('#e9f7cb','#bfd8b8', '#aac9af', '#95baa5', '#80ac9c', '#548f8a', '#1b7378')
# scales::seq_gradient_pal("white", "black")(range)
setSpec <- function(y){
  kableExtra::cell_spec(y, "latex", background = pal[cut(y, breaks=range, include.lowest = TRUE)])
}

metrics_training %>%
  # unite("phenomenon", phenomenon, only_nmm, sep = "_") %>%
  select(phenomenon, only_nmm, inference_condition, n, accuracy) %>% 
  mutate(
    accuracy = round(accuracy,2)
    # glue::glue("{format(round(accuracy*100,1), nsmall=1)}%")
  ) %>%
  rename(NonManual = only_nmm) %>%
  pivot_wider(names_from = inference_condition, values_from = accuracy) %>%
  mutate_at(c("AC", "NF", "NFB"), .funs = setSpec) %>%
  mutate(
    `Primary Cue` = case_when(
      phenomenon %in% c("Numbers", "Fingerspelling", "Classifiers", "Negation-Reversed", "Y/N-Questions-Reversed") ~ "Hands",
      phenomenon == "Conditionals" & NonManual == 1 ~ "Brow raise",
      phenomenon == "Y/N-Questions" & NonManual == 1 ~ "Brow raise",
      phenomenon == "Y/N-Questions" & NonManual == 0 ~ "Hands + Brow raise",
      phenomenon == "Wh-Questions" ~ "Hands + Brow lowered",
      phenomenon == "Negation" ~ "Hands + Head shake",
      phenomenon == "Conditionals" & NonManual == 0 ~ "Hands + Brow raise"
    ),
    `Secondary Cue` = case_when(
      phenomenon %in% c("Conditionals") ~ "Head thrust",
      phenomenon == "Y/N-Questions" & NonManual == 1 ~ "Head forward",
      phenomenon == "Wh-Questions" ~ "Head shake",
      TRUE ~ ""
    )
  ) %>%
  select(Phenomenon = phenomenon, `Primary Cue`, `Secondary Cue`, N = n, AC:NFB) %>%
  kable("latex", escape = FALSE, booktabs=TRUE)

correctness_training <- exp3 %>%
  mutate(
    # nll_diff_scaled = scale(nll_diff)
    only_nmm = case_when(
      phenomenon %in% c("Negation", "Wh-Questions") ~ 0,
      TRUE ~ only_nmm
    ),
    nll_diff_scaled = nll_diff,
    correct = nll_diff > 0
  )

accuracies_training <- correctness_training %>%
  group_by(phenomenon, only_nmm, inference_condition) %>%
  summarize(acc = mean(correct))

correctness_training %>% 
  inner_join(ac_correctness %>% select(-training_condition)) %>%
  filter(inference_condition == training_condition) %>%
  group_by(phenomenon, only_nmm, inference_condition) %>%
  # summarize(n = n())
  nest() %>%
  mutate(
    test_results = map(data, function(x) {
      contingency_table <- table(Condition_A = x$correct, Condition_B = x$ac_correct)
      # mcnemar.test(contingency_table)
      
      print(contingency_table)
      
      
      A_wrong_B_right <- contingency_table[1, 2] # Row 0, Column 1
      A_right_B_wrong <- contingency_table[2, 1] # Row 1, Column 0
      
      if (A_right_B_wrong + A_wrong_B_right == 0) {
        return(tibble::tibble(
          estimate = 0.5,
          statistic = 0,
          p.value = 1,          # No difference means p-value of 1
          parameter = 0,
          conf.low = NA_real_,
          conf.high = NA_real_,
          method = "Exact binomial test (Skipped: 0 discordant pairs)",
          alternative = "two.sided"
        ))
      }
      
      else {
        binom.test(x = A_wrong_B_right, 
                   n = A_wrong_B_right + A_right_B_wrong, 
                   p = 0.5, 
                   alternative = "two.sided") %>%
          broom::tidy()
      }
      
      # Test if B being right is significantly greater than A being right
    })
  ) %>%
  select(-data) %>%
  unnest(test_results) %>% 
  group_by(phenomenon) %>%
  mutate(
    padj_bh = p.adjust(p.value, method = "BH"),
    padj_bonf = p.adjust(p.value, method = "bonferroni")
  ) %>%
  arrange(phenomenon, inference_condition) %>% 
  mutate(
    significant = p.value < .05,
    sign_bh = padj_bh < .05,
    sign_bonf = padj_bonf < .05
  ) %>%
  inner_join(accuracies_training) %>%
  select(phenomenon, only_nmm, inference_condition, acc, significant, sign_bh, sign_bonf)

## NLL DIFF

nll_diffs <- exp3 %>%
  filter(training_condition=="AC") %>%
  mutate(
    only_nmm = case_when(
      phenomenon %in% c("Negation", "Wh-Questions") ~ 0,
      TRUE ~ only_nmm
    ),
    nll_diff_scaled = nll_diff
  ) 

ac_diffs <- nll_diffs %>%
  filter(inference_condition == "AC") %>%
  select(-inference_condition, -training_condition, -nll_diff_scaled) %>%
  rename(ac_diff = nll_diff)

nll_diffs_training <- exp3 %>%
  mutate(
    # nll_diff_scaled = scale(nll_diff)
    only_nmm = case_when(
      phenomenon %in% c("Negation", "Wh-Questions") ~ 0,
      TRUE ~ only_nmm
    ),
    nll_diff_scaled = nll_diff
  ) 

sigs_training <- nll_diffs_training %>%
  filter(inference_condition!="AC") %>%
  inner_join(ac_diffs) %>%
  group_by(phenomenon, only_nmm, inference_condition) %>%
  nest() %>%
  mutate(
    t_test = map(data, function(x) {
      t.test(x$ac_diff, x$nll_diff) %>%
        broom::tidy()
    })
  ) %>%
  ungroup() %>%
  select(-data) %>%
  unnest(t_test) %>%
  group_by(phenomenon) %>%
  mutate(
    padj_bh = p.adjust(p.value, method = "BH"),
    padj_bonf = p.adjust(p.value, method = "bonferroni")
  ) %>%
  arrange(phenomenon, inference_condition) %>% 
  mutate(
    significant = p.value < .05,
    sign_bh = padj_bh < .05,
    sign_bonf = padj_bonf < .05,
    sig = case_when(
      padj_bonf >= 0.05 ~ "",
      padj_bonf >= 0.01 & padj_bonf < 0.05 ~ "*",
      padj_bonf >= 0.001 & padj_bonf < 0.01 ~ "**",
      padj_bonf < 0.001 ~ "***"
    ) 
  ) %>% filter(sign_bonf == TRUE) %>%
  ungroup()

sigs_x_training <- bind_cols(
  sigs_training %>%
    distinct(phenomenon),
  tibble(
    x = c(0.09, 0.14, 0.18, 0.098, 0.38, -0.1)
  )
)


metrics_training %>%
  mutate(
    discard = case_when(
      phenomenon %in% c("Negation", "Wh-Questions") & only_nmm == 1 ~ TRUE,
      TRUE ~ FALSE
    )
  ) %>%
  filter(discard == FALSE) %>%
  mutate(
    inference_condition = fct_rev(inference_condition),
    color = case_when(
      inference_condition == "AC" ~ "#045a8d",
      inference_condition %in% c("NE", "NM", "NF", "NFB") ~ "#2b8cbe",
      inference_condition %in% c("NH", "NHM", "NHF", "NHB") ~ "#a6bddb"
    )
  ) %>%
  ggplot(aes(mean, inference_condition, color = color)) +
  geom_point(size = 2) +
  geom_linerangeh(aes(xmin = mean-cb, xmax = mean+cb)) +
  geom_vline(xintercept = 0.0, linetype = "dashed") +
  geom_text(data = sigs_training %>% inner_join(sigs_x_training), aes(x = x, y = inference_condition, label = sig), 
            color = "red", family = "Helvetica", size = 5, fontface="bold") +
  facet_wrap(~phenomenon, scales = "free_x", nrow=2) +
  facetted_pos_scales(
    x = list(
      phenomenon == "Classifiers" ~ scale_x_continuous(limits = c(0,0.1), breaks=c(0,0.025, 0.05, 0.075, 0.1), labels = c("0", "0.025", "0.05", "0.075", "0.1")),
      phenomenon == "Conditionals" ~ scale_x_continuous(limits = c(0, 0.1), labels = c("0", "0.025", "0.05", "0.075", "0.1")),
      phenomenon == "Fingerspelling" ~ scale_x_continuous(limits = c(0, 0.15), labels = c("0", "0.05", "0.10", "0.15")),
      str_detect(phenomenon, "Conditionals<br>") ~ scale_x_continuous(limits = c(-0.03, 0.09), breaks = c(-0.03, 0, 0.03, 0.06, 0.09)),
      phenomenon == "Numbers" ~ scale_x_continuous(limits = c(0, 0.1), breaks=c(0,0.025, 0.05, 0.075, 0.1), labels = c("0", "0.025", "0.05", "0.075", "0.1")),
      phenomenon == "Negation vs. Positive" ~ scale_x_continuous(limits = c(0, 0.2), labels = c("0", "0.05", "0.10", "0.15", "0.2")),
      phenomenon == "Wh-Questions" ~ scale_x_continuous(limits = c(0, 0.2), labels = c("0", "0.05", "0.10", "0.15", "0.2"))
    )
  ) +
  # ggh4x::facet_grid2(~phenomenon, scales = "free", independent = "all") +
  scale_color_identity() +
  theme_bw(base_size = 16, base_family = "Times") +
  theme(
    legend.position = "top",
    strip.background = element_rect(color=NA, fill = "transparent"),
    strip.text = element_markdown(),
    axis.title.y = element_blank(),
    axis.text = element_text(color="black"),
    axis.text.x = element_text(size = 12),
    panel.grid = element_blank(),
    panel.background = element_rect(fill = "transparent", color = NA),
    plot.background = element_rect(fill = "transparent", color=NA)
  ) +
  labs(
    x = "Mean Surprisal Difference (95% CI)"
  )

# 4.33 x 15.28
ggsave("images/asllrp-surprisal-difference-inference-training.pdf", height = 5, width = 15.28, dpi=300, device=cairo_pdf)
ggsave("images/asllrp-surprisal-difference-inference-training-cue.pdf", height = 5, width = 15.28, dpi=300, device=cairo_pdf)
ggsave("images/asllrp-surprisal-difference-inference-training.svg", height = 5, width = 15.28, dpi=300)

