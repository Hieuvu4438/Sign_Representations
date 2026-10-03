library(tidyverse)
library(ggtext)
library(ggstance)
library(kableExtra)
library(ggh4x)

og_bleurt <- read_csv("data/asllrp-collective.csv") %>%
  mutate(
    source = "ASLLRP"
  )

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

combined <- asllrp %>%
  left_join(og_bleurt %>% select(utterance_id, training_condition, inference_condition, bleurt, source)) %>%
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
    discard = str_detect(source, "Legacy") & phenomenon %in% c("Conditionals", "Wh-Questions")
  ) %>%
  filter(discard != TRUE) %>%
  # group_by(phenomenon) %>%
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
  )

agg <- combined %>%
  rename(NonManual = only_nmm) %>%
  filter(training_condition=="AC") %>%
  filter(source == "ASLLRP") %>%
  mutate(bleurt = bleurt/100) %>%
  group_by(phenomenon, inference_condition, NonManual, source) %>%
  summarize(
    n = n(),
    diff = mean(nll_diff),
    acc = mean(nll_diff > 0),
    bleurt = mean(bleurt)
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

agg %>%
  select(-diff, -acc, -source) %>%
  mutate(
    bleurt = round(bleurt, 2)
  ) %>%
  pivot_wider(names_from = inference_condition, values_from = bleurt) %>%
  mutate(
    `Primary Cue` = case_when(
      phenomenon %in% c("Numbers", "Fingerspelling", "Classifiers", "Positive vs. Negation", "Declaratives vs. Polar Questions") ~ "Hands",
      phenomenon == "Conditionals<br><span style='font-size:12pt'>(<i>only non-manual cue usage</i>)</span>" & NonManual == 1 ~ "Brow raise",
      phenomenon == "Polar Questions vs. Declaratives<br><span style='font-size:12pt'>(<i>only non-manual cue usage</i>)</span>" ~ "Brow raise",
      phenomenon == "Polar Questions vs. Declaratives" ~ "Hands + Brow raise",
      phenomenon == "Wh-Questions" ~ "Hands + Brow lowered",
      phenomenon == "Negation vs. Positive" ~ "Hands + Head shake",
      phenomenon == "Conditionals" & NonManual == FALSE ~ "Hands + Brow raise"
    ),
    `Secondary Cue` = case_when(
      # phenomenon %in% c("Conditionals") ~ "Head thrust",
      str_detect(phenomenon, "Conditionals") ~ "Head thrust",
      str_detect(phenomenon, "Polar Questions vs. Declaratives") ~ "Head forward",
      phenomenon == "Wh-Questions" ~ "Head shake",
      TRUE ~ ""
    )
  ) %>%
  mutate_at(c("AC", "NE", "NF", "NH", "NHM", "NM", "NHF", "NHB", "NFB"), .funs = setSpec) %>%
  select(Phenomenon = phenomenon, `Primary Cue`, `Secondary Cue`, N = n, AC:NHB) %>%
  kable("latex", escape = FALSE, booktabs=TRUE)

ac_bleurt <- combined %>%
  filter(training_condition=="AC", inference_condition == "AC") %>%
  filter(source == "ASLLRP") %>%
  rename(ac_bleurt = bleurt)

ac_rest <- combined %>%
  filter(training_condition=="AC", inference_condition != "AC") %>%
  filter(source == "ASLLRP")

ac_rest %>%
  inner_join(ac_bleurt %>% select(utterance_id, phenomenon, good_sentence, bad_sentence, ac_bleurt)) %>%
  group_by(phenomenon, inference_condition) %>%
  nest() %>%
  mutate(
    t_test = map(data, function(X){
      t.test(X$bleurt, X$ac_bleurt) %>%
        broom::tidy()
    })
  ) %>%
  unnest(t_test) %>%
  mutate(
    # padj_bh = p.adjust(p.value, method = "BH"),
    padj_bonf = p.adjust(p.value, method = "bonferroni")
  ) %>%
  arrange(phenomenon, inference_condition) %>% 
  mutate(
    significant = p.value < .05,
    # sign_bh = padj_bh < .05,
    sign_bonf = padj_bonf < .05
  ) %>% View()
