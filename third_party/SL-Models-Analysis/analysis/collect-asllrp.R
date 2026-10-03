library(tidyverse)

results <- fs::dir_ls("data/results/ASLLRP_Processed-refs/", regexp = "*.csv") %>%
  # keep(str_detect(., "_AC_\\d")) %>%
  map_df(read_csv, .id = "file") %>%
  mutate(
    training_condition = case_when(
      str_detect(file, "_AC_\\d") ~ "AC",
      str_detect(file, "_NF_\\d") ~ "NF",
      str_detect(file, "_NFB_\\d") ~ "NFB",
    ),
    inference_condition = str_extract(file, "(?<=0_)(.*)(?=\\.csv)"),
    phenomenon_family = case_when(
      phenomenon == "negation-reversed" ~ "negation",
      phenomenon == "yes/no-reversed" ~ "yes/no",
      TRUE ~ phenomenon
    )
  )

results %>%
  write_csv("data/asllrp-collective.csv")
