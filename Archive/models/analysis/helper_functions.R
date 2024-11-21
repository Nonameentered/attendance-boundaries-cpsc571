euclidean <- function(a, b)
  sqrt(sum((a - b) ^ 2))

samplewmean <- function(data, d) {
  return(weighted.mean(x = data[d, 1], w = data[d, 2]))
}

samplemean <- function(data, d) {
  return(mean(data[d], na.rm = T))
}

clean_district_name <- function(name) {
  name <- str_replace(name, 'Co', 'County')
  name <- str_replace(name, 'Pblc Schs', '')
}

count_total_additional_cross_cutting_exposures <-
  function(cats, df_best) {
    total_additional_xexposures <- c()
    for (c1 in cats) {
      curr_df <- df_best %>% filter(cat_best_for == c1)
      num_additional_xexposures <- 0
      for (c2 in cats) {
        num_additional_xexposures <-
          num_additional_xexposures + sum(curr_df[[paste(c2, 'total_xexposure_num_diff', sep =
                                                           '_')]], na.rm = T)
      }
      total_additional_xexposures <-
        c(total_additional_xexposures, num_additional_xexposures)
    }
    
    return (data.frame(cat = cats, total_additional_xexposures = total_additional_xexposures))
  }

get_vector_for_string_list <- function(curr_string) {
  string_of_vals <-
    substr(curr_string,
           2,
           nchar(curr_string) - 1)
  vector_of_vals <- strsplit(string_of_vals, ',')
  vector_of_vals <- as.numeric(vector_of_vals[[1]])
  return (vector_of_vals)
}

get_group_label <- function(cat, sep = '\n') {
  if (cat == 'frl') {
    return (sprintf('Free/Red. %sLunch', sep))
  }
  else if (cat == 'ell') {
    return (sprintf('English %sLearner', sep))
  }
  else if (cat == 'black' || cat == 'white' || cat == 'asian') {
    return (str_to_title(cat))
  }
  else if (cat == 'hisp') {
    return (sprintf('Hispanic/%sLatinx', sep))
  }
  else if (cat == 'native') {
    return (sprintf('Native%sAmerican', sep))
  }
}

get_group_plotting_color <- function(cat) {
  colors <- c('#d55e00', '#56b4e9', '#009e73', '#cc79a7', '#0072b2')
  if (cat == 'black') {
    return (colors[1])
  }
  else if (cat == 'ell') {
    return (colors[2])
  }
  else if (cat == 'frl') {
    return (colors[3])
  }
  else if (cat == 'hisp') {
    return (colors[4])
  }
  else if (cat == 'white') {
    return (colors[5])
  }
  else{
    return ('#000000')
  }
}

parse_and_agg_list_entries_xexposure_for_max_calc <-
  function(curr_df,
           status_quo_df,
           col_name,
           metric_for_exposure) {
    all_vals <- c()
    status_quo_vals <-
      get_vector_for_string_list(status_quo_df[[col_name]])
    curr_config_vals <-
      get_vector_for_string_list(curr_df[[col_name]])
    vals_to_add <- c()
    if (metric_for_exposure == 'diff') {
      vals_to_add <- curr_config_vals - status_quo_vals
    }
    else if (metric_for_exposure == 'change') {
      vals_to_add <-
        (curr_config_vals - status_quo_vals) / status_quo_vals
    }
    all_vals <- c(all_vals, vals_to_add)
    return (all_vals[!is.na(all_vals) & !is.infinite(all_vals)])
  }

parse_and_agg_list_entries_travel <- function(curr_df, col_name) {
  all_vals <- c()
  for (i in 1:length(curr_df[[col_name]])) {
    curr_config_vals <-
      get_vector_for_string_list(curr_df[[col_name]][i])
    all_vals <- c(all_vals, curr_config_vals)
  }
  return (all_vals[!is.na(all_vals) & !is.infinite(all_vals)])
}

parse_and_agg_list_entries_xexposure <-
  function(curr_df, col_name, metric_for_exposure) {
    all_vals <- c()
    for (i in 1:length(curr_df[[col_name]])) {
      status_quo_vals <-
        get_vector_for_string_list((
          df %>% filter(
            district_id == curr_df$district_id[i],
            config == 'status_quo_zoning'
          )
        )[[col_name]])
      curr_config_vals <-
        get_vector_for_string_list(curr_df[[col_name]][i])
      vals_to_add <- c()
      if (metric_for_exposure == 'diff') {
        vals_to_add <- curr_config_vals - status_quo_vals
      }
      else if (metric_for_exposure == 'change') {
        vals_to_add <-
          (curr_config_vals - status_quo_vals) / status_quo_vals
      }
      all_vals <- c(all_vals, vals_to_add)
    }
    return (all_vals[!is.na(all_vals) & !is.infinite(all_vals)])
  }

get_state_data <- function(data_dir_root) {
  states_list <-
    list.files(path = paste(data_dir_root, '2122', sep = ""))
  df_state_raw <- data.frame()
  for (s in states_list) {
    curr_df <-
      read.csv(
        paste(
          '../data/derived_data/2122/',
          s,
          '/schools_file_for_assignment.csv',
          sep = ""
        )
      )
    curr_df$ncessch <- format(curr_df$ncessch, scientific = F)
    df_state_raw <- rbind(df_state_raw, curr_df)
  }
  df_state_raw$ncessch <-
    format(df_state_raw$ncessch, scientific = F)
  return (df_state_raw)
}

get_district_metadata <- function(data_dir_root) {
  df_state_raw <- get_state_data(data_dir_root)
  df_dist_chars <- df_state_raw %>%
    group_by(leaid) %>%
    summarize(
      district_perwht = first(district_perwht),
      district_perblk = first(district_perblk),
      district_perfrl = first(district_perfrl),
      district_perell = first(district_perell),
      district_perhsp = first(district_perhsp),
      district_pernam = first(district_pernam),
      district_perasn = first(district_perasn),
      district_total_enroll = first(district_totenrl),
      district_name = first(LEA_NAME)
    )
  df_dist_chars$leaid <- str_pad(df_dist_chars$leaid, 7, pad = '0')
  df_dist_chars$leaid <- as.factor(df_dist_chars$leaid)
  df_dist_to_county <-
    read.csv('../data/school_covariates/district_county_mapping_2020.csv')
  df_dist_to_county$state_fips <-
    str_pad(df_dist_to_county$state_fips, 2, pad = '0')
  df_dist_to_county$district_fips <-
    str_pad(df_dist_to_county$district_fips, 5, pad = '0')
  df_dist_to_county$county_fips <-
    str_pad(df_dist_to_county$county_fips, 3, pad = '0')
  df_dist_to_county$leaid <-
    as.factor(paste(
      df_dist_to_county$state_fips,
      df_dist_to_county$district_fips,
      sep = ""
    ))
  df_dist_to_county$full_county_fips <-
    paste(df_dist_to_county$state_fips,
          df_dist_to_county$county_fips,
          sep = "")
  df_dist_to_county <- df_dist_to_county %>%
    group_by(leaid) %>%
    summarize(
      leaid = first(leaid),
      full_county_fips = first(full_county_fips),
      district_name = first(district_name),
      county_name = first(county_name),
    )
  df_county_pres <-
    read.csv('../data/school_covariates/countypres_2000-2020.csv')
  df_county_pres <- df_county_pres %>% filter(year == 2020)
  df_county_pres$full_county_fips <-
    str_pad(df_county_pres$county_fips, 5, pad = '0')
  df_county_pres <-
    df_county_pres %>% group_by(full_county_fips) %>% summarize(candidate_pref = party[which.max(candidatevotes)], state = first(state))
  df_county_urbanicity <-
    read.csv('../data/school_covariates/county_urbanicity_2013.csv')
  df_county_urbanicity$full_county_fips <-
    str_pad(df_county_urbanicity$county_fips, 5, pad = '0')
  urbanicity <- c()
  for (i in 1:length(df_county_urbanicity$code_2013)) {
    if (df_county_urbanicity$code_2013[i] == 1) {
      urbanicity <- c(urbanicity, 'urban')
    }
    else if (df_county_urbanicity$code_2013[i] == 2) {
      urbanicity <- c(urbanicity, 'suburban')
    }
    else if (df_county_urbanicity$code_2013[i] == 3 ||
             df_county_urbanicity$code_2013[i] == 4) {
      urbanicity <- c(urbanicity, 'small city')
    }
    else if (df_county_urbanicity$code_2013[i] == 5 ||
             df_county_urbanicity$code_2013[i] == 6) {
      urbanicity <- c(urbanicity, 'rural')
    }
    else{
      cat(i)
      urbanicity <- c(urbanicity, NA)
    }
  }
  df_county_urbanicity$urbanicity <- urbanicity
  df_dist_urbanicity <-
    df_dist_to_county %>% left_join(df_county_urbanicity, by = 'full_county_fips')
  df_dist_urbanicity <-
    df_dist_urbanicity %>% left_join(df_county_pres, by = 'full_county_fips')
  df_dist_urbanicity <-
    df_dist_urbanicity[c('leaid', 'urbanicity', 'candidate_pref')]
  df_dist_metadata <-
    df_dist_chars %>% left_join(df_dist_urbanicity, by = 'leaid')
  df_dist_metadata$urbanicity <-
    relevel(as.factor(df_dist_metadata$urbanicity), ref = 'rural')
  df_dist_metadata$candidate_pref <-
    relevel(as.factor(df_dist_metadata$candidate_pref), ref = 'REPUBLICAN')
  df_dist_metadata$leaid <- as.character(df_dist_metadata$leaid)
  return (df_dist_metadata)
}

get_main_df <- function(df_dist_metadata, data_dir_root) {
  # Solver config parameters
  cats = c('black', 'hisp', 'ell', 'frl', 'white', 'native', 'asian')
  
  # Outcomes
  metrics = c('total', 'median', 'mean', 'gini')
  values = c(
    'segregation',
    'travel_time',
    'normalized_exposure',
    'gini',
    'segregation_choice_0.5',
    'segregation_choice_1',
    'normalized_exposure_choice_0.5',
    'normalized_exposure_choice_1',
    'gini_choice_0.5',
    'gini_choice_1'
  )
  df <- data.frame()
  states_list <-
    list.files(path = paste(data_dir_root, '2122', sep = ""))
  
  for (s in states_list) {
    data_dir <- paste(data_dir_root, "2122/", s, "/", sep = "")
    file_list <- list.files(path = data_dir)
    
    for (f in file_list) {
      curr_df <- read.csv(paste(data_dir, f, sep = ""))
      curr_df$district_id <- as.factor(curr_df$district_id)
      status_quo <-
        curr_df %>% filter(curr_df$config == "status_quo_zoning")
      for (c in cats) {
        for (m in metrics) {
          for (v in values) {
            base_key <- paste(c, m, v, sep = "_")
            sq_base_key <- base_key
            # For the choice scenarios, set the base key to be the non choice version
            if (grepl('choice', v, '')) {
              # We only computed the choice stuff for the total metric
              if (m != 'total') {
                next
              }
              sq_base_key <-
                paste(c, m, str_split(v, "_choice")[[1]][1], sep = "_")
            }
            new_key <- paste(base_key, 'change', sep = "_")
            curr_df[[new_key]] <-
              (curr_df[[base_key]] - status_quo[[sq_base_key]]) / status_quo[[sq_base_key]]
            new_key <- paste(base_key, 'diff', sep = "_")
            curr_df[[new_key]] <-
              curr_df[[base_key]] - status_quo[[sq_base_key]]
          }
        }
        
      }
      df <- rbind(df, curr_df)
    }
  }
  
  is.na(df) <- sapply(df, is.infinite)
  df[is.na(df)] <- NA
  
  df$district_id <-
    as.character(str_pad(df$district_id, 7, pad = '0'))
  
  # Merge district metadata
  df <-
    df %>% left_join(df_dist_metadata, by = c('district_id' = 'leaid'))
  
  # Output CSV
  write.csv(df,
            paste(data_dir_root, "consolidated_simulation_results.csv", sep = ""))
  
  return (df)
}

compute_seg_histogram <- function(df_sq,
                                  baseline_outcome,
                                  curr_cat,
                                  curr_title,
                                  df_post = NULL,
                                  post_outcome = NULL) {
  label_offset_before <- 0.15
  label_offset_after <- 0.1
  if (baseline_outcome == 'total_normalized_exposure') {
    label_offset_before <- 0.1
    label_offset_after <- 0.07
  } else if (baseline_outcome == 'total_gini') {
    label_offset_before <- 0.2
    label_offset_after <- 0.2
  }
  
  curr_outcome_field <- paste(curr_cat, baseline_outcome, sep = "_")
  df_sq$key_outcome <- df_sq[[curr_outcome_field]]
  df_sq <- df_sq[c('district_id', 'key_outcome')]
  df_sq$time <- rep('before', times = nrow(df_sq))
  fill_vals <- c("#404080")
  combined_df <- df_sq
  median_before <- median(df_sq$key_outcome, na.rm = T)
  median_after <- NULL
  if (!is.null(df_post)) {
    curr_post_outcome <- paste(curr_cat, post_outcome, sep = "_")
    df_post$key_outcome <- df_post[[curr_post_outcome]]
    df_post <- df_post[c('district_id', 'key_outcome')]
    df_post$time <- rep('after' , times = nrow(df_post))
    combined_df <- rbind(combined_df, df_post)
    fill_vals <- c(fill_vals, "#69b3a2")
    median_after <- median(df_post$key_outcome, na.rm = T)
  }
  combined_df$time <-
    factor(combined_df$time, levels = c("before", "after"))
  curr_plot <- ggplot(combined_df , aes(x = key_outcome, fill = time)) +
    geom_histogram(
      color = "#e9ecef",
      alpha = 0.7,
      bins = 25,
      position = 'identity'
    ) +
    geom_vline(xintercept = median_before,
               linetype = 'dashed',
               color = "#404080") +
    annotate(
      "text",
      x = median_before + label_offset_before,
      y = Inf,
      label = sprintf("Median: %s", round(median_before, 2)),
      size = 4,
      color = "#404080",
      vjust = 1
    ) +
    
    {
      if (!is.null(df_post)) {
        geom_vline(xintercept = median_after,
                   linetype = 'dashed',
                   color = "#69b3a2")
      }
    } +
    {
      if (!is.null(df_post)) {
        annotate(
          "text",
          x = median_after - label_offset_after,
          y = Inf,
          label = sprintf("Median: %s", round(median_after, 2)),
          size = 4,
          color = "#69b3a2",
          vjust = 1
        )
      }
    } +
    labs(
      x = "",
      y = "",
      title = sprintf(curr_title,
                      get_group_label(curr_cat, sep = "")),
    ) +
    scale_fill_manual(values = fill_vals) +
    theme_ipsum() +
    {
      if (is.null(df_post)) {
        theme(legend.position = 'none')
      }
      else{
        labs(fill = "")
      }
      
    }
  
  
  return (curr_plot)
}
# District fixed effects, cluster standard errors at the district level
compute_regression_plot <-
  function(cats, outcome, title, df_reg = df) {
    plots <- list()
    reg_models <- list()
    legend_labels <- c()
    for (i in 1:length(cats)) {
      cat <- cats[i]
      legend_labels <- c(legend_labels, get_group_label(cat))
      if (outcome == 'total_travel_time_for_rezoned_diff') {
        df_reg[[paste(cat, outcome, sep = '_')]] <-
          (df_reg[[paste(cat, outcome, sep = '_')]] / df_reg[[paste(cat, 'num_rezoned', sep = "_")]]) / 60
      }
      df_reg$cat_optimizing_for <-
        as.factor(df_reg$cat_optimizing_for)
      df_reg$cat_optimizing_for <-
        relevel(df_reg$cat_optimizing_for, cat)
      curr_m <-
        felm(
          eval(parse(text = paste(
            cat, outcome, sep = '_'
          ))) ~ travel_time_threshold + school_size_threshold + community_cohesion_threshold + objective_function + cat_optimizing_for |
            district_id | 0 | district_id,
          data = df_reg
        )
      
      reg_models[[i]] <- curr_m
    }
    
    curr_plot <-
      plot_summs(
        reg_models,
        model.names = legend_labels,
        coefs = c(
          "Max % travel time increase" = "travel_time_threshold",
          "Max % school size increase" = "school_size_threshold",
          "Min % geographic cohesion" = "community_cohesion_threshold",
          "Objective: Max total" = "objective_functionmin_total",
          "Optimizing for Black" = "cat_optimizing_forblack",
          "Optimizing for Hispanic/Latinx" = "cat_optimizing_forhisp",
          "Optimizing for Free/Red. Lunch" = "cat_optimizing_forfrl",
          "Optimizing for English Learner" = "cat_optimizing_forell",
          "Optimizing for White" = "cat_optimizing_forwhite"
        ),
        legend.title = "Student group",
        colors = 'CUD'
      ) +
      labs(x = title) +
      geom_hline(yintercept = 0) + geom_vline(xintercept = 0)
    
    # return (list(plot = wrap_labs(curr_plot, 'normal'), models = reg_models))
    return (list(plot = curr_plot, 'normal', models = reg_models))
  }

get_best_configs <-
  function(df,
           outcome_cats,
           outcome_string,
           travel_time_condition = 0.5,
           school_size_condition = 0.15,
           cohesion_condition = 0.5,
           contiguity_condition = T,
           min_max_condition = F) {
    all_districts <- unique(df$district_id)
    
    df_best <- data.frame()
    for (d in all_districts) {
      df_dist <-
        df %>%
        filter(district_id == d, config != 'status_quo_zoning') %>%
        filter(travel_time_threshold <= travel_time_condition) %>%
        filter(school_size_threshold <= school_size_condition) %>%
        filter(community_cohesion_threshold >= cohesion_condition)
      
      if (contiguity_condition) {
        df_dist <- df_dist %>% filter(is_contiguous == "True")
      }
      
      if (min_max_condition) {
        df_dist <- df_dist %>% filter(objective_function == 'min_max')
      }
      
      for (c in outcome_cats) {
        col_name <- paste(c, outcome_string, sep = "_")
        best_ind <- which.min(df_dist[[col_name]])
        best_config <- df_dist[best_ind,]
        if (nrow(best_config) == 0) {
          next
        }
        best_config$cat_best_for <- c
        df_best <- rbind(df_best, best_config)
      }
    }
    
    return (df_best)
  }

compute_barplots_of_outcomes_best_configs <-
  function(df,
           opt_cats,
           outcome_cats,
           outcome_string,
           metric_for_exposure = 'diff') {
    estimates <- c()
    cat_best_for <- c()
    curr_outcome_cat <- c()
    ci_lower <- c()
    ci_upper <- c()
    plot_title <- ""
    
    for (i in 1:length(opt_cats)) {
      c1 <- opt_cats[i]
      curr_df <- df %>% filter(cat_best_for == c1)
      curr_df[is.na(curr_df)] <- 0
      for (j in 1:length(outcome_cats)) {
        c2 <- outcome_cats[j]
        col_name <- paste(c2, outcome_string, sep = "_")
        
        cat_best_for <- c(cat_best_for, get_group_label(c1))
        curr_outcome_cat <- c(curr_outcome_cat, get_group_label(c2))
        
        if (outcome_string == 'all_xexposure_prob') {
          exposure_vals <-
            100 * parse_and_agg_list_entries_xexposure(curr_df, col_name, metric_for_exposure)
          bootstrap_obj <-
            boot(exposure_vals, statistic = samplemean, R = 5000)
          bootstrap_ci <- boot.ci(bootstrap_obj)
          estimates <- c(estimates, mean(exposure_vals, na.rm = T))
          ci_lower <- c(ci_lower, bootstrap_ci$bca[4])
          ci_upper <- c(ci_upper, bootstrap_ci$bca[5])
          
          if (metric_for_exposure == 'diff') {
            plot_title <-
              TeX('Mean absolute $\\Delta$ in prob. of cross-cutting exposure')
          }
          else if (metric_for_exposure == 'change') {
            plot_title <-
              TeX('Mean relative $\\Delta$ in prob. of cross-cutting exposure')
          }
        }
        else if (outcome_string == 'total_xexposure_num_diff') {
          additional_exposures <- sum(curr_df[[col_name]])
          estimates <- c(estimates, additional_exposures)
          ci_lower <- c(ci_lower, NA)
          ci_upper <- c(ci_upper, NA)
          plot_title <-
            TeX('Total $\\Delta$ in cross-cutting exposures')
        }
        if (outcome_string == 'gini_xexposure_prob_diff') {
          bootstrap_obj <-
            boot(curr_df[[col_name]], statistic = samplemean, R = 5000)
          bootstrap_ci <- boot.ci(bootstrap_obj)
          mean_diff <- mean(curr_df[[col_name]], na.rm = T)
          estimates <- c(estimates, mean_diff)
          ci_lower <- c(ci_lower, bootstrap_ci$bca[4])
          ci_upper <- c(ci_upper, bootstrap_ci$bca[5])
          plot_title <-
            TeX('Mean absolute $\\Delta$ in gini of p(cross-cutting exposures)')
        }
        else if (outcome_string == 'total_segregation_diff' ||
                 outcome_string == 'total_normalized_exposure_diff' ||
                 outcome_string == 'total_gini_diff') {
          bootstrap_obj <-
            boot(curr_df[[col_name]], statistic = samplemean, R =
                   5000)
          bootstrap_ci <- boot.ci(bootstrap_obj)
          rezoned_percent <-
            mean(curr_df[[col_name]], na.rm = T)
          estimates <- c(estimates, rezoned_percent)
          ci_lower <- c(ci_lower, bootstrap_ci$bca[4])
          ci_upper <- c(ci_upper, bootstrap_ci$bca[5])
          plot_title <- 'Mean absolute change in segregation'
        }
        
        else if (outcome_string == 'total_segregation_change') {
          bootstrap_obj <-
            boot(curr_df[[col_name]], statistic = samplemean, R =
                   5000)
          bootstrap_ci <- boot.ci(bootstrap_obj)
          rezoned_percent <-
            mean(curr_df[[col_name]], na.rm = T)
          estimates <- c(estimates, rezoned_percent)
          ci_lower <- c(ci_lower, bootstrap_ci$bca[4])
          ci_upper <- c(ci_upper, bootstrap_ci$bca[5])
          plot_title <- 'Mean relative change in segregation'
        }
        else if (outcome_string == 'total_travel_time_for_rezoned_change') {
          bootstrap_obj <-
            boot(100 * curr_df[[col_name]], statistic = samplemean, R =
                   5000)
          bootstrap_ci <- boot.ci(bootstrap_obj)
          total_travel_time_change <-
            mean(100 * curr_df[[col_name]], na.rm = T)
          estimates <- c(estimates, total_travel_time_change)
          ci_lower <- c(ci_lower, bootstrap_ci$bca[4])
          ci_upper <- c(ci_upper, bootstrap_ci$bca[5])
          plot_title <-
            TeX('Mean relative $\\Delta$ in travel times')
        }
        else if (outcome_string == 'total_travel_time_for_rezoned_diff') {
          travel_time_changes <-
            curr_df[[col_name]] / curr_df[[paste(c2, 'num_rezoned', sep = "_")]]
          bootstrap_obj <-
            boot(travel_time_changes / 60,
                 statistic = samplemean,
                 R =
                   5000)
          bootstrap_ci <- boot.ci(bootstrap_obj)
          total_travel_time_change <-
            mean(travel_time_changes / 60, na.rm = T)
          estimates <- c(estimates, total_travel_time_change)
          ci_lower <- c(ci_lower, bootstrap_ci$bca[4])
          ci_upper <- c(ci_upper, bootstrap_ci$bca[5])
          plot_title <- 'Change in travel (minutes)'
        }
        else if (outcome_string == 'all_travel_time_for_rezoned') {
          travel_time_changes <-
            parse_and_agg_list_entries_travel(curr_df, col_name) / 60
          bootstrap_obj <-
            boot(travel_time_changes,
                 statistic = samplemean,
                 R = 5000)
          bootstrap_ci <- boot.ci(bootstrap_obj)
          all_travel_time_change <-
            mean(travel_time_changes / 60, na.rm = T)
          estimates <- c(estimates, all_travel_time_change)
          ci_lower <- c(ci_lower, bootstrap_ci$bca[4])
          ci_upper <- c(ci_upper, bootstrap_ci$bca[5])
          plot_title <-
            TeX('Mean $\\Delta$ in travel times (minutes)')
        }
        else if (outcome_string == 'gini_travel_time_for_rezoned_diff') {
          bootstrap_obj <-
            boot(curr_df[[col_name]], statistic = samplemean, R = 5000)
          bootstrap_ci <- boot.ci(bootstrap_obj)
          mean_value <- mean(curr_df[[col_name]], na.rm = T)
          estimates <- c(estimates, mean_value)
          ci_lower <- c(ci_lower, bootstrap_ci$bca[4])
          ci_upper <- c(ci_upper, bootstrap_ci$bca[5])
          plot_title <-
            TeX('Mean absolute $\\Delta$ in gini of travel times')
        }
        else if (outcome_string == 'percent_rezoned') {
          bootstrap_obj <-
            boot(100 * curr_df[[col_name]], statistic = samplemean, R =
                   5000)
          bootstrap_ci <- boot.ci(bootstrap_obj)
          rezoned_percent <-
            mean(100 * curr_df[[col_name]], na.rm = T)
          estimates <- c(estimates, rezoned_percent)
          ci_lower <- c(ci_lower, bootstrap_ci$bca[4])
          ci_upper <- c(ci_upper, bootstrap_ci$bca[5])
          plot_title <- 'Percent students rezoned'
        }
        
        # else if (outcome_string == 'mean_xexposure_prob_change') {
        #   bootstrap_obj <-
        #     boot(100 * curr_df[[col_name]], statistic = samplemean, R =
        #            5000)
        #   bootstrap_ci <- boot.ci(bootstrap_obj)
        #   rezoned_percent <-
        #     mean(100 * curr_df[[col_name]], na.rm = T)
        #   estimates <- c(estimates, rezoned_percent)
        #   ci_lower <- c(ci_lower, bootstrap_ci$bca[4])
        #   ci_upper <- c(ci_upper, bootstrap_ci$bca[5])
        #   plot_title <- 'Mean relative change in cross exposure prob'
        # }
        
      }
    }
    df_for_plotting <-
      data.frame(
        estimates = estimates,
        ci_lower = ci_lower,
        ci_upper = ci_upper,
        cat_best_for = cat_best_for,
        curr_outcome_cat = curr_outcome_cat
      )
    p <- df_for_plotting %>%
      ggplot(aes(x = cat_best_for,
                 y = estimates,
                 fill = curr_outcome_cat)) +
      geom_bar(
        position = position_dodge(),
        stat = "identity",
        alpha = 0.5,
        color = 'black',
      ) +
      scale_fill_manual(values = c('#d55e00', '#56b4e9', '#009e73', '#cc79a7', '#0072b2')) +
      {
        if (outcome_string != 'total_xexposure_num_diff')
          geom_errorbar(
            aes(ymin = ci_lower, ymax = ci_upper),
            width = 0.2,
            color = "gray",
            alpha = 0.8,
            size = 1,
            position = position_dodge(0.9),
            na.rm = T
          )
      } +
      # theme_void() +
      theme_ipsum() +
      labs(
        title = plot_title,
        x = '',
        y = '',
        fill = 'Student group'
      ) +
      ggtitle(plot_title) +
      theme(
        # axis.text.x = element_text(
        #   color = "black",
        #   size = 9,
        #   margin = margin(l = 2, r = 2)
        # ),
        axis.text.x = element_blank(),
        axis.text.y = element_text(color = "black", size = 9),
        axis.title.x = element_text(
          color = "black",
          size = 11,
          margin = margin(t = 5)
        ),
        legend.title = element_text(color = "black", size = 10),
        legend.spacing.y = unit(0.5, 'cm')
      ) +
      guides(fill = guide_legend(byrow = TRUE))
    
    return (p)
  }

compute_bubble_chart <-
  function(df_for_bubble,
           x,
           y,
           size,
           color,
           x_label,
           y_label,
           curr_title,
           show_correlation = F,
           show_median_change = F) {
    ref_line_vals <- seq(from = 0,
                         to = 1,
                         length.out = 10)
    df_identity <- data.frame(x = ref_line_vals, y = ref_line_vals)
    bubble_chart <- df_for_bubble %>%
      ggplot(aes_string(
        x = x,
        y = y,
        size = size,
        color = color
      )) +
      geom_point(alpha = 0.5) +
      {
        if (show_correlation) {
          annotate(
            "text",
            x = median(df_for_bubble[[x]], na.rm = T),
            y = Inf,
            label = TeX(paste0(
              "Spearman $\\rho$: ", round(
                cor.test(
                  df_for_bubble[[x]],
                  df_for_bubble[[y]],
                  method = 'spearman',
                  na.rm = T
                )$estimate,
                2
              )
            )),
            size = 4,
            color = "#000000",
            vjust = 1
          )
        }
      } +
      {
        if (show_median_change) {
          annotate(
            "text",
            x = median(df_for_bubble[[x]], na.rm = T),
            y = Inf,
            label = TeX(paste0(
              "Median relative change: ", round(median((df_for_bubble[[y]] - df_for_bubble[[x]]) / df_for_bubble[[x]],
                                                       na.rm = T
              ), 3) * 100, '%'
            )),
            size = 4,
            color = "#000000",
            vjust = 1
          )
        }
      } +
      scale_size(range = c(.1, 5), name = "District elementary enrollment") +
      geom_abline(intercept = 0, linetype = "dashed") +
      theme_ipsum() +
      theme(legend.position = "right") +
      labs(x = x_label,
           y = y_label,
           title = curr_title) +
      guides(fill = guide_legend(byrow = TRUE))
    # theme(legend.position = "none")
    return (bubble_chart)
  }

## Identifying "prototypical" districts in small city, suburban, urban areas
get_district_centroids <- function(df) {
  df_dist <-
    df %>% group_by(district_id) %>% summarize(
      district_perwht = first(district_perwht),
      district_perblk = first(district_perblk),
      district_perfrl = first(district_perfrl),
      district_perell = first(district_perell),
      district_perhsp = first(district_perhsp),
      total_enrollment_in_district = first(total_enrollment_in_district),
      num_schools_in_district = first(num_schools_in_district),
      candidate_pref = first(candidate_pref),
      urbanicity = first(urbanicity)
    )
  df_urban_full <-
    df_dist %>% filter(urbanicity == 'urban')
  df_suburban_full <-
    df_dist %>% filter(urbanicity == 'suburban')
  df_small_city_full <-
    df_dist %>% filter(urbanicity == 'small city')
  
  df_urban_nums <-
    df_urban_full[c(
      'district_perwht',
      'district_perblk',
      'district_perfrl',
      'district_perell',
      'district_perhsp',
      'total_enrollment_in_district',
      'num_schools_in_district'
    )]
  df_suburban_nums <-
    df_suburban_full[c(
      'district_perwht',
      'district_perblk',
      'district_perfrl',
      'district_perell',
      'district_perhsp',
      'total_enrollment_in_district',
      'num_schools_in_district'
    )]
  df_small_city_nums <-
    df_small_city_full[c(
      'district_perwht',
      'district_perblk',
      'district_perfrl',
      'district_perell',
      'district_perhsp',
      'total_enrollment_in_district',
      'num_schools_in_district'
    )]
  
  regional_dfs <-
    list(df_urban_nums, df_suburban_nums, df_small_city_nums)
  regional_centroids <- c()
  
  for (r_df in regional_dfs) {
    r_df_scaled <- data.frame(scale(r_df, center = T, scale = T))
    dists <- c()
    for (i in 1:nrow(r_df_scaled)) {
      dists <-
        c(dists, euclidean(as.numeric(r_df_scaled[i,]), as.numeric(sapply(
          r_df_scaled, mean
        ))))
    }
    regional_centroids <- c(regional_centroids, which.min(dists))
  }
  
  district_id_urban_centroid <-
    df_urban_full[regional_centroids[1],]$district_id
  district_id_suburban_centroid <-
    df_suburban_full[regional_centroids[2],]$district_id
  district_id_small_city_centroid <-
    df_small_city_full[regional_centroids[3],]$district_id
  
  return (
    list(
      district_id_urban_centroid = district_id_urban_centroid,
      district_id_suburban_centroid = district_id_suburban_centroid,
      district_id_small_city_centroid = district_id_small_city_centroid
    )
  )
  
}

### Case studies for prototypical districts (Figures 4-6)
compute_case_study_plots <-
  function(curr_district_id,
           district_type,
           df,
           df_best,
           data_dir_root,
           sim_root_dir) {
    # Identify
    all_cats <- c('asian', 'black', 'hisp', 'native', 'white')
    opt_cats <- data.frame(perkey = c('perwht'), catkey = c('white'))
    pers_and_cats <-
      data.frame(
        perkey = c('perasn', 'perblk', 'perhsp', 'pernam', 'perwht'),
        catkey = all_cats
      )
    seg_indices <- c()
    df_sq <-
      df %>% filter(config == 'status_quo_zoning',
                    district_id == curr_district_id)
    for (cat in opt_cats$catkey) {
      seg_indices <-
        c(seg_indices, df_sq[[paste(cat, 'total_segregation_diff', sep = "_")]])
    }
    cats <- as.character(opt_cats[which.min(seg_indices),])
    # cats <- c('perwht', 'white')
    
    df_best_curr <-
      df_best %>% filter(district_id == curr_district_id)
    curr_config <-
      (df_best_curr %>% filter(cat_best_for == cats[2]))
    
    solution_file_path <- sprintf(
      '%s/%s/%s/%s/solution_*.csv',
      sim_root_dir,
      curr_config$state,
      curr_district_id,
      curr_config$config
    )
    
    # cat(solution_file_path)
    # return ()
    soln_file <- Sys.glob(solution_file_path)
    df_rezoning <- read.csv(soln_file)
    # df_rezoning$ncessch <- str_pad(as.character(df_rezoning$ncessch), 12, pad="0")
    df_rezoning$leaid <-
      str_pad(as.character(df_rezoning$leaid), 7, pad = "0")
    df_rezoning$new_school_nces <-
      format(df_rezoning$new_school_nces, scientific = F)
    df_rezoning$new_school_nces <-
      str_pad(as.character(df_rezoning$new_school_nces), 12, pad = "0")
    df_rezoning_by_school <-
      df_rezoning %>%
      group_by(new_school_nces) %>%
      summarize(
        new_total_enrollment = sum(num_total_to_school),
        new_white_to_school = sum(num_white_to_school),
        new_black_to_school = sum(num_black_to_school),
        new_frl_to_school = sum(num_frl_to_school),
        new_ell_to_school = sum(num_ell_to_school),
        new_hispanic_to_school = sum(num_hispanic_to_school),
        new_native_to_school = sum(num_native_to_school),
        new_asian_to_school = sum(num_asian_to_school),
      )
    df_rezoning_by_school$new_perwht <-
      df_rezoning_by_school$new_white_to_school / df_rezoning_by_school$new_total_enrollment
    df_rezoning_by_school$new_perblk <-
      df_rezoning_by_school$new_black_to_school / df_rezoning_by_school$new_total_enrollment
    df_rezoning_by_school$new_perfrl <-
      df_rezoning_by_school$new_frl_to_school / df_rezoning_by_school$new_total_enrollment
    df_rezoning_by_school$new_perell <-
      df_rezoning_by_school$new_ell_to_school / df_rezoning_by_school$new_total_enrollment
    df_rezoning_by_school$new_perhsp <-
      df_rezoning_by_school$new_hispanic_to_school / df_rezoning_by_school$new_total_enrollment
    df_rezoning_by_school$new_pernam <-
      df_rezoning_by_school$new_native_to_school / df_rezoning_by_school$new_total_enrollment
    df_rezoning_by_school$new_perasn <-
      df_rezoning_by_school$new_asian_to_school / df_rezoning_by_school$new_total_enrollment
    df_rezoning_by_school <-
      df_rezoning_by_school[c(
        'new_school_nces',
        'new_total_enrollment',
        'new_perwht',
        'new_perblk',
        'new_perfrl',
        'new_perell',
        'new_perhsp',
        'new_pernam',
        'new_perasn'
      )]
    df_state_raw <- get_state_data(data_dir_root)
    df_state_raw$leaid <-
      str_pad(as.character(df_state_raw$leaid), 7, pad = "0")
    df_state_raw$ncessch <-
      str_pad(str_trim(as.character(df_state_raw$ncessch)), 12, pad = "0")
    df_schools <-
      df_state_raw %>%
      filter(leaid == curr_district_id) %>%
      group_by(ncessch) %>%
      summarize(
        orig_perwht = first(perwht),
        orig_perblk = first(perblk),
        orig_perhsp = first(perhsp),
        orig_perfrl = first(perfrl),
        orig_perell = first(perell),
        orig_pernam = first(pernam),
        orig_perasn = first(perasn),
        district_perwht = first(district_perwht),
        district_perblk = first(district_perblk),
        district_perhsp = first(district_perhsp),
        district_perfrl = first(district_perfrl),
        district_perell = first(district_perell),
        district_pernam = first(district_pernam),
        district_perasn = first(district_perasn),
        district_name = first(LEA_NAME),
      )
    df_schools$ncessch <-
      str_pad(as.character(df_schools$ncessch), 12, pad = "0")
    df_rezoning_by_school$new_school_nces <-
      str_pad(as.character(df_rezoning_by_school$new_school_nces),
              12,
              pad = "0")
    district_name <- df_schools$district_name[1]
    df_schools <-
      df_schools %>% left_join(df_rezoning_by_school, by = c('ncessch' = 'new_school_nces'))
    df_schools$diff_key <-
      df_schools[[paste('new', cats[1], sep = '_')]] - df_schools[[paste('orig', cats[1], sep =
                                                                           '_')]]
    df_schools[order(df_schools$diff_key),]
    df_schools_to_plot_1 <- data.frame(ncessch = df_schools$ncessch)
    df_schools_to_plot_2 <- data.frame(ncessch = df_schools$ncessch)
    df_schools_to_plot_1$perkey <-
      df_schools[[paste('orig', cats[1], sep = '_')]]
    df_schools_to_plot_1$time <- 'before'
    df_schools_to_plot_2$perkey <-
      df_schools[[paste('new', cats[1], sep = '_')]]
    df_schools_to_plot_2$time <- 'after'
    df_schools_to_plot <-
      rbind.data.frame(df_schools_to_plot_1, df_schools_to_plot_2)
    df_schools_to_plot$time <-
      factor(df_schools_to_plot$time, levels = c('before', 'after'))
    
    orig_zoning_png_path <- sprintf(
      '%s/%s/%s/%s/%s.png',
      sim_root_dir,
      curr_config$state,
      curr_district_id,
      curr_config$config,
      'original_zoning'
    )
    
    orig_zoning_image <-
      fig_lab(
        fig(
          orig_zoning_png_path,
          aspect.ratio = 'free',
          b_col = 'white',
          b_size = 0,
          b_margin = ggplot2::margin(0, 0, 0, 0)
        ),
        'Original attendance zones',
        pos = 'top',
        hjust = 0,
        size = 12,
        fontfamily = 'Helvetica'
      )
    
    cat_png_path <- sprintf(
      '%s/%s/%s/%s/%s.png',
      sim_root_dir,
      curr_config$state,
      curr_district_id,
      curr_config$config,
      paste('num', cats[2], sep = '_')
    )
    
    cat_image <-
      fig_lab(
        fig(
          cat_png_path,
          aspect.ratio = 'free',
          b_col = 'white',
          b_size = 0,
          b_margin = ggplot2::margin(0, 0, 0, 0)
        ),
        sprintf('%s population', get_group_label(cats[2])),
        pos = 'top',
        hjust = 0,
        size = 12,
        fontfamily = 'Helvetica'
      )
    
    rezoning_png_path <- sprintf(
      '%s/%s/%s/%s/%s.png',
      sim_root_dir,
      curr_config$state,
      curr_district_id,
      curr_config$config,
      'rezoning'
    )
    rezoning_image <-
      fig_lab(
        fig(
          rezoning_png_path,
          aspect.ratio = 'free',
          b_col = 'white',
          b_size = 0,
          b_margin = ggplot2::margin(0, 0, 0, 0)
        ),
        'Hypothetical rezoning',
        pos = 'top',
        hjust = 0,
        size = 12,
        fontfamily = 'Helvetica'
      )
    
    offset_val <- 1
    # if (district_type == 'small city') {
    #   offset_val <- 0
    # }
    
    before_after_fills <- c("#404080", "#69b3a2")
    
    schools_plot <- df_schools_to_plot %>%
      ggplot(aes(
        x = reorder(ncessch, desc(perkey), first),
        y = perkey,
        fill = time
      )) +
      geom_bar(alpha = 0.7,
               position = "dodge",
               stat = "identity") +
      scale_fill_manual(values = before_after_fills) +
      geom_hline(yintercept = df_schools[[paste('district', cats[1], sep = '_')]][1],
                 linetype = "dashed",
                 color = "black") +
      annotate(
        "text",
        x = nrow(df_schools) - offset_val,
        y = df_schools[[paste('district', cats[1], sep = '_')]][1],
        label = sprintf("District \n %% %s", cats[2]),
        size = 3
      ) +
      theme_void() +
      labs(title = sprintf(
        'Proportion %s students at each school (before and after rezoning)',
        cats[2]
      )) +
      theme(
        axis.text.y = element_text(color = "black", size = 9),
        plot.title = element_text(size = 12),
        legend.title = element_blank()
      )
    
    best_outcomes <-
      df_best_curr %>% filter(cat_best_for == cats[2])
    all_cats_for_display <- c()
    for (cat in all_cats) {
      all_cats_for_display <-
        c(all_cats_for_display,
          get_group_label(cat))
    }
    outcomes_df <-
      data.frame(cat = all_cats, cat_readable = all_cats_for_display)
    # num_exposures <- c()
    percent_rezoned <- c()
    travel_for_rezoned_diff <- c()
    seg_vals <- data.frame(cat_readable = c(),
                           time = c(),
                           seg_val = c())
    df_sq <-
      df %>% filter(district_id == curr_district_id,
                    config == 'status_quo_zoning')
    for (i in 1:length(all_cats)) {
      seg_before <-
        list(cat_readable = all_cats_for_display[i],
             time = 'before',
             seg_val = df_sq[[paste(all_cats[i], 'total_normalized_exposure', sep = '_')]])
      seg_after <-
        list(cat_readable = all_cats_for_display[i],
             time = 'after',
             seg_val = best_outcomes[[paste(all_cats[i], 'total_normalized_exposure', sep = '_')]])
      seg_vals <- rbind(seg_vals, seg_before)
      seg_vals <- rbind(seg_vals, seg_after)
      percent_rezoned <-
        c(percent_rezoned, best_outcomes[[paste(all_cats[i], 'percent_rezoned', sep =
                                                  '_')]])
      travel_to_append <-
        best_outcomes[[paste(all_cats[i], 'total_travel_time_for_rezoned_diff', sep = '_')]] / 60 / best_outcomes[[paste(all_cats[i], 'num_rezoned', sep = '_')]]
      travel_for_rezoned_diff <-
        c(travel_for_rezoned_diff, travel_to_append)
    }
    
    outcomes_df$percent_rezoned <- percent_rezoned
    outcomes_df$travel_for_rezoned_diff <- travel_for_rezoned_diff
    seg_vals$time <-
      factor(seg_vals$time, levels = c('before', 'after'))
    seg_change_plot <- seg_vals %>%
      ggplot(aes(x = cat_readable,
                 y = seg_val,
                 fill = time)) +
      geom_bar(alpha = 0.7,
               position = "dodge",
               stat = "identity") +
      scale_fill_manual(values = before_after_fills) +
      theme_void() +
      labs(title = 'Change in segregation by group') +
      theme(
        axis.text.x = element_text(color = "black", size = 9),
        axis.text.y = element_text(color = "black", size = 9),
        plot.title = element_text(size = 12),
        legend.title = element_blank()
      ) +
      theme(legend.position = 'none')
    # seg_change_plot <- wrap_labs(seg_change_plot, 'normal')
    
    percent_rezoned_plot <- outcomes_df %>%
      ggplot(aes(
        x = cat_readable,
        y = 100 * percent_rezoned,
        fill = cat_readable
      )) +
      geom_bar(alpha = 0.7,
               position = "dodge",
               stat = "identity") +
      scale_fill_manual(values = c('#d55e00', '#56b4e9', '#009e73', '#cc79a7', '#0072b2')) +
      theme_void() +
      labs(title = 'Percent rezoned') +
      theme(
        axis.text.x = element_text(color = "black", size = 9),
        axis.text.y = element_text(color = "black", size = 9),
        plot.title = element_text(size = 12)
      ) +
      theme(legend.position = 'none')
    # percent_rezoned_plot <- wrap_labs(percent_rezoned_plot, 'normal')
    
    travel_change_for_rezoned_plot <- outcomes_df %>%
      ggplot(aes(x = cat_readable,
                 y = travel_for_rezoned_diff,
                 fill = cat_readable)) +
      geom_bar(alpha = 0.7,
               position = "dodge",
               stat = "identity") +
      scale_fill_manual(values = c('#d55e00', '#56b4e9', '#009e73', '#cc79a7', '#0072b2')) +
      theme_void() +
      labs(title = 'Average change in travel time') +
      theme(
        axis.text.x = element_text(color = "black", size = 9),
        axis.text.y = element_text(color = "black", size = 9),
        plot.title = element_text(size = 12)
      ) +
      theme(legend.position = 'none')
    
    travel_change_for_rezoned_plot <-
      # wrap_labs(travel_change_for_rezoned_plot, 'normal')
      
      all_plots <-
      (
        (orig_zoning_image + cat_image + rezoning_image) / plot_spacer() / schools_plot / plot_spacer() / (
          seg_change_plot + percent_rezoned_plot + travel_change_for_rezoned_plot
        )
      ) +
      plot_layout(heights = c(8, 1, 6, 1 , 4)) +
      plot_annotation(
        title = sprintf(
          '%s case study: %s',
          district_type,
          clean_district_name(str_to_title(district_name))
        ),
        theme = theme(plot.title = element_text(size = 14)),
        tag_levels = 'a',
        tag_suffix = ')'
      ) &
      theme(plot.tag = element_text(size = 12))
    
    return (all_plots)
  }


analyze_school_demos_changes <- function(curr_df) {
  df_schools <-
    df_state_raw %>%
    filter(leaid %in% curr_df$district_id) %>%
    group_by(ncessch) %>%
    summarize(
      orig_perwht = first(perwht),
      orig_perblk = first(perblk),
      orig_perhsp = first(perhsp),
      orig_perfrl = first(perfrl),
      orig_perell = first(perell),
      district_perwht = first(district_perwht),
      district_perblk = first(district_perblk),
      district_perhsp = first(district_perhsp),
      district_perfrl = first(district_perfrl),
      district_perell = first(district_perell),
      district_name = first(LEA_NAME),
      district_id = first(leaid),
      orig_seg_index_perblk = first(seg_index_perblk),
      orig_seg_index_perwht = first(seg_index_perwht),
      orig_seg_index_perfrl = first(seg_index_perfrl),
      orig_seg_index_perell = first(seg_index_perell),
      orig_seg_index_perhsp = first(seg_index_perhsp),
      orig_total_enrollment = first(total_enrollment),
    )
  
  all_cats <- c('black', 'ell', 'frl', 'hisp', 'white')
  all_cats_for_display <- c()
  pers_and_cats <-
    data.frame(
      perkey = c('perblk', 'perell', 'perfrl', 'perhsp', 'perwht'),
      catkey = all_cats
    )
  
  all_districts <- unique(as.character(curr_df$district_id))
  black <- c()
  black_from_dist <- c()
  ell <- c()
  ell_from_dist <- c()
  frl <- c()
  frl_from_dist <- c()
  hisp <- c()
  hisp_from_dist <- c()
  white <- c()
  white_from_dist <- c()
  school_ids <- c()
  district_ids <- c()
  for (i in 1:length(all_districts)) {
    curr_district_id <- all_districts[i]
    for (j in 1:nrow(pers_and_cats)) {
      cats <- as.character(pers_and_cats[j, ])
      df_best_curr <-
        curr_df %>% filter(district_id == curr_district_id)
      curr_config <-
        (df_best_curr %>% filter(cat_best_for == cats[2]))
      
      solution_file_path <- sprintf(
        '%s/%s/%s/%s/solution_*.csv',
        sim_root_dir,
        curr_config$state,
        curr_district_id,
        curr_config$config
      )
      
      soln_file <- Sys.glob(solution_file_path)
      df_rezoning <- read.csv(soln_file)
      df_rezoning$new_school_nces <-
        format(df_rezoning$new_school_nces, scientific = F)
      df_rezoning_by_school <-
        df_rezoning %>%
        group_by(new_school_nces) %>%
        summarize(
          new_total_enrollment = sum(num_total_to_school),
          new_white_to_school = sum(num_white_to_school),
          new_black_to_school = sum(num_black_to_school),
          new_frl_to_school = sum(num_frl_to_school),
          new_ell_to_school = sum(num_ell_to_school),
          new_hispanic_to_school = sum(num_hispanic_to_school),
        )
      df_rezoning_by_school$new_perwht <-
        df_rezoning_by_school$new_white_to_school / df_rezoning_by_school$new_total_enrollment
      df_rezoning_by_school$new_perblk <-
        df_rezoning_by_school$new_black_to_school / df_rezoning_by_school$new_total_enrollment
      df_rezoning_by_school$new_perfrl <-
        df_rezoning_by_school$new_frl_to_school / df_rezoning_by_school$new_total_enrollment
      df_rezoning_by_school$new_perell <-
        df_rezoning_by_school$new_ell_to_school / df_rezoning_by_school$new_total_enrollment
      df_rezoning_by_school$new_perhsp <-
        df_rezoning_by_school$new_hispanic_to_school / df_rezoning_by_school$new_total_enrollment
      df_rezoning_by_school <-
        df_rezoning_by_school[c(
          'new_school_nces',
          'new_total_enrollment',
          'new_perwht',
          'new_perblk',
          'new_perfrl',
          'new_perell',
          'new_perhsp'
        )]
      
      df_schools_curr <-
        df_schools %>% inner_join(df_rezoning_by_school,
                                  by = c('ncessch' = 'new_school_nces'))
      diff_from_old <-
        abs(df_schools_curr[[paste('new', cats[1], sep = '_')]] - df_schools_curr[[paste('orig', cats[1], sep = '_')]])
      orig_diff_from_dist <-
        abs(df_schools_curr[[paste('orig', cats[1], sep = '_')]] - df_schools_curr[[paste('district', cats[1], sep = '_')]])
      
      assign(cats[2], c(get(cats[2]), diff_from_old))
      assign(paste(cats[2], 'from_dist', sep = '_'),
             c(get(paste(
               cats[2], 'from_dist', sep = '_'
             )), orig_diff_from_dist))
      
      # Doesn't matter which category we select here ... just doing this because we only want to store these values once
      if (cats[2] == 'black') {
        school_ids <- c(school_ids, df_schools_curr$ncessch)
        district_ids <- c(district_ids, df_schools_curr$district_id)
      }
    }
    
  }
  to_return <-
    list(
      black = black,
      black_from_dist = black_from_dist,
      ell = ell,
      ell_from_dist = ell_from_dist,
      frl = frl,
      frl_from_dist = frl_from_dist,
      hisp = hisp,
      hisp_from_dist = hisp_from_dist,
      white = white,
      white_from_dist = white_from_dist,
      school_id = school_ids,
      district_id = district_ids
    )
  return (to_return)
  
}
euclidean <- function(a, b)
  sqrt(sum((a - b) ^ 2))

samplewmean <- function(data, d) {
  return(weighted.mean(x = data[d, 1], w = data[d, 2]))
}

samplemean <- function(data, d) {
  return(mean(data[d], na.rm = T))
}

clean_district_name <- function(name) {
  name <- str_replace(name, 'Co', 'County')
  name <- str_replace(name, 'Pblc Schs', '')
}

count_total_additional_cross_cutting_exposures <-
  function(cats, df_best) {
    total_additional_xexposures <- c()
    for (c1 in cats) {
      curr_df <- df_best %>% filter(cat_best_for == c1)
      num_additional_xexposures <- 0
      for (c2 in cats) {
        num_additional_xexposures <-
          num_additional_xexposures + sum(curr_df[[paste(c2, 'total_xexposure_num_diff', sep =
                                                           '_')]], na.rm = T)
      }
      total_additional_xexposures <-
        c(total_additional_xexposures, num_additional_xexposures)
    }
    
    return (data.frame(cat = cats, total_additional_xexposures = total_additional_xexposures))
  }

get_vector_for_string_list <- function(curr_string) {
  string_of_vals <-
    substr(curr_string,
           2,
           nchar(curr_string) - 1)
  vector_of_vals <- strsplit(string_of_vals, ',')
  vector_of_vals <- as.numeric(vector_of_vals[[1]])
  return (vector_of_vals)
}

get_group_label <- function(cat, sep = '\n') {
  if (cat == 'frl') {
    return (sprintf('Free/Red. %sLunch', sep))
  }
  else if (cat == 'ell') {
    return (sprintf('English %sLearner', sep))
  }
  else if (cat == 'black' || cat == 'white' || cat == 'asian') {
    return (str_to_title(cat))
  }
  else if (cat == 'hisp') {
    return (sprintf('Hispanic/%sLatinx', sep))
  }
  else if (cat == 'native') {
    return (sprintf('Native%sAmerican', sep))
  }
}

get_group_plotting_color <- function(cat) {
  colors <- c('#d55e00', '#56b4e9', '#009e73', '#cc79a7', '#0072b2')
  if (cat == 'black') {
    return (colors[1])
  }
  else if (cat == 'ell') {
    return (colors[2])
  }
  else if (cat == 'frl') {
    return (colors[3])
  }
  else if (cat == 'hisp') {
    return (colors[4])
  }
  else if (cat == 'white') {
    return (colors[5])
  }
  else{
    return ('#000000')
  }
}

parse_and_agg_list_entries_xexposure_for_max_calc <-
  function(curr_df,
           status_quo_df,
           col_name,
           metric_for_exposure) {
    all_vals <- c()
    status_quo_vals <-
      get_vector_for_string_list(status_quo_df[[col_name]])
    curr_config_vals <-
      get_vector_for_string_list(curr_df[[col_name]])
    vals_to_add <- c()
    if (metric_for_exposure == 'diff') {
      vals_to_add <- curr_config_vals - status_quo_vals
    }
    else if (metric_for_exposure == 'change') {
      vals_to_add <-
        (curr_config_vals - status_quo_vals) / status_quo_vals
    }
    all_vals <- c(all_vals, vals_to_add)
    return (all_vals[!is.na(all_vals) & !is.infinite(all_vals)])
  }

parse_and_agg_list_entries_travel <- function(curr_df, col_name) {
  all_vals <- c()
  for (i in 1:length(curr_df[[col_name]])) {
    curr_config_vals <-
      get_vector_for_string_list(curr_df[[col_name]][i])
    all_vals <- c(all_vals, curr_config_vals)
  }
  return (all_vals[!is.na(all_vals) & !is.infinite(all_vals)])
}

parse_and_agg_list_entries_xexposure <-
  function(curr_df, col_name, metric_for_exposure) {
    all_vals <- c()
    for (i in 1:length(curr_df[[col_name]])) {
      status_quo_vals <-
        get_vector_for_string_list((
          df %>% filter(
            district_id == curr_df$district_id[i],
            config == 'status_quo_zoning'
          )
        )[[col_name]])
      curr_config_vals <-
        get_vector_for_string_list(curr_df[[col_name]][i])
      vals_to_add <- c()
      if (metric_for_exposure == 'diff') {
        vals_to_add <- curr_config_vals - status_quo_vals
      }
      else if (metric_for_exposure == 'change') {
        vals_to_add <-
          (curr_config_vals - status_quo_vals) / status_quo_vals
      }
      all_vals <- c(all_vals, vals_to_add)
    }
    return (all_vals[!is.na(all_vals) & !is.infinite(all_vals)])
  }

get_state_data <- function(data_dir_root) {
  states_list <-
    list.files(path = paste(data_dir_root, '2122', sep = ""))
  df_state_raw <- data.frame()
  for (s in states_list) {
    curr_df <-
      read.csv(
        paste(
          '../data/derived_data/2122/',
          s,
          '/schools_file_for_assignment.csv',
          sep = ""
        )
      )
    curr_df$ncessch <- format(curr_df$ncessch, scientific = F)
    df_state_raw <- rbind(df_state_raw, curr_df)
  }
  df_state_raw$ncessch <-
    format(df_state_raw$ncessch, scientific = F)
  return (df_state_raw)
}

get_district_metadata <- function(data_dir_root) {
  df_state_raw <- get_state_data(data_dir_root)
  df_dist_chars <- df_state_raw %>%
    group_by(leaid) %>%
    summarize(
      district_perwht = first(district_perwht),
      district_perblk = first(district_perblk),
      district_perfrl = first(district_perfrl),
      district_perell = first(district_perell),
      district_perhsp = first(district_perhsp),
      district_pernam = first(district_pernam),
      district_perasn = first(district_perasn),
      district_total_enroll = first(district_totenrl),
      district_name = first(LEA_NAME)
    )
  df_dist_chars$leaid <- str_pad(df_dist_chars$leaid, 7, pad = '0')
  df_dist_chars$leaid <- as.factor(df_dist_chars$leaid)
  df_dist_to_county <-
    read.csv('../data/school_covariates/district_county_mapping_2020.csv')
  df_dist_to_county$state_fips <-
    str_pad(df_dist_to_county$state_fips, 2, pad = '0')
  df_dist_to_county$district_fips <-
    str_pad(df_dist_to_county$district_fips, 5, pad = '0')
  df_dist_to_county$county_fips <-
    str_pad(df_dist_to_county$county_fips, 3, pad = '0')
  df_dist_to_county$leaid <-
    as.factor(paste(
      df_dist_to_county$state_fips,
      df_dist_to_county$district_fips,
      sep = ""
    ))
  df_dist_to_county$full_county_fips <-
    paste(df_dist_to_county$state_fips,
          df_dist_to_county$county_fips,
          sep = "")
  df_dist_to_county <- df_dist_to_county %>%
    group_by(leaid) %>%
    summarize(
      leaid = first(leaid),
      full_county_fips = first(full_county_fips),
      district_name = first(district_name),
      county_name = first(county_name),
    )
  df_county_pres <-
    read.csv('../data/school_covariates/countypres_2000-2020.csv')
  df_county_pres <- df_county_pres %>% filter(year == 2020)
  df_county_pres$full_county_fips <-
    str_pad(df_county_pres$county_fips, 5, pad = '0')
  df_county_pres <-
    df_county_pres %>% group_by(full_county_fips) %>% summarize(candidate_pref = party[which.max(candidatevotes)], state = first(state))
  df_county_urbanicity <-
    read.csv('../data/school_covariates/county_urbanicity_2013.csv')
  df_county_urbanicity$full_county_fips <-
    str_pad(df_county_urbanicity$county_fips, 5, pad = '0')
  urbanicity <- c()
  for (i in 1:length(df_county_urbanicity$code_2013)) {
    if (df_county_urbanicity$code_2013[i] == 1) {
      urbanicity <- c(urbanicity, 'urban')
    }
    else if (df_county_urbanicity$code_2013[i] == 2) {
      urbanicity <- c(urbanicity, 'suburban')
    }
    else if (df_county_urbanicity$code_2013[i] == 3 ||
             df_county_urbanicity$code_2013[i] == 4) {
      urbanicity <- c(urbanicity, 'small city')
    }
    else if (df_county_urbanicity$code_2013[i] == 5 ||
             df_county_urbanicity$code_2013[i] == 6) {
      urbanicity <- c(urbanicity, 'rural')
    }
    else{
      cat(i)
      urbanicity <- c(urbanicity, NA)
    }
  }
  df_county_urbanicity$urbanicity <- urbanicity
  df_dist_urbanicity <-
    df_dist_to_county %>% left_join(df_county_urbanicity, by = 'full_county_fips')
  df_dist_urbanicity <-
    df_dist_urbanicity %>% left_join(df_county_pres, by = 'full_county_fips')
  df_dist_urbanicity <-
    df_dist_urbanicity[c('leaid', 'urbanicity', 'candidate_pref')]
  df_dist_metadata <-
    df_dist_chars %>% left_join(df_dist_urbanicity, by = 'leaid')
  df_dist_metadata$urbanicity <-
    relevel(as.factor(df_dist_metadata$urbanicity), ref = 'rural')
  df_dist_metadata$candidate_pref <-
    relevel(as.factor(df_dist_metadata$candidate_pref), ref = 'REPUBLICAN')
  df_dist_metadata$leaid <- as.character(df_dist_metadata$leaid)
  return (df_dist_metadata)
}

get_main_df <- function(df_dist_metadata, data_dir_root) {
  # Solver config parameters
  cats = c('black', 'hisp', 'ell', 'frl', 'white', 'native', 'asian')
  
  # Outcomes
  metrics = c('total', 'median', 'mean', 'gini')
  values = c(
    'segregation',
    'travel_time',
    'normalized_exposure',
    'gini',
    'segregation_choice_0.5',
    'segregation_choice_1',
    'normalized_exposure_choice_0.5',
    'normalized_exposure_choice_1',
    'gini_choice_0.5',
    'gini_choice_1'
  )
  df <- data.frame()
  states_list <-
    list.files(path = paste(data_dir_root, '2122', sep = ""))
  
  for (s in states_list) {
    data_dir <- paste(data_dir_root, "2122/", s, "/", sep = "")
    file_list <- list.files(path = data_dir)
    
    for (f in file_list) {
      curr_df <- read.csv(paste(data_dir, f, sep = ""))
      curr_df$district_id <- as.factor(curr_df$district_id)
      status_quo <-
        curr_df %>% filter(curr_df$config == "status_quo_zoning")
      for (c in cats) {
        for (m in metrics) {
          for (v in values) {
            base_key <- paste(c, m, v, sep = "_")
            sq_base_key <- base_key
            # For the choice scenarios, set the base key to be the non choice version
            if (grepl('choice', v, '')) {
              # We only computed the choice stuff for the total metric
              if (m != 'total') {
                next
              }
              sq_base_key <-
                paste(c, m, str_split(v, "_choice")[[1]][1], sep = "_")
            }
            new_key <- paste(base_key, 'change', sep = "_")
            curr_df[[new_key]] <-
              (curr_df[[base_key]] - status_quo[[sq_base_key]]) / status_quo[[sq_base_key]]
            new_key <- paste(base_key, 'diff', sep = "_")
            curr_df[[new_key]] <-
              curr_df[[base_key]] - status_quo[[sq_base_key]]
          }
        }
        
      }
      df <- rbind(df, curr_df)
    }
  }
  
  is.na(df) <- sapply(df, is.infinite)
  df[is.na(df)] <- NA
  
  df$district_id <-
    as.character(str_pad(df$district_id, 7, pad = '0'))
  
  # Merge district metadata
  df <-
    df %>% left_join(df_dist_metadata, by = c('district_id' = 'leaid'))
  
  # Output CSV
  write.csv(df,
            paste(data_dir_root, "consolidated_simulation_results.csv", sep = ""))
  
  return (df)
}

compute_seg_histogram <- function(df_sq,
                                  baseline_outcome,
                                  curr_cat,
                                  curr_title,
                                  df_post = NULL,
                                  post_outcome = NULL) {
  label_offset_before <- 0.15
  label_offset_after <- 0.1
  if (baseline_outcome == 'total_normalized_exposure') {
    label_offset_before <- 0.1
    label_offset_after <- 0.07
  } else if (baseline_outcome == 'total_gini') {
    label_offset_before <- 0.2
    label_offset_after <- 0.2
  }
  
  curr_outcome_field <- paste(curr_cat, baseline_outcome, sep = "_")
  df_sq$key_outcome <- df_sq[[curr_outcome_field]]
  df_sq <- df_sq[c('district_id', 'key_outcome')]
  df_sq$time <- rep('before', times = nrow(df_sq))
  fill_vals <- c("#404080")
  combined_df <- df_sq
  median_before <- median(df_sq$key_outcome, na.rm = T)
  median_after <- NULL
  if (!is.null(df_post)) {
    curr_post_outcome <- paste(curr_cat, post_outcome, sep = "_")
    df_post$key_outcome <- df_post[[curr_post_outcome]]
    df_post <- df_post[c('district_id', 'key_outcome')]
    df_post$time <- rep('after' , times = nrow(df_post))
    combined_df <- rbind(combined_df, df_post)
    fill_vals <- c(fill_vals, "#69b3a2")
    median_after <- median(df_post$key_outcome, na.rm = T)
  }
  combined_df$time <-
    factor(combined_df$time, levels = c("before", "after"))
  curr_plot <- ggplot(combined_df , aes(x = key_outcome, fill = time)) +
    geom_histogram(
      color = "#e9ecef",
      alpha = 0.7,
      bins = 25,
      position = 'identity'
    ) +
    geom_vline(xintercept = median_before,
               linetype = 'dashed',
               color = "#404080") +
    annotate(
      "text",
      x = median_before + label_offset_before,
      y = Inf,
      label = sprintf("Median: %s", round(median_before, 2)),
      size = 4,
      color = "#404080",
      vjust = 1
    ) +
    
    {
      if (!is.null(df_post)) {
        geom_vline(xintercept = median_after,
                   linetype = 'dashed',
                   color = "#69b3a2")
      }
    } +
    {
      if (!is.null(df_post)) {
        annotate(
          "text",
          x = median_after - label_offset_after,
          y = Inf,
          label = sprintf("Median: %s", round(median_after, 2)),
          size = 4,
          color = "#69b3a2",
          vjust = 1
        )
      }
    } +
    labs(
      x = "",
      y = "",
      title = sprintf(curr_title,
                      get_group_label(curr_cat, sep = "")),
    ) +
    scale_fill_manual(values = fill_vals) +
    theme_ipsum() +
    {
      if (is.null(df_post)) {
        theme(legend.position = 'none')
      }
      else{
        labs(fill = "")
      }
      
    }
  
  
  return (curr_plot)
}
# District fixed effects, cluster standard errors at the district level
compute_regression_plot <-
  function(cats, outcome, title, df_reg = df) {
    plots <- list()
    reg_models <- list()
    legend_labels <- c()
    for (i in 1:length(cats)) {
      cat <- cats[i]
      legend_labels <- c(legend_labels, get_group_label(cat))
      if (outcome == 'total_travel_time_for_rezoned_diff') {
        df_reg[[paste(cat, outcome, sep = '_')]] <-
          (df_reg[[paste(cat, outcome, sep = '_')]] / df_reg[[paste(cat, 'num_rezoned', sep = "_")]]) / 60
      }
      df_reg$cat_optimizing_for <-
        as.factor(df_reg$cat_optimizing_for)
      df_reg$cat_optimizing_for <-
        relevel(df_reg$cat_optimizing_for, cat)
      curr_m <-
        felm(
          eval(parse(text = paste(
            cat, outcome, sep = '_'
          ))) ~ travel_time_threshold + school_size_threshold + community_cohesion_threshold + objective_function + cat_optimizing_for |
            district_id | 0 | district_id,
          data = df_reg
        )
      
      reg_models[[i]] <- curr_m
    }
    
    curr_plot <-
      plot_summs(
        reg_models,
        model.names = legend_labels,
        coefs = c(
          "Max % travel time increase" = "travel_time_threshold",
          "Max % school size increase" = "school_size_threshold",
          "Min % geographic cohesion" = "community_cohesion_threshold",
          "Objective: Max total" = "objective_functionmin_total",
          "Optimizing for Black" = "cat_optimizing_forblack",
          "Optimizing for Hispanic/Latinx" = "cat_optimizing_forhisp",
          "Optimizing for Free/Red. Lunch" = "cat_optimizing_forfrl",
          "Optimizing for English Learner" = "cat_optimizing_forell",
          "Optimizing for White" = "cat_optimizing_forwhite"
        ),
        legend.title = "Student group",
        colors = 'CUD'
      ) +
      labs(x = title) +
      geom_hline(yintercept = 0) + geom_vline(xintercept = 0)
    
    # return (list(plot = wrap_labs(curr_plot, 'normal'), models = reg_models))
    return (list(plot = curr_plot, 'normal', models = reg_models))
  }

get_best_configs <-
  function(df,
           outcome_cats,
           outcome_string,
           travel_time_condition = 0.5,
           school_size_condition = 0.15,
           cohesion_condition = 0.5,
           contiguity_condition = T,
           min_max_condition = F) {
    all_districts <- unique(df$district_id)
    
    df_best <- data.frame()
    for (d in all_districts) {
      df_dist <-
        df %>%
        filter(district_id == d, config != 'status_quo_zoning') %>%
        filter(travel_time_threshold <= travel_time_condition) %>%
        filter(school_size_threshold <= school_size_condition) %>%
        filter(community_cohesion_threshold >= cohesion_condition)
      
      if (contiguity_condition) {
        df_dist <- df_dist %>% filter(is_contiguous == "True")
      }
      
      if (min_max_condition) {
        df_dist <- df_dist %>% filter(objective_function == 'min_max')
      }
      
      for (c in outcome_cats) {
        col_name <- paste(c, outcome_string, sep = "_")
        best_ind <- which.min(df_dist[[col_name]])
        best_config <- df_dist[best_ind,]
        if (nrow(best_config) == 0) {
          next
        }
        best_config$cat_best_for <- c
        df_best <- rbind(df_best, best_config)
      }
    }
    
    return (df_best)
  }

compute_barplots_of_outcomes_best_configs <-
  function(df,
           opt_cats,
           outcome_cats,
           outcome_string,
           metric_for_exposure = 'diff') {
    estimates <- c()
    cat_best_for <- c()
    curr_outcome_cat <- c()
    ci_lower <- c()
    ci_upper <- c()
    plot_title <- ""
    
    for (i in 1:length(opt_cats)) {
      c1 <- opt_cats[i]
      curr_df <- df %>% filter(cat_best_for == c1)
      curr_df[is.na(curr_df)] <- 0
      for (j in 1:length(outcome_cats)) {
        c2 <- outcome_cats[j]
        col_name <- paste(c2, outcome_string, sep = "_")
        
        cat_best_for <- c(cat_best_for, get_group_label(c1))
        curr_outcome_cat <- c(curr_outcome_cat, get_group_label(c2))
        
        if (outcome_string == 'all_xexposure_prob') {
          exposure_vals <-
            100 * parse_and_agg_list_entries_xexposure(curr_df, col_name, metric_for_exposure)
          bootstrap_obj <-
            boot(exposure_vals, statistic = samplemean, R = 5000)
          bootstrap_ci <- boot.ci(bootstrap_obj)
          estimates <- c(estimates, mean(exposure_vals, na.rm = T))
          ci_lower <- c(ci_lower, bootstrap_ci$bca[4])
          ci_upper <- c(ci_upper, bootstrap_ci$bca[5])
          
          if (metric_for_exposure == 'diff') {
            plot_title <-
              TeX('Mean absolute $\\Delta$ in prob. of cross-cutting exposure')
          }
          else if (metric_for_exposure == 'change') {
            plot_title <-
              TeX('Mean relative $\\Delta$ in prob. of cross-cutting exposure')
          }
        }
        else if (outcome_string == 'total_xexposure_num_diff') {
          additional_exposures <- sum(curr_df[[col_name]])
          estimates <- c(estimates, additional_exposures)
          ci_lower <- c(ci_lower, NA)
          ci_upper <- c(ci_upper, NA)
          plot_title <-
            TeX('Total $\\Delta$ in cross-cutting exposures')
        }
        if (outcome_string == 'gini_xexposure_prob_diff') {
          bootstrap_obj <-
            boot(curr_df[[col_name]], statistic = samplemean, R = 5000)
          bootstrap_ci <- boot.ci(bootstrap_obj)
          mean_diff <- mean(curr_df[[col_name]], na.rm = T)
          estimates <- c(estimates, mean_diff)
          ci_lower <- c(ci_lower, bootstrap_ci$bca[4])
          ci_upper <- c(ci_upper, bootstrap_ci$bca[5])
          plot_title <-
            TeX('Mean absolute $\\Delta$ in gini of p(cross-cutting exposures)')
        }
        else if (outcome_string == 'total_segregation_diff' ||
                 outcome_string == 'total_normalized_exposure_diff' ||
                 outcome_string == 'total_gini_diff') {
          bootstrap_obj <-
            boot(curr_df[[col_name]], statistic = samplemean, R =
                   5000)
          bootstrap_ci <- boot.ci(bootstrap_obj)
          rezoned_percent <-
            mean(curr_df[[col_name]], na.rm = T)
          estimates <- c(estimates, rezoned_percent)
          ci_lower <- c(ci_lower, bootstrap_ci$bca[4])
          ci_upper <- c(ci_upper, bootstrap_ci$bca[5])
          plot_title <- 'Mean absolute change in segregation'
        }
        
        else if (outcome_string == 'total_segregation_change') {
          bootstrap_obj <-
            boot(curr_df[[col_name]], statistic = samplemean, R =
                   5000)
          bootstrap_ci <- boot.ci(bootstrap_obj)
          rezoned_percent <-
            mean(curr_df[[col_name]], na.rm = T)
          estimates <- c(estimates, rezoned_percent)
          ci_lower <- c(ci_lower, bootstrap_ci$bca[4])
          ci_upper <- c(ci_upper, bootstrap_ci$bca[5])
          plot_title <- 'Mean relative change in segregation'
        }
        else if (outcome_string == 'total_travel_time_for_rezoned_change') {
          bootstrap_obj <-
            boot(100 * curr_df[[col_name]], statistic = samplemean, R =
                   5000)
          bootstrap_ci <- boot.ci(bootstrap_obj)
          total_travel_time_change <-
            mean(100 * curr_df[[col_name]], na.rm = T)
          estimates <- c(estimates, total_travel_time_change)
          ci_lower <- c(ci_lower, bootstrap_ci$bca[4])
          ci_upper <- c(ci_upper, bootstrap_ci$bca[5])
          plot_title <-
            TeX('Mean relative $\\Delta$ in travel times')
        }
        else if (outcome_string == 'total_travel_time_for_rezoned_diff') {
          travel_time_changes <-
            curr_df[[col_name]] / curr_df[[paste(c2, 'num_rezoned', sep = "_")]]
          bootstrap_obj <-
            boot(travel_time_changes / 60,
                 statistic = samplemean,
                 R =
                   5000)
          bootstrap_ci <- boot.ci(bootstrap_obj)
          total_travel_time_change <-
            mean(travel_time_changes / 60, na.rm = T)
          estimates <- c(estimates, total_travel_time_change)
          ci_lower <- c(ci_lower, bootstrap_ci$bca[4])
          ci_upper <- c(ci_upper, bootstrap_ci$bca[5])
          plot_title <- 'Change in travel (minutes)'
        }
        else if (outcome_string == 'all_travel_time_for_rezoned') {
          travel_time_changes <-
            parse_and_agg_list_entries_travel(curr_df, col_name) / 60
          bootstrap_obj <-
            boot(travel_time_changes,
                 statistic = samplemean,
                 R = 5000)
          bootstrap_ci <- boot.ci(bootstrap_obj)
          all_travel_time_change <-
            mean(travel_time_changes / 60, na.rm = T)
          estimates <- c(estimates, all_travel_time_change)
          ci_lower <- c(ci_lower, bootstrap_ci$bca[4])
          ci_upper <- c(ci_upper, bootstrap_ci$bca[5])
          plot_title <-
            TeX('Mean $\\Delta$ in travel times (minutes)')
        }
        else if (outcome_string == 'gini_travel_time_for_rezoned_diff') {
          bootstrap_obj <-
            boot(curr_df[[col_name]], statistic = samplemean, R = 5000)
          bootstrap_ci <- boot.ci(bootstrap_obj)
          mean_value <- mean(curr_df[[col_name]], na.rm = T)
          estimates <- c(estimates, mean_value)
          ci_lower <- c(ci_lower, bootstrap_ci$bca[4])
          ci_upper <- c(ci_upper, bootstrap_ci$bca[5])
          plot_title <-
            TeX('Mean absolute $\\Delta$ in gini of travel times')
        }
        else if (outcome_string == 'percent_rezoned') {
          bootstrap_obj <-
            boot(100 * curr_df[[col_name]], statistic = samplemean, R =
                   5000)
          bootstrap_ci <- boot.ci(bootstrap_obj)
          rezoned_percent <-
            mean(100 * curr_df[[col_name]], na.rm = T)
          estimates <- c(estimates, rezoned_percent)
          ci_lower <- c(ci_lower, bootstrap_ci$bca[4])
          ci_upper <- c(ci_upper, bootstrap_ci$bca[5])
          plot_title <- 'Percent students rezoned'
        }
        
        # else if (outcome_string == 'mean_xexposure_prob_change') {
        #   bootstrap_obj <-
        #     boot(100 * curr_df[[col_name]], statistic = samplemean, R =
        #            5000)
        #   bootstrap_ci <- boot.ci(bootstrap_obj)
        #   rezoned_percent <-
        #     mean(100 * curr_df[[col_name]], na.rm = T)
        #   estimates <- c(estimates, rezoned_percent)
        #   ci_lower <- c(ci_lower, bootstrap_ci$bca[4])
        #   ci_upper <- c(ci_upper, bootstrap_ci$bca[5])
        #   plot_title <- 'Mean relative change in cross exposure prob'
        # }
        
      }
    }
    df_for_plotting <-
      data.frame(
        estimates = estimates,
        ci_lower = ci_lower,
        ci_upper = ci_upper,
        cat_best_for = cat_best_for,
        curr_outcome_cat = curr_outcome_cat
      )
    p <- df_for_plotting %>%
      ggplot(aes(x = cat_best_for,
                 y = estimates,
                 fill = curr_outcome_cat)) +
      geom_bar(
        position = position_dodge(),
        stat = "identity",
        alpha = 0.5,
        color = 'black',
      ) +
      scale_fill_manual(values = c('#d55e00', '#56b4e9', '#009e73', '#cc79a7', '#0072b2')) +
      {
        if (outcome_string != 'total_xexposure_num_diff')
          geom_errorbar(
            aes(ymin = ci_lower, ymax = ci_upper),
            width = 0.2,
            color = "gray",
            alpha = 0.8,
            size = 1,
            position = position_dodge(0.9),
            na.rm = T
          )
      } +
      # theme_void() +
      theme_ipsum() +
      labs(
        title = plot_title,
        x = '',
        y = '',
        fill = 'Student group'
      ) +
      ggtitle(plot_title) +
      theme(
        # axis.text.x = element_text(
        #   color = "black",
        #   size = 9,
        #   margin = margin(l = 2, r = 2)
        # ),
        axis.text.x = element_blank(),
        axis.text.y = element_text(color = "black", size = 9),
        axis.title.x = element_text(
          color = "black",
          size = 11,
          margin = margin(t = 5)
        ),
        legend.title = element_text(color = "black", size = 10),
        legend.spacing.y = unit(0.5, 'cm')
      ) +
      guides(fill = guide_legend(byrow = TRUE))
    
    return (p)
  }

compute_bubble_chart <-
  function(df_for_bubble,
           x,
           y,
           size,
           color,
           x_label,
           y_label,
           curr_title,
           show_correlation = F,
           show_median_change = F) {
    ref_line_vals <- seq(from = 0,
                         to = 1,
                         length.out = 10)
    df_identity <- data.frame(x = ref_line_vals, y = ref_line_vals)
    bubble_chart <- df_for_bubble %>%
      ggplot(aes_string(
        x = x,
        y = y,
        size = size,
        color = color
      )) +
      geom_point(alpha = 0.5) +
      {
        if (show_correlation) {
          annotate(
            "text",
            x = median(df_for_bubble[[x]], na.rm = T),
            y = Inf,
            label = TeX(paste0(
              "Spearman $\\rho$: ", round(
                cor.test(
                  df_for_bubble[[x]],
                  df_for_bubble[[y]],
                  method = 'spearman',
                  na.rm = T
                )$estimate,
                2
              )
            )),
            size = 4,
            color = "#000000",
            vjust = 1
          )
        }
      } +
      {
        if (show_median_change) {
          annotate(
            "text",
            x = median(df_for_bubble[[x]], na.rm = T),
            y = Inf,
            label = TeX(paste0(
              "Median relative change: ", round(median((df_for_bubble[[y]] - df_for_bubble[[x]]) / df_for_bubble[[x]],
                                                       na.rm = T
              ), 3) * 100, '%'
            )),
            size = 4,
            color = "#000000",
            vjust = 1
          )
        }
      } +
      scale_size(range = c(.1, 5), name = "District elementary enrollment") +
      geom_abline(intercept = 0, linetype = "dashed") +
      theme_ipsum() +
      theme(legend.position = "right") +
      labs(x = x_label,
           y = y_label,
           title = curr_title) +
      guides(fill = guide_legend(byrow = TRUE))
    # theme(legend.position = "none")
    return (bubble_chart)
  }

get_sensitivity_plots <-
  function(df,
           opt_cats,
           outcome,
           travel_time_constraints,
           contiguity_constraints) {
    race_cats <- c('black', 'asian', 'native', 'hisp', 'white')
    # These are the values we want to track and eventually visualize
    travel_constraint <- c()
    contiguity <- c()
    white_norm_exp <- c()
    percent_rezoned <- c()
    travel_diff <- c()
    
    for (t in travel_time_constraints) {
      for (cont in contiguity_constraints) {
        cohesion <- 0
        if (cont) {
          cohesion <- 0.5
        }
        df_curr_best <-
          get_best_configs(
            df,
            opt_cats,
            outcome,
            travel_time_condition = t,
            contiguity_condition = cont,
            cohesion_condition = cohesion
          )
        travel_constraint <- c(travel_constraint, t)
        contiguity <- c(contiguity, if(cont == T) 'Contiguity required' else 'Contiguity not required')
        white_norm_exp <-
          c(
            white_norm_exp,
            median(
              df_curr_best$white_total_normalized_exposure_change,
              na.rm = T
            )
          )
        
        df_curr_best$all_percent_rezoned <- 0
        df_curr_best$all_travel_time_diffs <- 0
        df_curr_best$num_total_students_across_race <- 0
        for (cat in race_cats) {
          df_curr_best$num_total_students_across_race <-
            df_curr_best$num_total_students_across_race + df_curr_best[[paste(cat, 'num_total', sep =
                                                                                "_")]]
          df_curr_best$all_percent_rezoned <-
            df_curr_best$all_percent_rezoned + df_curr_best[[paste(cat, 'percent_rezoned', sep =
                                                                     "_")]] * df_curr_best[[paste(cat, 'num_total', sep = "_")]]
          df_curr_best$all_travel_time_diffs <-
            df_curr_best$all_travel_time_diffs + df_curr_best[[paste(cat, 'total_travel_time_for_rezoned_diff', sep =
                                                                       "_")]]
        }
        df_curr_best$all_percent_rezoned <-
          df_curr_best$all_percent_rezoned / df_curr_best$num_total_students_across_race
        percent_rezoned <-
          c(percent_rezoned,
            median(df_curr_best$all_percent_rezoned, na.rm = T))
        df_curr_best$all_travel_time_diffs <-
          df_curr_best$all_travel_time_diffs / df_curr_best$num_total_students_across_race / 60
        travel_diff <-
          c(travel_diff,
            median(df_curr_best$all_travel_time_diffs, na.rm = T))
      }
    }
    
    df_for_plots <-
      data.frame(
        travel_constraint = travel_constraint,
        contiguity = contiguity,
        white_norm_exp = white_norm_exp,
        percent_rezoned = percent_rezoned,
        travel_diff = travel_diff
      )
    
    get_double_line_chart <- function(df_for_plots, outcome, title){
      line_fills <- c("#404080", "#69b3a2")
      df_for_plots$travel_constraint <- 100 * df_for_plots$travel_constraint
      df_for_plots$curr_outcome <- df_for_plots[[outcome]]
      if (!grepl('travel', outcome)){
        df_for_plots$curr_outcome <- 100 * df_for_plots$curr_outcome
      }
      curr_plot <- df_for_plots %>%
        ggplot(aes(
          x = travel_constraint,
          y = curr_outcome,
          linetype = contiguity
        )) +
        geom_line(alpha = 0.7) +
        labs(y = '',
             x = 'Max % travel time increase',
             title = title) +
        theme_bw() + 
        theme(legend.title = element_blank())
      return (curr_plot)
    }
    
    seg_plot <- get_double_line_chart(df_for_plots, 'white_norm_exp', 'Median % change in V')
    rezoned_plot <- get_double_line_chart(df_for_plots, 'percent_rezoned', 'Median % rezoned')
    travel_plot <- get_double_line_chart(df_for_plots, 'travel_diff', 'Median change in travel (minutes)')
    
    p <- (seg_plot | rezoned_plot | travel_plot) + 
      plot_layout(
        # heights = c(4, 4, 4),
        guides = 'collect',
      )
    # gt <- patchwork::patchworkGrob(p)
    # gridExtra::grid.arrange(gt, bottom = TeX('Max \\% travel time increase'))
    # return (gt)
    return (p)
  }

## Identifying "prototypical" districts in small city, suburban, urban areas
get_district_centroids <- function(df) {
  df_dist <-
    df %>% group_by(district_id) %>% summarize(
      district_perwht = first(district_perwht),
      district_perblk = first(district_perblk),
      district_perfrl = first(district_perfrl),
      district_perell = first(district_perell),
      district_perhsp = first(district_perhsp),
      total_enrollment_in_district = first(total_enrollment_in_district),
      num_schools_in_district = first(num_schools_in_district),
      candidate_pref = first(candidate_pref),
      urbanicity = first(urbanicity)
    )
  df_urban_full <-
    df_dist %>% filter(urbanicity == 'urban')
  df_suburban_full <-
    df_dist %>% filter(urbanicity == 'suburban')
  df_small_city_full <-
    df_dist %>% filter(urbanicity == 'small city')
  
  df_urban_nums <-
    df_urban_full[c(
      'district_perwht',
      'district_perblk',
      'district_perfrl',
      'district_perell',
      'district_perhsp',
      'total_enrollment_in_district',
      'num_schools_in_district'
    )]
  df_suburban_nums <-
    df_suburban_full[c(
      'district_perwht',
      'district_perblk',
      'district_perfrl',
      'district_perell',
      'district_perhsp',
      'total_enrollment_in_district',
      'num_schools_in_district'
    )]
  df_small_city_nums <-
    df_small_city_full[c(
      'district_perwht',
      'district_perblk',
      'district_perfrl',
      'district_perell',
      'district_perhsp',
      'total_enrollment_in_district',
      'num_schools_in_district'
    )]
  
  regional_dfs <-
    list(df_urban_nums, df_suburban_nums, df_small_city_nums)
  regional_centroids <- c()
  
  for (r_df in regional_dfs) {
    r_df_scaled <- data.frame(scale(r_df, center = T, scale = T))
    dists <- c()
    for (i in 1:nrow(r_df_scaled)) {
      dists <-
        c(dists, euclidean(as.numeric(r_df_scaled[i,]), as.numeric(sapply(
          r_df_scaled, mean
        ))))
    }
    regional_centroids <- c(regional_centroids, which.min(dists))
  }
  
  district_id_urban_centroid <-
    df_urban_full[regional_centroids[1],]$district_id
  district_id_suburban_centroid <-
    df_suburban_full[regional_centroids[2],]$district_id
  district_id_small_city_centroid <-
    df_small_city_full[regional_centroids[3],]$district_id
  
  return (
    list(
      district_id_urban_centroid = district_id_urban_centroid,
      district_id_suburban_centroid = district_id_suburban_centroid,
      district_id_small_city_centroid = district_id_small_city_centroid
    )
  )
  
}

### Case studies for prototypical districts (Figures 4-6)
compute_case_study_plots <-
  function(curr_district_id,
           district_type,
           df,
           df_best,
           data_dir_root,
           sim_root_dir) {
    # Identify
    all_cats <- c('asian', 'black', 'hisp', 'native', 'white')
    opt_cats <- data.frame(perkey = c('perwht'), catkey = c('white'))
    pers_and_cats <-
      data.frame(
        perkey = c('perasn', 'perblk', 'perhsp', 'pernam', 'perwht'),
        catkey = all_cats
      )
    seg_indices <- c()
    df_sq <-
      df %>% filter(config == 'status_quo_zoning',
                    district_id == curr_district_id)
    for (cat in opt_cats$catkey) {
      seg_indices <-
        c(seg_indices, df_sq[[paste(cat, 'total_segregation_diff', sep = "_")]])
    }
    cats <- as.character(opt_cats[which.min(seg_indices),])
    # cats <- c('perwht', 'white')
    
    df_best_curr <-
      df_best %>% filter(district_id == curr_district_id)
    curr_config <-
      (df_best_curr %>% filter(cat_best_for == cats[2]))
    
    solution_file_path <- sprintf(
      '%s/%s/%s/%s/solution_*.csv',
      sim_root_dir,
      curr_config$state,
      curr_district_id,
      curr_config$config
    )
    
    # cat(solution_file_path)
    # return ()
    soln_file <- Sys.glob(solution_file_path)
    df_rezoning <- read.csv(soln_file)
    # df_rezoning$ncessch <- str_pad(as.character(df_rezoning$ncessch), 12, pad="0")
    df_rezoning$leaid <-
      str_pad(as.character(df_rezoning$leaid), 7, pad = "0")
    df_rezoning$new_school_nces <-
      format(df_rezoning$new_school_nces, scientific = F)
    df_rezoning$new_school_nces <-
      str_pad(as.character(df_rezoning$new_school_nces), 12, pad = "0")
    df_rezoning_by_school <-
      df_rezoning %>%
      group_by(new_school_nces) %>%
      summarize(
        new_total_enrollment = sum(num_total_to_school),
        new_white_to_school = sum(num_white_to_school),
        new_black_to_school = sum(num_black_to_school),
        new_frl_to_school = sum(num_frl_to_school),
        new_ell_to_school = sum(num_ell_to_school),
        new_hispanic_to_school = sum(num_hispanic_to_school),
        new_native_to_school = sum(num_native_to_school),
        new_asian_to_school = sum(num_asian_to_school),
      )
    df_rezoning_by_school$new_perwht <-
      df_rezoning_by_school$new_white_to_school / df_rezoning_by_school$new_total_enrollment
    df_rezoning_by_school$new_perblk <-
      df_rezoning_by_school$new_black_to_school / df_rezoning_by_school$new_total_enrollment
    df_rezoning_by_school$new_perfrl <-
      df_rezoning_by_school$new_frl_to_school / df_rezoning_by_school$new_total_enrollment
    df_rezoning_by_school$new_perell <-
      df_rezoning_by_school$new_ell_to_school / df_rezoning_by_school$new_total_enrollment
    df_rezoning_by_school$new_perhsp <-
      df_rezoning_by_school$new_hispanic_to_school / df_rezoning_by_school$new_total_enrollment
    df_rezoning_by_school$new_pernam <-
      df_rezoning_by_school$new_native_to_school / df_rezoning_by_school$new_total_enrollment
    df_rezoning_by_school$new_perasn <-
      df_rezoning_by_school$new_asian_to_school / df_rezoning_by_school$new_total_enrollment
    df_rezoning_by_school <-
      df_rezoning_by_school[c(
        'new_school_nces',
        'new_total_enrollment',
        'new_perwht',
        'new_perblk',
        'new_perfrl',
        'new_perell',
        'new_perhsp',
        'new_pernam',
        'new_perasn'
      )]
    df_state_raw <- get_state_data(data_dir_root)
    df_state_raw$leaid <-
      str_pad(as.character(df_state_raw$leaid), 7, pad = "0")
    df_state_raw$ncessch <-
      str_pad(str_trim(as.character(df_state_raw$ncessch)), 12, pad = "0")
    df_schools <-
      df_state_raw %>%
      filter(leaid == curr_district_id) %>%
      group_by(ncessch) %>%
      summarize(
        orig_perwht = first(perwht),
        orig_perblk = first(perblk),
        orig_perhsp = first(perhsp),
        orig_perfrl = first(perfrl),
        orig_perell = first(perell),
        orig_pernam = first(pernam),
        orig_perasn = first(perasn),
        district_perwht = first(district_perwht),
        district_perblk = first(district_perblk),
        district_perhsp = first(district_perhsp),
        district_perfrl = first(district_perfrl),
        district_perell = first(district_perell),
        district_pernam = first(district_pernam),
        district_perasn = first(district_perasn),
        district_name = first(LEA_NAME),
      )
    df_schools$ncessch <-
      str_pad(as.character(df_schools$ncessch), 12, pad = "0")
    df_rezoning_by_school$new_school_nces <-
      str_pad(as.character(df_rezoning_by_school$new_school_nces),
              12,
              pad = "0")
    district_name <- df_schools$district_name[1]
    df_schools <-
      df_schools %>% left_join(df_rezoning_by_school, by = c('ncessch' = 'new_school_nces'))
    df_schools$diff_key <-
      df_schools[[paste('new', cats[1], sep = '_')]] - df_schools[[paste('orig', cats[1], sep =
                                                                           '_')]]
    df_schools[order(df_schools$diff_key),]
    df_schools_to_plot_1 <- data.frame(ncessch = df_schools$ncessch)
    df_schools_to_plot_2 <- data.frame(ncessch = df_schools$ncessch)
    df_schools_to_plot_1$perkey <-
      df_schools[[paste('orig', cats[1], sep = '_')]]
    df_schools_to_plot_1$time <- 'before'
    df_schools_to_plot_2$perkey <-
      df_schools[[paste('new', cats[1], sep = '_')]]
    df_schools_to_plot_2$time <- 'after'
    df_schools_to_plot <-
      rbind.data.frame(df_schools_to_plot_1, df_schools_to_plot_2)
    df_schools_to_plot$time <-
      factor(df_schools_to_plot$time, levels = c('before', 'after'))
    
    orig_zoning_png_path <- sprintf(
      '%s/%s/%s/%s/%s.png',
      sim_root_dir,
      curr_config$state,
      curr_district_id,
      curr_config$config,
      'original_zoning'
    )
    
    orig_zoning_image <-
      fig_lab(
        fig(
          orig_zoning_png_path,
          aspect.ratio = 'free',
          b_col = 'white',
          b_size = 0,
          b_margin = ggplot2::margin(0, 0, 0, 0)
        ),
        'Original attendance zones',
        pos = 'top',
        hjust = 0,
        size = 12,
        fontfamily = 'Helvetica'
      )
    
    cat_png_path <- sprintf(
      '%s/%s/%s/%s/%s.png',
      sim_root_dir,
      curr_config$state,
      curr_district_id,
      curr_config$config,
      paste('num', cats[2], sep = '_')
    )
    
    cat_image <-
      fig_lab(
        fig(
          cat_png_path,
          aspect.ratio = 'free',
          b_col = 'white',
          b_size = 0,
          b_margin = ggplot2::margin(0, 0, 0, 0)
        ),
        sprintf('%s population', get_group_label(cats[2])),
        pos = 'top',
        hjust = 0,
        size = 12,
        fontfamily = 'Helvetica'
      )
    
    rezoning_png_path <- sprintf(
      '%s/%s/%s/%s/%s.png',
      sim_root_dir,
      curr_config$state,
      curr_district_id,
      curr_config$config,
      'rezoning'
    )
    rezoning_image <-
      fig_lab(
        fig(
          rezoning_png_path,
          aspect.ratio = 'free',
          b_col = 'white',
          b_size = 0,
          b_margin = ggplot2::margin(0, 0, 0, 0)
        ),
        'Hypothetical rezoning',
        pos = 'top',
        hjust = 0,
        size = 12,
        fontfamily = 'Helvetica'
      )
    
    offset_val <- 1
    # if (district_type == 'small city') {
    #   offset_val <- 0
    # }
    
    before_after_fills <- c("#404080", "#69b3a2")
    
    schools_plot <- df_schools_to_plot %>%
      ggplot(aes(
        x = reorder(ncessch, desc(perkey), first),
        y = perkey,
        fill = time
      )) +
      geom_bar(alpha = 0.7,
               position = "dodge",
               stat = "identity") +
      scale_fill_manual(values = before_after_fills) +
      geom_hline(yintercept = df_schools[[paste('district', cats[1], sep = '_')]][1],
                 linetype = "dashed",
                 color = "black") +
      annotate(
        "text",
        x = nrow(df_schools) - offset_val,
        y = df_schools[[paste('district', cats[1], sep = '_')]][1],
        label = sprintf("District \n %% %s", cats[2]),
        size = 3
      ) +
      theme_void() +
      labs(title = sprintf(
        'Proportion %s students at each school (before and after rezoning)',
        cats[2]
      )) +
      theme(
        axis.text.y = element_text(color = "black", size = 9),
        plot.title = element_text(size = 12),
        legend.title = element_blank()
      )
    
    best_outcomes <-
      df_best_curr %>% filter(cat_best_for == cats[2])
    all_cats_for_display <- c()
    for (cat in all_cats) {
      all_cats_for_display <-
        c(all_cats_for_display,
          get_group_label(cat))
    }
    outcomes_df <-
      data.frame(cat = all_cats, cat_readable = all_cats_for_display)
    # num_exposures <- c()
    percent_rezoned <- c()
    travel_for_rezoned_diff <- c()
    seg_vals <- data.frame(cat_readable = c(),
                           time = c(),
                           seg_val = c())
    df_sq <-
      df %>% filter(district_id == curr_district_id,
                    config == 'status_quo_zoning')
    for (i in 1:length(all_cats)) {
      seg_before <-
        list(cat_readable = all_cats_for_display[i],
             time = 'before',
             seg_val = df_sq[[paste(all_cats[i], 'total_normalized_exposure', sep = '_')]])
      seg_after <-
        list(cat_readable = all_cats_for_display[i],
             time = 'after',
             seg_val = best_outcomes[[paste(all_cats[i], 'total_normalized_exposure', sep = '_')]])
      seg_vals <- rbind(seg_vals, seg_before)
      seg_vals <- rbind(seg_vals, seg_after)
      percent_rezoned <-
        c(percent_rezoned, best_outcomes[[paste(all_cats[i], 'percent_rezoned', sep =
                                                  '_')]])
      travel_to_append <-
        best_outcomes[[paste(all_cats[i], 'total_travel_time_for_rezoned_diff', sep = '_')]] / 60 / best_outcomes[[paste(all_cats[i], 'num_rezoned', sep = '_')]]
      travel_for_rezoned_diff <-
        c(travel_for_rezoned_diff, travel_to_append)
    }
    
    outcomes_df$percent_rezoned <- percent_rezoned
    outcomes_df$travel_for_rezoned_diff <- travel_for_rezoned_diff
    seg_vals$time <-
      factor(seg_vals$time, levels = c('before', 'after'))
    seg_change_plot <- seg_vals %>%
      ggplot(aes(x = cat_readable,
                 y = seg_val,
                 fill = time)) +
      geom_bar(alpha = 0.7,
               position = "dodge",
               stat = "identity") +
      scale_fill_manual(values = before_after_fills) +
      theme_void() +
      labs(title = 'Change in segregation by group') +
      theme(
        axis.text.x = element_text(color = "black", size = 9),
        axis.text.y = element_text(color = "black", size = 9),
        plot.title = element_text(size = 12),
        legend.title = element_blank()
      ) +
      theme(legend.position = 'none')
    # seg_change_plot <- wrap_labs(seg_change_plot, 'normal')
    
    percent_rezoned_plot <- outcomes_df %>%
      ggplot(aes(
        x = cat_readable,
        y = 100 * percent_rezoned,
        fill = cat_readable
      )) +
      geom_bar(alpha = 0.7,
               position = "dodge",
               stat = "identity") +
      scale_fill_manual(values = c('#d55e00', '#56b4e9', '#009e73', '#cc79a7', '#0072b2')) +
      theme_void() +
      labs(title = 'Percent rezoned') +
      theme(
        axis.text.x = element_text(color = "black", size = 9),
        axis.text.y = element_text(color = "black", size = 9),
        plot.title = element_text(size = 12)
      ) +
      theme(legend.position = 'none')
    # percent_rezoned_plot <- wrap_labs(percent_rezoned_plot, 'normal')
    
    travel_change_for_rezoned_plot <- outcomes_df %>%
      ggplot(aes(x = cat_readable,
                 y = travel_for_rezoned_diff,
                 fill = cat_readable)) +
      geom_bar(alpha = 0.7,
               position = "dodge",
               stat = "identity") +
      scale_fill_manual(values = c('#d55e00', '#56b4e9', '#009e73', '#cc79a7', '#0072b2')) +
      theme_void() +
      labs(title = 'Average change in travel time') +
      theme(
        axis.text.x = element_text(color = "black", size = 9),
        axis.text.y = element_text(color = "black", size = 9),
        plot.title = element_text(size = 12)
      ) +
      theme(legend.position = 'none')
    
    travel_change_for_rezoned_plot <-
      # wrap_labs(travel_change_for_rezoned_plot, 'normal')
      
      all_plots <-
      (
        (orig_zoning_image + cat_image + rezoning_image) / plot_spacer() / schools_plot / plot_spacer() / (
          seg_change_plot + percent_rezoned_plot + travel_change_for_rezoned_plot
        )
      ) +
      plot_layout(heights = c(8, 1, 6, 1 , 4)) +
      plot_annotation(
        title = sprintf(
          '%s case study: %s',
          district_type,
          clean_district_name(str_to_title(district_name))
        ),
        theme = theme(plot.title = element_text(size = 14)),
        tag_levels = 'a',
        tag_suffix = ')'
      ) &
      theme(plot.tag = element_text(size = 12))
    
    return (all_plots)
  }


analyze_school_demos_changes <- function(curr_df) {
  df_schools <-
    df_state_raw %>%
    filter(leaid %in% curr_df$district_id) %>%
    group_by(ncessch) %>%
    summarize(
      orig_perwht = first(perwht),
      orig_perblk = first(perblk),
      orig_perhsp = first(perhsp),
      orig_perfrl = first(perfrl),
      orig_perell = first(perell),
      district_perwht = first(district_perwht),
      district_perblk = first(district_perblk),
      district_perhsp = first(district_perhsp),
      district_perfrl = first(district_perfrl),
      district_perell = first(district_perell),
      district_name = first(LEA_NAME),
      district_id = first(leaid),
      orig_seg_index_perblk = first(seg_index_perblk),
      orig_seg_index_perwht = first(seg_index_perwht),
      orig_seg_index_perfrl = first(seg_index_perfrl),
      orig_seg_index_perell = first(seg_index_perell),
      orig_seg_index_perhsp = first(seg_index_perhsp),
      orig_total_enrollment = first(total_enrollment),
    )
  
  all_cats <- c('black', 'ell', 'frl', 'hisp', 'white')
  all_cats_for_display <- c()
  pers_and_cats <-
    data.frame(
      perkey = c('perblk', 'perell', 'perfrl', 'perhsp', 'perwht'),
      catkey = all_cats
    )
  
  all_districts <- unique(as.character(curr_df$district_id))
  black <- c()
  black_from_dist <- c()
  ell <- c()
  ell_from_dist <- c()
  frl <- c()
  frl_from_dist <- c()
  hisp <- c()
  hisp_from_dist <- c()
  white <- c()
  white_from_dist <- c()
  school_ids <- c()
  district_ids <- c()
  for (i in 1:length(all_districts)) {
    curr_district_id <- all_districts[i]
    for (j in 1:nrow(pers_and_cats)) {
      cats <- as.character(pers_and_cats[j, ])
      df_best_curr <-
        curr_df %>% filter(district_id == curr_district_id)
      curr_config <-
        (df_best_curr %>% filter(cat_best_for == cats[2]))
      
      solution_file_path <- sprintf(
        '%s/%s/%s/%s/solution_*.csv',
        sim_root_dir,
        curr_config$state,
        curr_district_id,
        curr_config$config
      )
      
      soln_file <- Sys.glob(solution_file_path)
      df_rezoning <- read.csv(soln_file)
      df_rezoning$new_school_nces <-
        format(df_rezoning$new_school_nces, scientific = F)
      df_rezoning_by_school <-
        df_rezoning %>%
        group_by(new_school_nces) %>%
        summarize(
          new_total_enrollment = sum(num_total_to_school),
          new_white_to_school = sum(num_white_to_school),
          new_black_to_school = sum(num_black_to_school),
          new_frl_to_school = sum(num_frl_to_school),
          new_ell_to_school = sum(num_ell_to_school),
          new_hispanic_to_school = sum(num_hispanic_to_school),
        )
      df_rezoning_by_school$new_perwht <-
        df_rezoning_by_school$new_white_to_school / df_rezoning_by_school$new_total_enrollment
      df_rezoning_by_school$new_perblk <-
        df_rezoning_by_school$new_black_to_school / df_rezoning_by_school$new_total_enrollment
      df_rezoning_by_school$new_perfrl <-
        df_rezoning_by_school$new_frl_to_school / df_rezoning_by_school$new_total_enrollment
      df_rezoning_by_school$new_perell <-
        df_rezoning_by_school$new_ell_to_school / df_rezoning_by_school$new_total_enrollment
      df_rezoning_by_school$new_perhsp <-
        df_rezoning_by_school$new_hispanic_to_school / df_rezoning_by_school$new_total_enrollment
      df_rezoning_by_school <-
        df_rezoning_by_school[c(
          'new_school_nces',
          'new_total_enrollment',
          'new_perwht',
          'new_perblk',
          'new_perfrl',
          'new_perell',
          'new_perhsp'
        )]
      
      df_schools_curr <-
        df_schools %>% inner_join(df_rezoning_by_school,
                                  by = c('ncessch' = 'new_school_nces'))
      diff_from_old <-
        abs(df_schools_curr[[paste('new', cats[1], sep = '_')]] - df_schools_curr[[paste('orig', cats[1], sep = '_')]])
      orig_diff_from_dist <-
        abs(df_schools_curr[[paste('orig', cats[1], sep = '_')]] - df_schools_curr[[paste('district', cats[1], sep = '_')]])
      
      assign(cats[2], c(get(cats[2]), diff_from_old))
      assign(paste(cats[2], 'from_dist', sep = '_'),
             c(get(paste(
               cats[2], 'from_dist', sep = '_'
             )), orig_diff_from_dist))
      
      # Doesn't matter which category we select here ... just doing this because we only want to store these values once
      if (cats[2] == 'black') {
        school_ids <- c(school_ids, df_schools_curr$ncessch)
        district_ids <- c(district_ids, df_schools_curr$district_id)
      }
    }
    
  }
  to_return <-
    list(
      black = black,
      black_from_dist = black_from_dist,
      ell = ell,
      ell_from_dist = ell_from_dist,
      frl = frl,
      frl_from_dist = frl_from_dist,
      hisp = hisp,
      hisp_from_dist = hisp_from_dist,
      white = white,
      white_from_dist = white_from_dist,
      school_id = school_ids,
      district_id = district_ids
    )
  return (to_return)
  
}
