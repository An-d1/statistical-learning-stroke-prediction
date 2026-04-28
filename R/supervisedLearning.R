knitr::opts_chunk$set(echo = TRUE, warning = FALSE, message = FALSE)

library(rsample)
library(recipes)
library(themis)
library(parsnip)
library(workflows)
library(tune)
library(doParallel)
library(ggplot2)
library(yardstick)

df_clean <- as.data.frame(readRDS(here::here("data", "processed", "cleaned_stroke_dataset.rds")))

df_clean$stroke <- factor(df_clean$stroke,
                          levels = c("1", "0"),
                          labels = c("Stroke", "No Stroke"))


set.seed(4092) 

# Stratified split that ensures the tiny percentage of stroke cases is distributed evenly
init_split <- initial_split(df_clean, prop = 0.80, strata = stroke)

train_df <- training(init_split)
test_df  <- testing(init_split)

levels(train_df$stroke)

# Printing to verify the imbalance in the training partition
cat("Original distribution in training data:\n")
print(table(train_df$stroke))

smote_blueprint <- recipe(stroke ~ ., data = train_df) %>%
  # will be oversampled until it is 25% the size of the majority class.
  step_smotenc(stroke, over_ratio = 0.25) %>%
  step_dummy(all_nominal_predictors())

# Extracting the finalized, balanced dataset to use for model fitting
train_resampled <- smote_blueprint %>% 
  prep() %>% 
  juice()

cat("\nNew distribution after SMOTENC:\n")
print(table(train_resampled$stroke))

log_spec <- logistic_reg() %>% 
  set_engine("glm") %>% 
  set_mode("classification")

rf_tune_spec <- rand_forest(
  mtry = tune(),      
  min_n = tune(),     
  trees = 500         
) %>% 
  set_engine("ranger", importance = "impurity") %>% 
  set_mode("classification")

xgb_tune_spec <- boost_tree(
  tree_depth = tune(), 
  learn_rate = tune(), 
  min_n = tune(),
  trees = 500
) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification")

set.seed(2026)

cv_folds <- vfold_cv(train_df, v = 5, strata = stroke)
cat("Successfully created 5 cross-validation folds.\n")

log_workflow <- workflow() %>%
  add_recipe(smote_blueprint) %>%
  add_model(log_spec)

rf_workflow <- workflow() %>%
  add_recipe(smote_blueprint) %>%
  add_model(rf_tune_spec)

xgb_workflow <- workflow() %>%
  add_recipe(smote_blueprint) %>%
  add_model(xgb_tune_spec)

# Leaving one core free 
registerDoParallel(cores = parallel::detectCores() - 1) 

# Logistic regression doesn't need tuning, we just fit it directly to the folds
cat("Fitting Logistic Regression...\n")
log_results <- fit_resamples(
log_workflow,
resamples = cv_folds,
control = control_resamples(save_pred = TRUE)
)

# Tune Random Forest
cat("Tuning Random Forest...\n")
set.seed(345)
rf_tune_results <- tune_grid(
rf_workflow,
resamples = cv_folds,
grid = 10 
)

# Tune XGBoost
cat("Tuning XGBoost...\n")
set.seed(456)
xgb_tune_results <- tune_grid(
xgb_workflow,
resamples = cv_folds,
grid = 10 
)

# Shut down the parallel processing when done
stopImplicitCluster()

# Logistic Regression
set.seed(123)
log_compare_results <- fit_resamples(
  log_workflow,
  resamples = cv_folds,
  metrics = metric_set(accuracy, sens, spec)
)

# Random Forest
best_rf_params <- select_best(rf_tune_results, metric = "roc_auc")
final_rf_workflow <- finalize_workflow(rf_workflow, best_rf_params)

set.seed(123)
rf_compare_results <- fit_resamples(
  final_rf_workflow,
  resamples = cv_folds,
  metrics = metric_set(accuracy, sens, spec)
)

# XGBoost
best_xgb_params <- select_best(xgb_tune_results, metric = "roc_auc")
final_xgb_workflow <- finalize_workflow(xgb_workflow, best_xgb_params)

set.seed(123)
xgb_compare_results <- fit_resamples(
  final_xgb_workflow,
  resamples = cv_folds,
  metrics = metric_set(accuracy, sens, spec)
)

# Combine all three
compare_df <- bind_rows(
  collect_metrics(log_compare_results) %>% mutate(Model = "Logistic Regression"),
  collect_metrics(rf_compare_results) %>% mutate(Model = "Random Forest"),
  collect_metrics(xgb_compare_results) %>% mutate(Model = "XGBoost")
) %>%
  filter(.metric %in% c("accuracy", "sens", "spec")) %>%
  mutate(
    Metric = recode(.metric,
      accuracy = "Overall Accuracy",
      sens = "Sensitivity\n(Stroke)",
      spec = "Specificity\n(No Stroke)"
    )
  )

# Plot
ggplot(compare_df, aes(x = Metric, y = mean, fill = Model)) +
  geom_col(position = "dodge", color = "black", alpha = 0.85) +
  geom_text(
    aes(label = round(mean, 3)),
    position = position_dodge(width = 0.9),
    vjust = -0.5,
    size = 4
  ) +
  scale_fill_manual(values = c(
    "Logistic Regression" = "#e74c3c",
    "Random Forest" = "#2c3e50",
    "XGBoost" = "#27ae60"
  )) +
  theme_minimal() +
  labs(
    title = "Model Comparison",
    subtitle = "Cross-validated performance on the resampled training data, tuned for ROC-AUC",
    x = NULL,
    y = "Cross-Validated Score",
    fill = "Model"
  ) +
  coord_cartesian(ylim = c(0, 1.1))

# Leaving one core free 
registerDoParallel(cores = parallel::detectCores() - 1) 

stroke_metrics <- metric_set(sens, spec, accuracy, roc_auc)

# Logistic regression doesn't need tuning, we just fit it directly to the folds
cat("Fitting Logistic Regression...\n")
log_results <- fit_resamples(
log_workflow,
resamples = cv_folds,
control = control_resamples(save_pred = TRUE)
)

# Tune Random Forest
cat("Tuning Random Forest...\n")
set.seed(345)
rf_tune_sens_results <- tune_grid(
rf_workflow,
resamples = cv_folds,
grid = 10,
metrics = stroke_metrics # Tuning for senstivity and the other metrics
)

# Tune XGBoost
cat("Tuning XGBoost...\n")
set.seed(456)
xgb_tune_sens_results <- tune_grid(
xgb_workflow,
resamples = cv_folds,
grid = 10,
metrics = stroke_metrics # Tuning for senstivity and the other metrics
)

# Shut down the parallel processing when done
stopImplicitCluster()

# Logistic Regression
set.seed(123)
log_compare_results <- fit_resamples(
  log_workflow,
  resamples = cv_folds,
  metrics = stroke_metrics
)

# Random Forest - Extract the hyperparameter combination that yields the highest sensitivity
best_rf_sens <- select_best(rf_tune_sens_results, metric = "sens")
final_rf_workflow <- finalize_workflow(rf_workflow, best_rf_sens)

set.seed(123)
rf_compare_results <- fit_resamples(
  final_rf_workflow,
  resamples = cv_folds,
  metrics = stroke_metrics
)

# XGBoost - Extract the hyperparameter combination that yields the highest sensitivity
best_xgb_sens <- select_best(xgb_tune_sens_results, metric = "sens")
final_xgb_workflow <- finalize_workflow(xgb_workflow, best_xgb_sens)

set.seed(123)
xgb_compare_results <- fit_resamples(
  final_xgb_workflow,
  resamples = cv_folds,
  metrics = stroke_metrics
)

# Combine all three
compare_df <- bind_rows(
  collect_metrics(log_compare_results) %>% mutate(Model = "Logistic Regression"),
  collect_metrics(rf_compare_results) %>% mutate(Model = "Random Forest"),
  collect_metrics(xgb_compare_results) %>% mutate(Model = "XGBoost")
) %>%
  filter(.metric %in% c("accuracy", "sens", "spec")) %>%
  mutate(
    Metric = recode(.metric,
      accuracy = "Overall Accuracy",
      sens = "Sensitivity\n(Stroke)",
      spec = "Specificity\n(No Stroke)"
    )
  )

# Plot
ggplot(compare_df, aes(x = Metric, y = mean, fill = Model)) +
  geom_col(position = "dodge", color = "black", alpha = 0.85) +
  geom_text(
    aes(label = round(mean, 3)),
    position = position_dodge(width = 0.9),
    vjust = -0.5,
    size = 4
  ) +
  scale_fill_manual(values = c(
    "Logistic Regression" = "#e74c3c",
    "Random Forest" = "#2c3e50",
    "XGBoost" = "#27ae60"
  )) +
  theme_minimal() +
  labs(
    title = "Model Comparison",
    subtitle = "Cross-validated performance on the resampled training data, tuned for sensitivity",
    x = NULL,
    y = "Cross-Validated Score",
    fill = "Model"
  ) +
  coord_cartesian(ylim = c(0, 1.1))

# Creating a blueprint with no resampling simmilarly to the resampled one
unsampled_blueprint <- recipe(stroke ~ ., data = train_df) %>%
  step_dummy(all_nominal_predictors())

# Building a workflow using an unsampled Logistic Regression specification
unsampled_log_workflow <- workflow() %>%
  add_recipe(unsampled_blueprint) %>%
  add_model(log_spec)

# Fitting it to the exact same cross-validation folds to ensure a fair comparison
set.seed(999)
unsampled_log_results <- fit_resamples(
  unsampled_log_workflow,
  resamples = cv_folds,
  # collecting sensitivity and specifity
  metrics = metric_set(roc_auc, accuracy, sens, spec)
)

library(dplyr)
library(ggplot2)

# Combine sampled and unsampled Logistic Regression results
log_resampling_compare_df <- bind_rows(
  collect_metrics(log_compare_results) %>% mutate(Model = "Sampled (SMOTENC)"),
  collect_metrics(unsampled_log_results) %>% mutate(Model = "Unsampled (Raw Data)")
) %>%
  filter(.metric %in% c("accuracy", "sens", "spec")) %>%
  mutate(
    Metric = recode(.metric,
      accuracy = "Overall Accuracy",
      sens = "Sensitivity\n(Stroke)",
      spec = "Specificity\n(No Stroke)"
    )
  )

# Plot
ggplot(log_resampling_compare_df, aes(x = Metric, y = mean, fill = Model)) +
  geom_col(position = "dodge", color = "black", alpha = 0.85) +
  geom_text(
    aes(label = round(mean, 3)),
    position = position_dodge(width = 0.9),
    vjust = -0.5,
    size = 4
  ) +
  scale_fill_manual(values = c(
    "Sampled (SMOTENC)" = "#2c3e50",
    "Unsampled (Raw Data)" = "#e74c3c"
  )) +
  theme_minimal() +
  labs(
    title = "Logistic Regression: Unsampled vs. SMOTENC Resampling",
    subtitle = "Cross-validated performance before and after class balancing",
    x = NULL,
    y = "Cross-Validated Score",
    fill = "Training Data"
  ) +
  coord_cartesian(ylim = c(0, 1.1))


# Fit logistic regression once on the training data
final_log_fit <- fit(log_workflow, data = train_df)

# Get predicted probabilities on the test set
log_probs <- predict(final_log_fit, test_df, type = "prob") %>%
  bind_cols(test_df %>% select(stroke))

# Thresholds to compare
thresholds_to_compare <- c(0.30, 0.20, 0.15)

# Build a table with accuracy, sensitivity, and specificity for each threshold
threshold_compare_df <- bind_rows(lapply(thresholds_to_compare, function(th) {
  
  preds <- log_probs %>%
    mutate(
      .pred_class = if_else(.pred_Stroke >= th, "Stroke", "No Stroke"),
      .pred_class = factor(.pred_class, levels = levels(stroke))
    )
  
  acc_df <- accuracy(preds, truth = stroke, estimate = .pred_class) %>%
    transmute(
      Threshold = paste0("t = ", th),
      Metric = "Overall Accuracy",
      Score = .estimate
    )
  
  sens_df <- sens(preds, truth = stroke, estimate = .pred_class) %>%
    transmute(
      Threshold = paste0("t = ", th),
      Metric = "Sensitivity\n(Stroke)",
      Score = .estimate
    )
  
  spec_df <- spec(preds, truth = stroke, estimate = .pred_class) %>%
    transmute(
      Threshold = paste0("t = ", th),
      Metric = "Specificity\n(No Stroke)",
      Score = .estimate
    )
  
  bind_rows(acc_df, sens_df, spec_df)
}))

# Plot side by side
ggplot(threshold_compare_df, aes(x = Metric, y = Score, fill = Threshold)) +
  geom_col(position = "dodge", color = "black", alpha = 0.85) +
  geom_text(
    aes(label = round(Score, 3)),
    position = position_dodge(width = 0.9),
    vjust = -0.5,
    size = 4
  ) +
  scale_fill_manual(values = c(
    "t = 0.3" = "#2c3e50",
    "t = 0.2" = "#e74c3c",
    "t = 0.15" = "#27ae60"
  )) +
  theme_minimal() +
  labs(
    title = "Logistic Regression Performance at Different Thresholds",
    subtitle = "Comparing overall accuracy, sensitivity, and specificity",
    x = NULL,
    y = "Performance Score",
    fill = "Threshold"
  ) +
  coord_cartesian(ylim = c(0, 1.1))

# Generate predicted probabilities on the completely unseen test set
test_probs <- predict(final_log_fit, test_df, type = "prob") %>%
  bind_cols(test_df %>% select(stroke))

# Apply chosen threshold (using 0.15 based on the previous analysis)
chosen_threshold <- 0.15

final_test_preds <- test_probs %>%
  mutate(
    .pred_class = if_else(.pred_Stroke >= chosen_threshold, "Stroke", "No Stroke"),
    .pred_class = factor(.pred_class, levels = levels(stroke))
  )

# Calculate final performance metrics
final_metrics <- bind_rows(
  accuracy(final_test_preds, truth = stroke, estimate = .pred_class),
  sens(final_test_preds, truth = stroke, estimate = .pred_class),
  spec(final_test_preds, truth = stroke, estimate = .pred_class)
) %>%
  select(.metric, .estimate) %>%
  mutate(Metric = recode(.metric, 
                         accuracy = "Overall Accuracy", 
                         sens = "Sensitivity", 
                         spec = "Specificity"))

cat("Final Model Performance on Unseen Test Data \n")
print(final_metrics)

# confusion matrix to visualize the exact classification counts
conf_mat(final_test_preds, truth = stroke, estimate = .pred_class) %>%
  autoplot(type = "heatmap") +
  theme_minimal() +
  scale_fill_gradient(low = "#fdf6f5", high = "#e74c3c") +
  labs(
    title = "Final Confusion Matrix (Test Set)", 
    subtitle = paste("Logistic Regression at Threshold =", chosen_threshold)
  )
