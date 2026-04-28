library(dplyr)
library(FactoMineR)
library(factoextra)

df_clean <- as.data.frame(readRDS(here::here("data", "processed", "cleaned_stroke_dataset.rds")))

df_unsupervised <- df_clean %>% 
  select(-stroke) %>%
  mutate(
    # Making the binary factor levels mathematically unique so FactoMineR doesn't crash
    hypertension = as.factor(ifelse(hypertension == "1", "Yes_HT", "No_HT")),
    heart_disease = as.factor(ifelse(heart_disease == "1", "Yes_HD", "No_HD"))
  )

famd_result <- FAMD(df_unsupervised, ncp = 5, graph = FALSE)

fviz_famd_var(famd_result, "var", repel = TRUE, 
              col.var = "contrib", 
              gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"))

library(ggplot2)

# Generate the Scree Plot
fviz_eig(famd_result, 
         choice = "eigenvalue", 
         addlabels = TRUE, 
         barfill = "#00AFBB", 
         barcolor = "#00AFBB",
         linecolor = "red",
         title = "Scree Plot: Eigenvalues of Principal Dimensions") +
  
  geom_hline(yintercept = 1, linetype = "dashed", color = "black", linewidth = 1)

fviz_famd_ind(famd_result, 
              geom = "point", 
              habillage = as.factor(df_clean$stroke), 
              palette = c("grey", "red"),
              alpha.ind = 0.4)

# Extract the numerical coordinates from the FAMD result (Top 5 dimensions)
famd_coords <- famd_result$ind$coord[, 1:5]

# Calculate and visualize the optimal number of clusters using WSS
fviz_nbclust(famd_coords, kmeans, method = "wss") +
  theme_minimal() +
  labs(title = "Elbow Method for Optimal k",
       subtitle = "Evaluating Total Within Sum of Square",
       x = "Number of Clusters (k)", 
       y = "Total Within Sum of Square (WSS)") +

  geom_vline(xintercept = 3, linetype = "dashed", color = "red", linewidth = 1)

# random seed for reproducibility
set.seed(123)

# Run K-Means Clustering
# We will test with 3 clusters (k = 3). 
kmeans_result <- kmeans(famd_coords, centers = 3, nstart = 25)

# Attach the cluster assignments back to our original dataset for later analysis
df_clean$Cluster <- as.factor(kmeans_result$cluster)

# Visualize the Clusters on the FAMD coordinate plane
fviz_famd_ind(famd_result, 
              geom = "point", 
              habillage = df_clean$Cluster, 
              palette = "Set1",             
              alpha.ind = 0.4,
              addEllipses = TRUE,
              ellipse.type = "convex",
              title = "K-Means Patient Clusters (Original FAMD Axes)") +
  
  labs(x = "Dimension 1 (Aging & Demographics Axis)", 
       y = "Dimension 2 (Cardiovascular & Lifestyle Risk Axis)")

# Calcsulate clinical profiles for each cluster
cluster_summary <- df_clean %>%
  group_by(Cluster) %>%
  summarise(
    Total_Patients = n(),
    Mean_Age = round(mean(age), 0),
    Mean_Glucose = round(mean(avg_glucose_level), 1),
    # Convert stroke factor back to numeric 0/1 to calculate the percentage
    Stroke_Rate_Percent = round(mean(as.numeric(as.character(stroke))) * 100, 2) 
  )

# Prints a formatted table for the report
knitr::kable(cluster_summary, caption = "Clinical Profile of Patient Clusters")

# Visualize Stroke Rate by Cluster
ggplot(df_clean, aes(x = Cluster, fill = stroke)) +
  # position = "fill" makes all bars the same height to compare percentages
  geom_bar(position = "fill") + 
  scale_y_continuous(labels = scales::percent) +
  theme_minimal() +
  scale_fill_manual(values = c("grey", "red"), labels = c("No Stroke", "Stroke")) +
  labs(title = "Stroke Incidence Rate by Patient Cluster",
       y = "Percentage of Patients",
       fill = "Outcome")

ggplot(df_clean, aes(x = Cluster, fill = Cluster)) +
  geom_bar() +
  theme_minimal() +
  labs(title = "Cluster Sizes in Patient Population")
