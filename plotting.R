library(ggplot2)
library(readr)

##########################################################
# 1. Plotting for simulated data
##########################################################
# 1.1 Visualize the predicted transcription rate and status with ground truth.
# Load the CSV files
generation_data <- read_csv('../Simulation/Result_for_Plot/simu6_gen.csv')
trans_status_data <- read_csv('../Simulation/Result_for_Plot/simu6_status.csv')
pdf('../Simulation/figures/simu6_rate.pdf')
# Plot generation_data
ggplot(generation_data, aes(x = Time)) +
  geom_line(aes(y = Ground_Truth), color = '#B23C3C', linewidth=1.2) +
  geom_line(aes(y = sample_1), color = '#6FA7B6', linewidth=1.2) +
  labs(x = "Time", y = "Rate") +
  theme_bw() +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))
dev.off()

pdf('../Simulation/figures/simu6_status.pdf')
# Plot trans_status_data
ggplot(trans_status_data, aes(x = Time)) +
  geom_line(aes(y = Ground_Truth), color = '#B23C3C', linewidth=1.2) +
  geom_line(aes(y = sample_1), color = '#6FA7B6', linewidth=1.2) +
  labs(x = "Time", y = "On/Off Status") +
  theme_bw() +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))
dev.off()

# 1.2 Barplot for evaluation (MSE of rates)
file_list <- list.files(path = "../Simulation/Result_for_Plot", pattern = "gen\\.csv$", full.names = TRUE)
data_list <- lapply(file_list, read.csv)
calculate_mse <- function(df) {
  gt <- df$Ground_Truth
  sample_columns <- grep("^sample_", names(df), value = TRUE)
  mse_values <- sapply(sample_columns, function(col) {
    mean((df[[col]] - gt)^2)
  })
  return(mse_values)
}
# Apply the MSE calculation to each data frame
mse_list <- lapply(data_list, calculate_mse)

# Calculate mean and variance for each file
mse_stats <- lapply(mse_list, function(mse) {
  list(mean = mean(mse), variance = var(mse))
})
# Create a data frame for plotting
plot_data <- data.frame(
  File = file_list,
  Mean_MSE = sapply(mse_stats, function(x) x$mean),
  Variance = sapply(mse_stats, function(x) x$variance)
)
plot_data$Scenario <- paste("Pattern", 1:nrow(plot_data))
pdf('../Simulation/figures/rate_mse.pdf')
ggplot(plot_data, aes(x = Scenario, y = Mean_MSE)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_errorbar(aes(ymin = Mean_MSE - sqrt(Variance), ymax = Mean_MSE + sqrt(Variance)), 
                width = 0.2) +
  labs(x = "Scenario", y = "MSE", title = "") +
  theme_bw() +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
dev.off()


# 1.3 Barplot for evaluation (accuracy of status)
file_list_status <- list.files(path = '../Simulation/Result_for_Plot', pattern = "status\\.csv$", full.names = TRUE)
data_list_status <- lapply(file_list_status, read.csv)
calculate_accuracy <- function(df) {
  gt <- df$Ground_Truth
  sample_columns <- grep("^sample_", names(df), value = TRUE)
  accuracy_values <- sapply(sample_columns, function(col) {
    mean((df[[col]] == gt))
  })
  return(accuracy_values)
}
# Apply the accuracy calculation to each data frame
accuracy_list <- lapply(data_list_status, calculate_accuracy)
# Calculate mean and variance for each file
accuracy_stats <- lapply(accuracy_list, function(acc) {
  list(mean = mean(acc), variance = var(acc))
})
# Create a data frame for plotting
plot_data_status <- data.frame(
  File = file_list_status,
  Mean_Accuracy = sapply(accuracy_stats, function(x) x$mean),
  Variance = sapply(accuracy_stats, function(x) x$variance)
)
# Add custom labels for x-axis (e.g., "Pattern 1" to "Pattern 5")
plot_data_status$Scenario <- paste("Pattern", 1:nrow(plot_data_status))
pdf('../Simulation/figures/status_acc.pdf')
ggplot(plot_data_status, aes(x = Scenario, y = Mean_Accuracy)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_errorbar(aes(ymin = Mean_Accuracy - sqrt(Variance), ymax = Mean_Accuracy + sqrt(Variance)), 
                width = 0.2) +
  labs(x = "Scenario", y = "Accuracy", title = "") +
  theme_minimal() +
  theme_bw() +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
dev.off()
