# Load necessary libraries
library(lmtest)
library(ggplot2)

# Read the saved velo comparison CSV file
velo_data <- read.csv("../Xenium/BreastS1R1/velos/GATA3_FOXA1.csv", header = FALSE)

# Initialize a vector to store Granger test scores
granger_scores <- c()

# Loop through each row in the data
for (i in 1:nrow(velo_data)) {
  # Split the row into velo_base and velo
  row <- as.numeric(velo_data[i, ])
  mid <- length(row) / 2
  velo_base <- row[1:mid]
  velo <- row[(mid + 1):length(row)]
  if ((sum(velo != 0) > (length(velo) / 2)) && (sum(velo_base != 0) > (length(velo_base) / 2))){
    score <- grangertest(velo_base, velo, order=3)[2,4]
    granger_scores <- c(granger_scores, score)
  }
}

# Create a boxplot of Granger test scores
ggplot(data.frame(scores = granger_scores), aes(x = "", y = scores)) +
  geom_boxplot(fill = "lightblue", color = "darkblue") +
  labs(title = "Boxplot of Granger Test Scores", y = "P-Value", x = "") +
  theme_minimal()





######################################################################
# This part is used to find the tf-gene pairs in the dataset
######################################################################
tf_info <- MsigDB_gene_sets_v6[[11]]
all_tfs <- names(tf_info)
# Define folders
data_folder <- '../Xenium/BreastS1R1/TimeSeries/'

# Fetch the list of genes
file_list <- list.files(data_folder, pattern = "\\.csv$", full.names = FALSE)
gene_list <- unique(sort(sapply(strsplit(file_list, "_"), `[`, 1)))
filtered_tfs <- intersect(gene_list, all_tfs)

filtered_tf_info <- list()
for (tf in filtered_tfs) {
  # Get the list of genes for the current gene from tf_info
  genes <- tf_info[[tf]]
  
  # Filter the gene list to keep only genes that appear in filtered_genes
  filtered_list <- genes[genes %in% gene_list]
  
  # Store the result in the filtered_tf_info list
  if (length(filtered_list) > 0) {
    filtered_tf_info[[tf]] <- filtered_list
  }
}

pairs_df <- data.frame(TF = character(), Gene = character())
for (tf in names(filtered_tf_info)) {
  genes <- filtered_tf_info[[tf]]
  if (length(genes) > 0) {
    # Create a data frame of TF and its genes
    tf_gene_pairs <- data.frame(TF = rep(tf, length(genes)), Gene = genes)
    # Append to the main data frame
    pairs_df <- rbind(pairs_df, tf_gene_pairs)
  }
}
write.csv(pairs_df, "../Xenium/BreastS1R1/tf_gene_pairs.csv", row.names = FALSE)




######################################################
# Boxplot on all the tf-gene pairs
######################################################
# Base folder containing subfolders
base_folder <- "../Xenium/BreastS1R1/DelayPairs"

# Initialize a list to store Granger test scores by subfolder
granger_scores_by_folder <- list()

# Traverse subfolders
subfolders <- list.dirs(base_folder, full.names = TRUE, recursive = FALSE)

for (subfolder in subfolders) {
  # Construct the file path
  velo_file <- file.path(subfolder, paste0(basename(subfolder), "_velos.csv"))
  genes <- strsplit(basename(subfolder), "_")[[1]]
  if (length(genes) == 2 && genes[1] == genes[2]) next
  
  # Check if the file exists
  if (file.exists(velo_file)) {
    # Read the velo data
    velo_data <- read.csv(velo_file, header = FALSE)
    if (nrow(velo_data) < 10) next
    
    # Check if the velo_base of the first ten rows are all zero
    skip_file <- all(rowSums(velo_data[1:10, 1:(ncol(velo_data) / 2)]) == 0)
    if (skip_file) next
    
    # Initialize a vector to store scores for this subfolder
    granger_scores <- c()
    
    # Loop through each row in the data
    for (i in 1:nrow(velo_data)) {
      # Split the row into velo_base and velo
      row <- as.numeric(velo_data[i, ])
      mid <- length(row) / 2
      velo_base <- row[1:mid]
      velo <- row[(mid + 1):length(row)]
      
      # Apply filtering criteria
      if ((sum(velo != 0) > (length(velo) / 2)) && (sum(velo_base != 0) > (length(velo_base) / 2))) {
        score <- grangertest(velo_base, velo, order = 3)[2, 4]
        granger_scores <- c(granger_scores, score)
      }
    }
    
    # If there are valid scores, add them to the list
    if (length(granger_scores) > 0) {
      granger_scores_by_folder[[basename(subfolder)]] <- granger_scores
    }
  }
}

# Simulate the random scenario
set.seed(123)  # For reproducibility
random_granger_scores <- c()
for (i in 1:10000) {  # Simulate 100 random pairs
  # Generate random velo_base and velo vectors
  velo_base <- rnorm(20, mean = 0, sd = 1)  # Random normal distribution
  velo <- rnorm(20, mean = 0, sd = 1)
  
  # Compute Granger test
  score <- grangertest(velo_base, velo, order = 3)[2, 4]
  random_granger_scores <- c(random_granger_scores, score)
}
# Add the random scenario to the data list
granger_scores_by_folder[["Random"]] <- random_granger_scores

# Simulate the delayed scenario
set.seed(789)  # For reproducibility
delayed_granger_scores <- c()
for (i in 1:10000) {  # Simulate 100 delayed pairs
  # Generate velo_base with larger variance
  velo_base <- rnorm(20, mean = 0, sd = 1)  # Larger variance (sd = 1)
  # Add noise to velo_base
  noise <- rnorm(20, mean = 0, sd = 0.5)   # Noise with larger variance
  velo_base_disturbed <- velo_base + noise  # Disturbed version of velo_base
  # Create velo with a lag of 3 and noise
  velo <- 3 * velo_base_disturbed[1:17] + rnorm(17, mean = 0, sd = 0.5)  # Linear delayed relationship with noise
  velo_base <- velo_base[4:20]             # Trim to match delay (lag 3)
  # Randomly shift some values in velo and velo_base
  shift_indices <- sample(1:17, 5)  # Randomly choose indices to shift
  velo_base[shift_indices] <- velo_base[shift_indices] + rnorm(5, mean = 0, sd = 0.2)
  velo[shift_indices] <- velo[shift_indices] + rnorm(5, mean = 0, sd = 0.2)
  
  
  # Check for sufficient variance
  if (var(velo_base) > 1e-6 && var(velo) > 1e-6) {
    score <- grangertest(velo_base, velo, order = 3)[2, 4]
    delayed_granger_scores <- c(delayed_granger_scores, score)
  }
}
# Remove NaN values
delayed_granger_scores <- delayed_granger_scores[!is.nan(delayed_granger_scores)]
# Add the delayed scenario to the data list
granger_scores_by_folder[["Delayed"]] <- delayed_granger_scores


# Convert the list to a data frame for plotting
plot_data <- do.call(rbind, lapply(names(granger_scores_by_folder), function(folder) {
  data.frame(folder = folder, score = granger_scores_by_folder[[folder]])
}))

# Create a boxplot with subfolder names as labels
ggplot(plot_data, aes(x = folder, y = score)) +
  geom_boxplot(fill = "lightblue", color = "darkblue") +
  labs(title = "Boxplot of Granger Test Scores by Subfolder", y = "P-Value", x = "Subfolder") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels for better readability

