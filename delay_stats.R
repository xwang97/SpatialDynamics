# Load necessary libraries
library(lmtest)
library(ggplot2)
library(philentropy)

#######################################################################
# 1. Test delay on one gene pair
#######################################################################
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
# 2. This part is used to find the tf-gene pairs in the dataset
######################################################################
tf_info <- MsigDB_gene_sets_v6[[11]]
all_tfs <- names(tf_info)
# Define folders
data_folder <- '../Jing/0029719/24h/TimeSeries/'

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
write.csv(pairs_df, "../Jing/0029719/tf_gene_pairs.csv", row.names = FALSE)




######################################################
# 3. Boxplot of delay effect on all the tf-gene pairs
######################################################
# Base folder containing subfolders
base_folder <- "../Xenium/BreastS1R1/DelayPairs"
molecules_folder <- "../Xenium/BreastS1R1/MoleculesPerGene"
# Initialize a list to store Granger test scores by subfolder
granger_scores_by_folder <- list()
subfolders <- list.dirs(base_folder, full.names = TRUE, recursive = FALSE)

for (subfolder in subfolders) {
  # Construct the file path
  velo_file <- file.path(subfolder, paste0(basename(subfolder), "_velos.csv"))
  genes <- strsplit(basename(subfolder), "_")[[1]]
  if (length(genes) == 2 && genes[1] == genes[2]) next
  # Check if the file exists
  if (file.exists(velo_file)) {
    velo_data <- read.csv(velo_file, header = FALSE)
    if (nrow(velo_data) < 10) next
    # Check if the velo_base of the first ten rows are all zero
    skip_file <- all(rowSums(velo_data[1:10, 1:(ncol(velo_data) / 2)]) == 0) || all(rowSums(velo_data[1:10, (ncol(velo_data)/2+1):ncol(velo_data)]) == 0)
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
        score <- grangertest(velo_base, velo, order = 2)[2, 3]
        granger_scores <- c(granger_scores, score)
      }
      else{
        granger_scores <- c(granger_scores, NA)
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
for (i in 1:2000) {  # Simulate 100 random pairs
  # Generate random velo_base and velo vectors
  velo_base <- rnorm(10, mean = 0, sd = 1)  # Random normal distribution
  velo <- rnorm(10, mean = 2, sd = 1)
  
  # Compute Granger test
  score <- grangertest(velo_base, velo, order = 2)[2, 3]
  random_granger_scores <- c(random_granger_scores, score)
}
# Add the random scenario to the data list
granger_scores_by_folder[["Random"]] <- random_granger_scores

# Simulate the delayed scenario
set.seed(789)  # For reproducibility
delayed_granger_scores <- c()
for (i in 1:2000) {  # Simulate 100 delayed pairs
  # Generate velo_base with larger variance
  velo_base <- rnorm(10, mean = 0, sd = 1)  # Larger variance (sd = 1)
  # Add noise to velo_base
  noise <- rnorm(10, mean = 0, sd = 0.5)   # Noise with larger variance
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
    score <- grangertest(velo_base, velo, order = 2)[2, 3]
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
ggplot(plot_data[plot_data$folder != "Delayed", ], aes(x = folder, y = score)) +
  geom_boxplot(fill = "lightblue", color = "darkblue", outlier.shape = NA) +
  ylim(0, 6) +
  labs(title = "Boxplot of Granger Test Scores by Subfolder", y = "F-Statistic", x = "Gene Pairs") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels for better readability



###################################################################
# 4. Density plot of delay statistics
###################################################################
# Create a new column to categorize data into three groups: Delayed, Random, and Group 3 (others)
plot_data$group <- ifelse(plot_data$folder == "Delayed", "Delayed",
                          ifelse(plot_data$folder == "Random", "Random", "Observed"))
# Create a list of unique folders from group 3
group3_folders <- unique(plot_data$folder[plot_data$group == "Observed"])

# Initialize an empty list to store plots
plots_list <- list()
# Loop through each folder in group 3 and create a density plot
for (folder_name in group3_folders) {
  # Filter the data for the current group 3 folder
  group3_data <- plot_data[plot_data$folder == folder_name, ]
  # Combine "Delayed" and "Random" data with the current group 3 folder data
  combined_data <- rbind(
    plot_data[plot_data$group == "Delayed", ],  # Delayed group data
    plot_data[plot_data$group == "Random", ],   # Random group data
    group3_data                                  # Current group 3 folder data
  )
  # Create a density plot for the current folder, comparing all three groups
  p <- ggplot(combined_data, aes(x = score, color = group)) +
    geom_density() +
    xlim(0, 20) +
    labs(title = paste("Density Plot for", folder_name), y = "Density", x = "F-Statistic") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +  # Rotate x-axis labels for better readability
    scale_color_manual(values = c("Delayed" = "blue", "Random" = "red", "Observed" = "green"))
  # Add the plot to the list
  plots_list[[folder_name]] <- p
}
# Combine all the plots into one figure with multiple subplots
library(gridExtra)
grid.arrange(grobs = plots_list, ncol = 2)  # You can adjust ncol for the number of columns in the grid




###################################################################
# 5. Visualize delay significance on spatial slice
###################################################################
library(RColorBrewer)
gene_pair <- "FOXA1_KRT7"
file_name <- paste0(strsplit(gene_pair, "_")[[1]][[2]], "_locs.csv")
locs_file <- file.path(base_folder, gene_pair, file_name)
scores <- granger_scores_by_folder[[gene_pair]]
locations <- read.csv(locs_file, header = FALSE)[-1,]
delay_spatial <- data.frame(
  score = scores, x = locations[[1]], y = locations[[2]]
)

df<-data.frame(a=delay_spatial$x, b=delay_spatial$y, velocity=delay_spatial$score)
df <- df[!is.na(df$velocity), ]
# df <- df[-which.max(df$velocity), ]

ggplot(df, aes(a, -b))+
  geom_point(aes(color=velocity))+
  scale_colour_gradientn("velocity",
                         colours=rev(brewer.pal(8,"Spectral")),
                         breaks=seq(0, 50, length.out=8))+
  labs(title = gene_pair,
       color = "Mean Antigen Level",
       size = "T Cell Level")+
  xlab("")+
  ylab("")+
  coord_fixed()


# signal amplify

k=100 # the number of neighbors we want to use

alpha=0.85 # the threshold of weight

# compute Gaussian kernel weight matrix

sigma=10

gua_kernel_weight<-function(coords,alpha){
  
  dis<-distance(coords,method = 'euclidean')
  
  kg<-exp(-dis/(2*sigma^2))
  
  # normailze
  
  # kg<-apply(kg,1,function(x){
  
  # x=x/sum(x)
  
  # return(x)
  
  # })
  
  kg2<-apply(kg,1,function(x){
    
    x[x<alpha]=0
    
    return(x)
    
  })
  
  # dd<-t(apply(dis,1,order))
  
  # knestest<-dd[,2:k+1]
  
  return(kg2)
  
}

weight<-gua_kernel_weight(df[, c("a", "b")],alpha)

velo_amp<-weight%*%array(df$velocity,dim=c(length(df$velocity),1))
df$amp = velo_amp



threshold <- quantile(df$velocity, 0.95)

# Assign colors based on the threshold
df$color <- ifelse(df$velocity > threshold, "red", "lightgrey")

# Plot using ggplot2
ggplot(df, aes(x = a, y = b, color = color)) +
  geom_point(size = 0.3) +  # Scatter plot with points
  scale_color_identity() +  # Use predefined colors
  theme_minimal() +
  labs(title = "Spatial Distribution of Points",
       x = "X Location", y = "Y Location")

