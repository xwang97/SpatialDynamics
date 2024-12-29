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
