library(ggplot2)
library(readr)

# Load the CSV files
generation_data <- read_csv('../Simulation/Result_for_Plot/simu6_gen.csv')
trans_status_data <- read_csv('../Simulation/Result_for_Plot/simu6_status.csv')
pdf('../Simulation/figures/simu6_rate.pdf')
# Plot generation_data
ggplot(generation_data, aes(x = Time)) +
  geom_line(aes(y = Ground_Truth), color = '#B23C3C') +
  geom_line(aes(y = Prediction), color = '#6FA7B6') +
  labs(x = "Time", y = "Rate") +
  theme_bw() +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))
dev.off()

pdf('../Simulation/figures/simu6_status.pdf')
# Plot trans_status_data
ggplot(trans_status_data, aes(x = Time)) +
  geom_line(aes(y = Ground_Truth), color = '#B23C3C') +
  geom_line(aes(y = Prediction), color = '#6FA7B6') +
  labs(x = "Time", y = "On/Off Status") +
  theme_bw() +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))
dev.off()