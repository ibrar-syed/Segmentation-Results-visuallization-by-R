# Main Analysis Script for cGAN_Metrics....
library(tidyverse)
library(readxl)
library(patchwork)
source("scripts/helper_functions.R")

# Paths
project_root <- "./my_project"
data_path <- file.path(project_root, "data/raw/my_data.xlsx")
output_path <- file.path(project_root, "outputs")

# Ensure output folder exists
if (!dir.exists(output_path)) dir.create(output_path, recursive = TRUE)

# ---- Plot Functions ----

# Dice line plot
create_dice_plot <- function(metrics_data) {
  dice_data <- metrics_data %>%
    select(Epoch, Train_Dice_Score, Val_Dice_Score) %>%
    pivot_longer(cols=-Epoch, names_to="Metric", values_to="Score")
  
  ggplot(dice_data, aes(x=Epoch, y=Score, color=Metric)) +
    geom_line(linewidth=1.2) + geom_point(size=2, alpha=0.8) +
    scale_color_manual(values=c("Train_Dice_Score"="#E69F00", "Val_Dice_Score"="#0072B2")) +
    labs(title="Dice Score Across Epochs", x="Epoch", y="Dice Score") +
    theme_minimal(base_size=14)
}

# Loss line plot
create_loss_plot <- function(metrics_data) {
  loss_data <- metrics_data %>%
    select(Epoch, Gen_Total_Loss, Gen_GAN_Loss, Gen_Focal_Tversky_Loss, Disc_Loss) %>%
    pivot_longer(cols=-Epoch, names_to="Metric", values_to="Value")
  
  ggplot(loss_data, aes(x=Epoch, y=Value, color=Metric)) +
    geom_line(linewidth=1) + geom_point(size=1.5, alpha=0.6) +
    facet_wrap(~Metric, scales="free_y", ncol=2) +
    labs(title="Generator & Discriminator Loss", x="Epoch", y="Loss Value") +
    theme_minimal(base_size=14) + theme(legend.position="none")
}

# Dice bar chart final epoch
create_dice_bar <- function(metrics_data) {
  final <- metrics_data %>% filter(Epoch==max(Epoch)) %>%
    select(Train_Dice_Score, Val_Dice_Score) %>%
    pivot_longer(everything(), names_to="Metric", values_to="Score")
  
  ggplot(final, aes(x=Metric, y=Score, fill=Metric)) +
    geom_col(width=0.6) + geom_text(aes(label=round(Score,3)), vjust=-0.3, size=4) +
    scale_fill_manual(values=c("Train_Dice_Score"="#E69F00", "Val_Dice_Score"="#0072B2")) +
    labs(title=paste("Dice Score Final Epoch (Epoch", max(metrics_data$Epoch), ")"),
         y="Dice Score", x="") + theme_minimal(base_size=14) +
    theme(legend.position="none")
}

# Loss stacked bar final epoch
create_loss_bar <- function(metrics_data) {
  final <- metrics_data %>% filter(Epoch==max(Epoch)) %>%
    select(Gen_Total_Loss, Gen_GAN_Loss, Gen_Focal_Tversky_Loss, Disc_Loss) %>%
    pivot_longer(everything(), names_to="Metric", values_to="Value")
  
  ggplot(final, aes(x="Loss Components", y=Value, fill=Metric)) +
    geom_col(width=0.6) +
    geom_text(aes(label=round(Value,3)), position=position_stack(vjust=0.5), size=4) +
    scale_fill_brewer(palette="Set2") +
    labs(title=paste("Loss Components Final Epoch (Epoch", max(metrics_data$Epoch), ")"),
         y="Loss Value", x="") + theme_minimal(base_size=14) +
    theme(legend.position="top")
}

# Dice boxplot
create_dice_boxplot <- function(metrics_data) {
  dice_data <- metrics_data %>%
    select(Epoch, Train_Dice_Score, Val_Dice_Score) %>%
    pivot_longer(cols=-Epoch, names_to="Metric", values_to="Score")
  
  ggplot(dice_data, aes(x=Metric, y=Score, fill=Metric)) +
    geom_boxplot(alpha=0.6) +
    scale_fill_manual(values=c("Train_Dice_Score"="#E69F00", "Val_Dice_Score"="#0072B2")) +
    labs(title="Dice Score Distribution Across Epochs", y="Dice Score", x="") +
    theme_minimal(base_size=14) + theme(legend.position="none")
}

# Loss boxplot
create_loss_boxplot <- function(metrics_data) {
  loss_data <- metrics_data %>%
    select(Epoch, Gen_Total_Loss, Gen_GAN_Loss, Gen_Focal_Tversky_Loss, Disc_Loss) %>%
    pivot_longer(cols=-Epoch, names_to="Metric", values_to="Value")
  
  ggplot(loss_data, aes(x=Metric, y=Value, fill=Metric)) +
    geom_boxplot(alpha=0.6) +
    scale_fill_brewer(palette="Set2") +
    labs(title="Loss Distribution Across Epochs", y="Loss Value", x="") +
    theme_minimal(base_size=14) + theme(legend.position="none")
}

# Combined multi-panel figure
make_full_figure <- function(metrics_data) {
  p1 <- create_dice_plot(metrics_data)
  p2 <- create_loss_plot(metrics_data)
  p3 <- create_dice_bar(metrics_data)
  p4 <- create_loss_bar(metrics_data)
  p5 <- create_dice_boxplot(metrics_data)
  p6 <- create_loss_boxplot(metrics_data)
  
  (p1 / p2) | (p3 / p4) | (p5 / p6)
}

# ---- Main Execution ----
main <- function() {
  metrics_data <- load_data(data_path)
  
  dice_plot <- create_dice_plot(metrics_data)
  loss_plot <- create_loss_plot(metrics_data)
  dice_bar <- create_dice_bar(metrics_data)
  loss_bar <- create_loss_bar(metrics_data)
  dice_box <- create_dice_boxplot(metrics_data)
  loss_box <- create_loss_boxplot(metrics_data)
  full_figure <- make_full_figure(metrics_data)
  
  # Save individual plots
  ggsave(file.path(output_path,"dice_plot.png"), dice_plot, width=8, height=6)
  ggsave(file.path(output_path,"loss_plot.png"), loss_plot, width=10, height=8)
  ggsave(file.path(output_path,"dice_bar.png"), dice_bar, width=6, height=6)
  ggsave(file.path(output_path,"loss_bar.png"), loss_bar, width=8, height=6)
  ggsave(file.path(output_path,"dice_boxplot.png"), dice_box, width=6, height=6)
  ggsave(file.path(output_path,"loss_boxplot.png"), loss_box, width=8, height=6)
  ggsave(file.path(output_path,"Figure_full.png"), full_figure, width=18, height=10)
  
  # Save processed data
  write_csv(metrics_data, file.path(output_path,"processed_metrics.csv"))
  
  return(list(
    data=metrics_data,
    dice_plot=dice_plot,
    loss_plot=loss_plot,
    dice_bar=dice_bar,
    loss_bar=loss_bar,
    dice_box=dice_box,
    loss_box=loss_box,
    full_figure=full_figure
  ))
}

# Execute
results <- main()
