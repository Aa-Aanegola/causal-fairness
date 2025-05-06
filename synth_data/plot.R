all_plots <- function(path, plot_path, img_dim=32, latent_dim=16, valid_latents = NA) {
    library(faircause)
    library(data.table)
    library(ggplot2)

    if (is.na(valid_latents)) {
        valid_latents <- 0:(latent_dim-1)
    }

    df <- read.csv(path)
    df <- data.table(df)
    W <- paste0("w_", valid_latents)
    W_prime <- paste0("w_prime_", valid_latents)
    W_empty <- list()
    W_img <- paste0("img_", 0:1023)
    X <- "X"
    Y <- "Y"
    Z <- "Z"
    fair_decomp <- fairness_cookbook(df, X=X, W=W, Z=Z, Y=Y, x0=0, x1=1)
    fair_decomp_prime <- fairness_cookbook(df, X=X, W=W_prime, Z=Z, Y=Y, x0=0, x1=1)
    fair_decomp_no_W <- fairness_cookbook(df, X=X, W=W_empty, Z=Z, Y=Y, x0=0, x1=1)
    fair_decomp_img <- fairness_cookbook(df, X=X, W=W_img, Z=Z, Y=Y, x0=0, x1=1)

    decompositions <- c("xspec", "both", "general")

    p <- autoplot(fair_decomp)
    ggsave(paste0(plot_path, "/tv_true_w.png"), plot=p)
    p <- autoplot(fair_decomp_prime)
    ggsave(paste0(plot_path, "/tv_prime_w.png"), plot=p)
    p <- autoplot(fair_decomp_no_W)
    ggsave(paste0(plot_path, "/tv_no_w.png"), plot=p)
    p <- autoplot(fair_decomp_img)
    ggsave(paste0(plot_path, "/tv_img.png"), plot=p)

    for (decomp in decompositions) {
        p <- autoplot(fair_decomp, decomp=decomp)
        ggsave(paste0(plot_path, "/tv_true_w_", decomp, ".png"), plot=p)
        
        p <- autoplot(fair_decomp_prime, decomp=decomp)
        ggsave(paste0(plot_path, "/tv_prime_w_", decomp, ".png"), plot=p)
        
        p <- autoplot(fair_decomp_no_W, decomp=decomp)
        ggsave(paste0(plot_path, "/tv_no_w_", decomp, ".png"), plot=p)
        
        p <- autoplot(fair_decomp_img, decomp=decomp)
        ggsave(paste0(plot_path, "/tv_img_", decomp, ".png"), plot=p)
    }
}