all_plots <- function(path, plot_path, img_dim=32, latent_dim=16, valid_latents = NA) {
    library(faircause)
    library(data.table)
    library(ggplot2)

    if (is.na(valid_latents)) {
        valid_latents <- 0:(latent_dim-1)
    }

    df <- read.csv(path)
    df <- data.table(df)
    W <- paste0("W_", valid_latents)
    W_prime <- paste0("W_prime_", valid_latents)
    W_empty <- list()
    W_img <- paste0("image_", 0:1023)
    W_emb <- paste0("embedding_", 0:127)
    X <- "X_0"
    Y <- "Y"
    Z <- "Z_0"
    
    print("Computing Fairness Decompositions")

    fair_decomp <- fairness_cookbook(df, X=X, W=W, Z=Z, Y=Y, x0=0, x1=1)
    # fair_decomp_prime <- fairness_cookbook(df, X=X, W=W_prime, Z=Z, Y=Y, x0=0, x1=1)
    fair_decomp_no_W <- fairness_cookbook(df, X=X, W=W_empty, Z=Z, Y=Y, x0=0, x1=1)
    fair_decomp_img <- fairness_cookbook(df, X=X, W=W_img, Z=Z, Y=Y, x0=0, x1=1)
    fair_decomp_emb <- fairness_cookbook(df, X=X, W=W_emb, Z=Z, Y=Y, x0=0, x1=1, embed=TRUE)

    decompositions <- c("xspec", "both", "general")

    print("Saving Plots")
    p <- autoplot(fair_decomp)
    ggsave(paste0(plot_path, "/tv_true_w.png"), plot=p)
    # p <- autoplot(fair_decomp_prime)
    # ggsave(paste0(plot_path, "/tv_prime_w.png"), plot=p)
    p <- autoplot(fair_decomp_no_W)
    ggsave(paste0(plot_path, "/tv_no_w.png"), plot=p)
    p <- autoplot(fair_decomp_img)
    ggsave(paste0(plot_path, "/tv_img.png"), plot=p)
    p <- autoplot(fair_decomp_emb)
    ggsave(paste0(plot_path, "/tv_emb.png"), plot=p)

    for (decomp in decompositions) {
        p <- autoplot(fair_decomp, decomp=decomp)
        ggsave(paste0(plot_path, "/tv_true_w_", decomp, ".png"), plot=p)
        
        # p <- autoplot(fair_decomp_prime, decomp=decomp)
        # ggsave(paste0(plot_path, "/tv_prime_w_", decomp, ".png"), plot=p)
        
        p <- autoplot(fair_decomp_no_W, decomp=decomp)
        ggsave(paste0(plot_path, "/tv_no_w_", decomp, ".png"), plot=p)
        
        p <- autoplot(fair_decomp_img, decomp=decomp)
        ggsave(paste0(plot_path, "/tv_img_", decomp, ".png"), plot=p)

        p <- autoplot(fair_decomp_emb, decomp=decomp)
        ggsave(paste0(plot_path, "/tv_emb_", decomp, ".png"), plot=p)
    }
}