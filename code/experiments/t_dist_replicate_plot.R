#!/usr/bin/env Rscript
# Plot cross-replicate results for the Student-t experiment
# Uses the 20-rep Hopper results in code/results/

library(graphics)

files <- list.files("code/results", pattern = "t_dist_rep.*\\.csv", full.names = TRUE)
dfs <- lapply(files, read.csv)
df <- do.call(rbind, dfs)

cat(sprintf("%d replications loaded\n", nrow(df)))

pdf("fig/t_dist_replicates.pdf", width = 10, height = 4)
par(mfrow = c(1, 3), mar = c(4.2, 4.2, 2, 1))

# Panel 1: Selected nu across replications
nu_mle <- df$nu_mle
nu_gen <- df$nu_gen
jitter_x <- 0.15
plot(rep(1, nrow(df)) - jitter_x, nu_mle, pch = 16, col = rgb(0, 0, 0, 0.5),
     xlim = c(0.5, 2.5), ylim = range(c(nu_mle, nu_gen, 4)) * c(0.8, 1.1),
     xlab = "", ylab = expression(paste("Selected ", nu)),
     xaxt = "n", cex = 1.2)
points(rep(2, nrow(df)) + jitter_x, nu_gen, pch = 17, col = rgb(0, 0, 0, 0.5), cex = 1.2)
# Means
points(1 - jitter_x, mean(nu_mle), pch = 16, col = "blue", cex = 2)
points(2 + jitter_x, mean(nu_gen), pch = 17, col = "blue", cex = 2)
abline(h = 4, col = "red", lwd = 2, lty = 2)
axis(1, at = c(1, 2), labels = c("Profile MLE", "Generator"))
legend("topright", legend = c("Individual reps", "Mean", expression(paste("True ", nu, " = 4"))),
       pch = c(16, 16, NA), lty = c(NA, NA, 2), col = c(rgb(0,0,0,0.5), "blue", "red"),
       lwd = c(NA, NA, 2), pt.cex = c(1.2, 2, NA), bty = "n", cex = 0.85)

# Panel 2: mu estimates across replications
mu_mle <- df$mu_mle
mu_gen <- df$mu_gen
plot(rep(1, nrow(df)) - jitter_x, mu_mle, pch = 16, col = rgb(0, 0, 0, 0.5),
     xlim = c(0.5, 2.5), ylim = range(c(mu_mle, mu_gen, 2)) * c(0.9, 1.05),
     xlab = "", ylab = expression(hat(mu)),
     xaxt = "n", cex = 1.2)
points(rep(2, nrow(df)) + jitter_x, mu_gen, pch = 17, col = rgb(0, 0, 0, 0.5), cex = 1.2)
points(1 - jitter_x, mean(mu_mle), pch = 16, col = "blue", cex = 2)
points(2 + jitter_x, mean(mu_gen), pch = 17, col = "blue", cex = 2)
abline(h = 2, col = "red", lwd = 2, lty = 2)
axis(1, at = c(1, 2), labels = c("Profile MLE", "Generator"))

# Panel 3: sigma estimates across replications
sigma_mle <- df$sigma_mle
sigma_gen <- df$sigma_gen
plot(rep(1, nrow(df)) - jitter_x, sigma_mle, pch = 16, col = rgb(0, 0, 0, 0.5),
     xlim = c(0.5, 2.5), ylim = range(c(sigma_mle, sigma_gen, 1.5)) * c(0.85, 1.1),
     xlab = "", ylab = expression(hat(sigma)),
     xaxt = "n", cex = 1.2)
points(rep(2, nrow(df)) + jitter_x, sigma_gen, pch = 17, col = rgb(0, 0, 0, 0.5), cex = 1.2)
points(1 - jitter_x, mean(sigma_mle), pch = 16, col = "blue", cex = 2)
points(2 + jitter_x, mean(sigma_gen), pch = 17, col = "blue", cex = 2)
abline(h = 1.5, col = "red", lwd = 2, lty = 2)
axis(1, at = c(1, 2), labels = c("Profile MLE", "Generator"))

dev.off()
cat("Wrote fig/t_dist_replicates.pdf\n")
