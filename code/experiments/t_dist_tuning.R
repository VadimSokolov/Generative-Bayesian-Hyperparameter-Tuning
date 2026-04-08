#!/usr/bin/env Rscript

# Empirical illustration: tuning degrees of freedom (nu) for the t-distribution
#
# The Student-t distribution has degrees-of-freedom parameter nu that controls
# tail heaviness. Estimating nu jointly with location and scale via MLE or MCMC
# is well known to be difficult (the likelihood surface is flat in nu; MCMC
# mixing is slow). Here we apply the WBB + generator template:
#   - Inner problem: weighted negative log-likelihood of t(nu) with WBB weights
#   - Generator: maps (omega, nu) -> (mu_hat, sigma_hat)
#   - Outer criterion: validation negative log-likelihood over a grid of nu
#
# Produces:
#   fig/t_dist_tuning.pdf
#   fig/t_dist_tuning_table.tex
#
# Run:
#   Rscript code/experiments/t_dist_tuning.R

set.seed(42)
options(stringsAsFactors = FALSE)

ensure_dir <- function(path) if (!dir.exists(path)) dir.create(path, recursive = TRUE)
ensure_dir("fig")

# ---- Data generation ----
# Generate data from a t-distribution with known parameters
nu_true <- 4       # degrees of freedom (ground truth)
mu_true <- 2.0     # location
sigma_true <- 1.5  # scale
n <- 500           # total observations
n_train <- 350
n_val <- 150

y_all <- mu_true + sigma_true * rt(n, df = nu_true)
y_train <- y_all[1:n_train]
y_val <- y_all[(n_train + 1):n]

# ---- Negative log-likelihood for location-scale t ----
# p(y | mu, sigma, nu) = dt((y - mu)/sigma, df = nu) / sigma
nll_t <- function(y, mu, sigma, nu) {
  z <- (y - mu) / sigma
  -sum(dt(z, df = nu, log = TRUE) - log(sigma))
}

# Weighted negative log-likelihood
wnll_t <- function(y, mu, sigma, nu, w) {
  z <- (y - mu) / sigma
  -sum(w * (dt(z, df = nu, log = TRUE) - log(sigma)))
}

# ---- Weighted MLE for fixed nu (inner problem) ----
# For fixed nu, find (mu, sigma) minimizing weighted NLL
# Use optim with L-BFGS-B (sigma > 0)
wmle_t <- function(y, nu, w = rep(1, length(y))) {
  obj <- function(par) {
    mu <- par[1]
    log_sigma <- par[2]  # optimize on log scale for positivity
    sigma <- exp(log_sigma)
    wnll_t(y, mu, sigma, nu, w)
  }
  # Initialize at weighted mean and sd
  mu0 <- weighted.mean(y, w)
  sigma0 <- sqrt(sum(w * (y - mu0)^2) / sum(w))
  res <- optim(c(mu0, log(sigma0)), obj, method = "BFGS")
  c(mu = res$par[1], sigma = exp(res$par[2]))
}

# ---- WBB block weights ----
n_blocks <- 16
block_id <- sample(rep(1:n_blocks, length.out = n_train))
idx_by_block <- split(seq_len(n_train), block_id)

sample_wbb_weights <- function() {
  # Dirichlet(1,...,1) scaled by n_blocks
  g <- rgamma(n_blocks, shape = 1, rate = 1)
  alpha <- n_blocks * g / sum(g)
  w <- numeric(n_train)
  for (s in seq_along(idx_by_block)) w[idx_by_block[[s]]] <- alpha[s]
  list(w = w, alpha = alpha)
}

# ---- Grid of nu values ----
nu_grid <- seq(1.5, 15, by = 0.5)

# ---- Method 1: Standard MLE profile over nu (no WBB) ----
val_nll_mle <- numeric(length(nu_grid))
train_nll_mle <- numeric(length(nu_grid))
mle_params <- matrix(NA, length(nu_grid), 2, dimnames = list(NULL, c("mu", "sigma")))

for (j in seq_along(nu_grid)) {
  nu <- nu_grid[j]
  par <- wmle_t(y_train, nu)
  mle_params[j, ] <- par
  train_nll_mle[j] <- nll_t(y_train, par["mu"], par["sigma"], nu)
  val_nll_mle[j] <- nll_t(y_val, par["mu"], par["sigma"], nu)
}

nu_mle <- nu_grid[which.min(val_nll_mle)]
par_mle <- mle_params[which.min(val_nll_mle), ]

# ---- Method 2: WBB + generator (amortized) ----
# Train a simple linear generator: (alpha, nu) -> (mu, sigma)
# Features: (1, nu, nu^2, alpha_1, ..., alpha_K)
make_features <- function(alpha, nu) {
  c(1, nu, nu^2, alpha)
}
k_feat <- 3 + n_blocks  # dimension of feature vector

# Generate training data for the generator
B <- 200  # number of (omega, nu) pairs
gen_alphas <- matrix(NA, B, n_blocks)
gen_nus <- numeric(B)
gen_thetas <- matrix(NA, B, 2)  # (mu, sigma)

cat("Training generator (supervised mode)...\n")
for (b in seq_len(B)) {
  wbb <- sample_wbb_weights()
  nu <- sample(nu_grid, 1)
  par <- wmle_t(y_train, nu, wbb$w)
  gen_alphas[b, ] <- wbb$alpha
  gen_nus[b] <- nu
  gen_thetas[b, ] <- par
}

# Fit linear generator: theta = A %*% z
Z <- t(sapply(seq_len(B), function(b) make_features(gen_alphas[b, ], gen_nus[b])))
# OLS: A = (Z'Z)^{-1} Z' Theta  =>  A is k_feat x 2
# Prediction: par = t(A) %*% z  or equivalently  par = z %*% A (if z is a row)
A_gen <- solve(crossprod(Z) + 1e-8 * diag(k_feat), crossprod(Z, gen_thetas))  # k_feat x 2

# Evaluate on validation set using generator with unit weights
alpha_ones <- rep(1, n_blocks)
val_nll_gen <- numeric(length(nu_grid))
gen_params <- matrix(NA, length(nu_grid), 2, dimnames = list(NULL, c("mu", "sigma")))

for (j in seq_along(nu_grid)) {
  nu <- nu_grid[j]
  z <- make_features(alpha_ones, nu)
  par <- as.numeric(crossprod(A_gen, z))  # [mu, sigma]
  par[2] <- max(par[2], 0.01)  # ensure sigma > 0
  gen_params[j, ] <- par
  val_nll_gen[j] <- nll_t(y_val, par[1], par[2], nu)
}

nu_gen <- nu_grid[which.min(val_nll_gen)]
par_gen <- gen_params[which.min(val_nll_gen), ]

# ---- WBB uncertainty at selected nu ----
M_wbb <- 200
wbb_mu <- numeric(M_wbb)
wbb_sigma <- numeric(M_wbb)

cat("Computing WBB posterior draws...\n")
for (m in seq_len(M_wbb)) {
  wbb <- sample_wbb_weights()
  z <- make_features(wbb$alpha, nu_gen)
  par <- as.numeric(crossprod(A_gen, z))
  par[2] <- max(par[2], 0.01)
  wbb_mu[m] <- par[1]
  wbb_sigma[m] <- par[2]
}

cat(sprintf("True: nu=%g, mu=%.2f, sigma=%.2f\n", nu_true, mu_true, sigma_true))
cat(sprintf("MLE profile: nu=%.1f, mu=%.3f, sigma=%.3f\n", nu_mle, par_mle["mu"], par_mle["sigma"]))
cat(sprintf("Generator:   nu=%.1f, mu=%.3f, sigma=%.3f\n", nu_gen, par_gen[1], par_gen[2]))
cat(sprintf("WBB posterior mu:    mean=%.3f, sd=%.3f\n", mean(wbb_mu), sd(wbb_mu)))
cat(sprintf("WBB posterior sigma: mean=%.3f, sd=%.3f\n", mean(wbb_sigma), sd(wbb_sigma)))

# ---- Figure (single panel: validation NLL only) ----
pdf("fig/t_dist_tuning.pdf", width = 6.5, height = 4.5)
par(mar = c(4.2, 4.2, 1.2, 1))

plot(nu_grid, val_nll_mle, type = "l", lwd = 2,
     xlab = expression(nu), ylab = "Validation negative log-likelihood")
lines(nu_grid, val_nll_gen, lwd = 2, lty = 2)
abline(v = nu_true, col = "gray50", lty = 3)
abline(v = nu_mle, col = "black", lty = 1, lwd = 0.8)
abline(v = nu_gen, col = "black", lty = 2, lwd = 0.8)
legend("topright",
       legend = c("Profile MLE", "Generator (amortized)",
                  expression(paste("True ", nu))),
       lwd = c(2, 2, 1), lty = c(1, 2, 3),
       col = c("black", "black", "gray50"),
       bty = "n", cex = 0.85)

# Panel 2: WBB posterior draws for (mu, sigma) at selected nu
dev.off()

# ---- Table (LaTeX) ----
fmt <- function(x) formatC(x, digits = 3, format = "f")
fmt1 <- function(x) formatC(x, digits = 1, format = "f")

tab <- c(
  "\\begin{table}[H]",
  "\\centering",
  paste0("\\caption{Student-$t$ tuning illustration ($n_{\\mathrm{train}}=", n_train,
         "$, $n_{\\mathrm{val}}=", n_val,
         "$). True parameters: $\\nu=", nu_true,
         "$, $\\mu=", fmt(mu_true),
         "$, $\\sigma=", fmt(sigma_true), "$.}"),
  "\\label{tab:t-dist-tuning}",
  "\\begin{tabular}{lccc}",
  "\\hline",
  "Method & Selected $\\nu$ & $\\hat{\\mu}$ & $\\hat{\\sigma}$ \\\\",
  "\\hline",
  paste0("Profile MLE & ", fmt1(nu_mle), " & ", fmt(par_mle["mu"]), " & ", fmt(par_mle["sigma"]), " \\\\"),
  paste0("Generator (amortized) & ", fmt1(nu_gen), " & ", fmt(par_gen[1]), " & ", fmt(par_gen[2]), " \\\\"),
  paste0("WBB posterior mean ($\\pm$ sd) & --- & ",
         fmt(mean(wbb_mu)), " ($\\pm$ ", fmt(sd(wbb_mu)), ") & ",
         fmt(mean(wbb_sigma)), " ($\\pm$ ", fmt(sd(wbb_sigma)), ") \\\\"),
  "\\hline",
  "\\end{tabular}",
  "\\end{table}"
)
writeLines(tab, con = "fig/t_dist_tuning_table.tex")

cat("Wrote fig/t_dist_tuning.pdf and fig/t_dist_tuning_table.tex\n")
