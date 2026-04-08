#!/usr/bin/env Rscript
# t-distribution tuning experiment for Hopper (multi-rep)
# Usage: Rscript t_dist_hopper.R --rep 1

args <- commandArgs(trailingOnly = TRUE)
# R absorbs --rep as its own flag; the value arrives as bare arg
rep_idx <- as.integer(args[length(args)])
stopifnot(length(rep_idx) == 1, !is.na(rep_idx))

cat(sprintf("Rep: %d\n", rep_idx))

seed <- rep_idx * 31 + 7
set.seed(seed)

dir.create("results", showWarnings = FALSE, recursive = TRUE)

# ---- Parameters ----
nu_true <- 4
mu_true <- 2.0
sigma_true <- 1.5
n <- 500
n_train <- 350
n_val <- 150
n_blocks <- 16
B <- 200
M_wbb <- 200
nu_grid <- seq(1.5, 15, by = 0.5)

# ---- Generate data ----
y_all <- mu_true + sigma_true * rt(n, df = nu_true)
y_train <- y_all[1:n_train]
y_val <- y_all[(n_train + 1):n]

# ---- Functions ----
nll_t <- function(y, mu, sigma, nu) {
  z <- (y - mu) / sigma
  -sum(dt(z, df = nu, log = TRUE) - log(sigma))
}

wnll_t <- function(y, mu, sigma, nu, w) {
  z <- (y - mu) / sigma
  -sum(w * (dt(z, df = nu, log = TRUE) - log(sigma)))
}

wmle_t <- function(y, nu, w = rep(1, length(y))) {
  obj <- function(par) {
    mu <- par[1]; sigma <- exp(par[2])
    wnll_t(y, mu, sigma, nu, w)
  }
  mu0 <- weighted.mean(y, w)
  sigma0 <- sqrt(sum(w * (y - mu0)^2) / sum(w))
  res <- optim(c(mu0, log(sigma0)), obj, method = "BFGS")
  c(mu = res$par[1], sigma = exp(res$par[2]))
}

block_id <- sample(rep(1:n_blocks, length.out = n_train))
idx_by_block <- split(seq_len(n_train), block_id)

sample_wbb_weights <- function() {
  g <- rgamma(n_blocks, shape = 1, rate = 1)
  alpha <- n_blocks * g / sum(g)
  w <- numeric(n_train)
  for (s in seq_along(idx_by_block)) w[idx_by_block[[s]]] <- alpha[s]
  list(w = w, alpha = alpha)
}

make_features <- function(alpha, nu) c(1, nu, nu^2, alpha)
k_feat <- 3 + n_blocks

# ---- Profile MLE ----
t0 <- proc.time()[["elapsed"]]
val_nll_mle <- numeric(length(nu_grid))
mle_params <- matrix(NA, length(nu_grid), 2)
for (j in seq_along(nu_grid)) {
  par <- wmle_t(y_train, nu_grid[j])
  mle_params[j, ] <- par
  val_nll_mle[j] <- nll_t(y_val, par[1], par[2], nu_grid[j])
}
nu_mle <- nu_grid[which.min(val_nll_mle)]
par_mle <- mle_params[which.min(val_nll_mle), ]
time_mle <- proc.time()[["elapsed"]] - t0

# ---- WBB + Generator ----
t0 <- proc.time()[["elapsed"]]
gen_alphas <- matrix(NA, B, n_blocks)
gen_nus <- numeric(B)
gen_thetas <- matrix(NA, B, 2)
for (b in seq_len(B)) {
  wbb <- sample_wbb_weights()
  nu <- sample(nu_grid, 1)
  par <- wmle_t(y_train, nu, wbb$w)
  gen_alphas[b, ] <- wbb$alpha
  gen_nus[b] <- nu
  gen_thetas[b, ] <- par
}
Z <- t(sapply(seq_len(B), function(b) make_features(gen_alphas[b, ], gen_nus[b])))
A_gen <- solve(crossprod(Z) + 1e-8 * diag(k_feat), crossprod(Z, gen_thetas))
time_gen_train <- proc.time()[["elapsed"]] - t0

alpha_ones <- rep(1, n_blocks)
val_nll_gen <- numeric(length(nu_grid))
gen_params <- matrix(NA, length(nu_grid), 2)
for (j in seq_along(nu_grid)) {
  z <- make_features(alpha_ones, nu_grid[j])
  par <- as.numeric(crossprod(A_gen, z))
  par[2] <- max(par[2], 0.01)
  gen_params[j, ] <- par
  val_nll_gen[j] <- nll_t(y_val, par[1], par[2], nu_grid[j])
}
nu_gen <- nu_grid[which.min(val_nll_gen)]
par_gen <- gen_params[which.min(val_nll_gen), ]

# ---- WBB posterior draws ----
wbb_mu <- wbb_sigma <- numeric(M_wbb)
for (m in seq_len(M_wbb)) {
  wbb <- sample_wbb_weights()
  z <- make_features(wbb$alpha, nu_gen)
  par <- as.numeric(crossprod(A_gen, z))
  par[2] <- max(par[2], 0.01)
  wbb_mu[m] <- par[1]
  wbb_sigma[m] <- par[2]
}

# ---- Save results ----
res <- data.frame(
  rep = rep_idx,
  nu_true = nu_true, mu_true = mu_true, sigma_true = sigma_true,
  nu_mle = nu_mle, mu_mle = par_mle[1], sigma_mle = par_mle[2],
  nu_gen = nu_gen, mu_gen = par_gen[1], sigma_gen = par_gen[2],
  wbb_mu_mean = mean(wbb_mu), wbb_mu_sd = sd(wbb_mu),
  wbb_sigma_mean = mean(wbb_sigma), wbb_sigma_sd = sd(wbb_sigma),
  time_mle = time_mle, time_gen_train = time_gen_train
)
write.csv(res, sprintf("results/t_dist_rep%d.csv", rep_idx), row.names = FALSE)
cat(sprintf("Done. nu_mle=%.1f, nu_gen=%.1f, time_mle=%.1f, time_gen=%.1f\n",
            nu_mle, nu_gen, time_mle, time_gen_train))
