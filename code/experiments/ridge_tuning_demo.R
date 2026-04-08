#!/usr/bin/env Rscript

# Empirical illustration: ridge tuning (GCV vs K-fold CV) and amortized evaluation
# Produces:
#   fig/ridge_tuning_demo.pdf
#   fig/ridge_tuning_table.tex
#
# Run:
#   Rscript code/experiments/ridge_tuning_demo.R

set.seed(123)
options(stringsAsFactors = FALSE)

ensure_dir <- function(path) if (!dir.exists(path)) dir.create(path, recursive = TRUE)
ensure_dir("fig")

n <- 300
p <- 80
sigma <- 1

# correlated design
rho <- 0.2
Sigma <- rho ^ abs(outer(1:p, 1:p, "-"))
X <- matrix(rnorm(n * p), n, p) %*% chol(Sigma)
beta0 <- c(rep(1, 8), rep(0, p - 8))
y <- as.numeric(X %*% beta0 + rnorm(n, sd = sigma))

# hold-out test set
n_te <- 300
Xte <- matrix(rnorm(n_te * p), n_te, p) %*% chol(Sigma)
yte <- as.numeric(Xte %*% beta0 + rnorm(n_te, sd = sigma))

lam_grid <- exp(seq(log(1e-4), log(5), length.out = 80))

ridge_fit <- function(X, y, lam) {
  XtX <- crossprod(X)
  rhs <- crossprod(X, y)
  solve(XtX + (n * lam) * diag(ncol(X)), rhs)
}

ridge_pred <- function(beta, X) as.numeric(X %*% beta)

gcv_score <- function(X, y, lam, eigvals = NULL) {
  # V(lam) = (rss/n) / (1 - tr(A)/n)^2, A = X (X'X + n lam I)^{-1} X'
  n <- nrow(X)
  beta <- ridge_fit(X, y, lam)
  r <- y - ridge_pred(beta, X)
  rss <- mean(r^2)
  if (is.null(eigvals)) eigvals <- eigen(crossprod(X), only.values = TRUE)$values
  trA <- sum(eigvals / (eigvals + n * lam))
  rss / (1 - trA / n)^2
}

# K-fold CV (Standard)
start_time_cv <- Sys.time()
K <- 5
fold_id <- sample(rep(1:K, length.out = n))
cv_mse <- function(lam) {
  err <- numeric(K)
  for (k in 1:K) {
    tr <- which(fold_id != k)
    va <- which(fold_id == k)
    beta <- ridge_fit(X[tr, , drop = FALSE], y[tr], lam)
    pred <- ridge_pred(beta, X[va, , drop = FALSE])
    err[k] <- mean((y[va] - pred)^2)
  }
  mean(err)
}

eigvals <- eigen(crossprod(X), only.values = TRUE)$values
gcv_vals <- sapply(lam_grid, \(lam) gcv_score(X, y, lam, eigvals))
cv_vals <- sapply(lam_grid, cv_mse)
end_time_cv <- Sys.time()
time_cv <- as.numeric(end_time_cv - start_time_cv, units = "secs")

lam_gcv <- lam_grid[which.min(gcv_vals)]
lam_cv <- lam_grid[which.min(cv_vals)]

beta_gcv <- ridge_fit(X, y, lam_gcv)
beta_cv <- ridge_fit(X, y, lam_cv)
mse_gcv <- mean((yte - ridge_pred(beta_gcv, Xte))^2)
mse_cv <- mean((yte - ridge_pred(beta_cv, Xte))^2)

# Amortized evaluation (Corrected: Per-fold generator training)
# To avoid data leakage, we train the generator on the training set of each fold.
start_time_gen <- Sys.time()
B <- 25
cv_gen_vals <- numeric(length(lam_grid))
# We sum MSEs across folds then divide by K
total_gen_mse <- numeric(length(lam_grid))

for (k in 1:K) {
  tr <- which(fold_id != k)
  va <- which(fold_id == k)
  
  X_tr <- X[tr, , drop = FALSE]
  y_tr <- y[tr]
  X_va <- X[va, , drop = FALSE]
  y_va <- y[va]
  
  # 1. Train generator on this fold's training data
  # Sample B lambdas
  lam_train <- exp(runif(B, log(min(lam_grid)), log(max(lam_grid))))
  
  # Solve on training set
  Theta_tr <- sapply(lam_train, \(lam) ridge_fit(X_tr, y_tr, lam)) # p x B
  
  # Fit linear map A_k: (1, log(lam), log(lam)^2) -> beta
  Z <- rbind(1, log(lam_train), log(lam_train)^2) # 3 x B
  ZZt <- Z %*% t(Z)
  # Ridge regression for the generator weights (to handle stability)
  A_k <- Theta_tr %*% t(Z) %*% solve(ZZt + 1e-10 * diag(nrow(ZZt))) # p x 3
  
  # 2. Evaluate on validation set for ALL grid lambdas using generator
  Z_grid <- rbind(1, log(lam_grid), log(lam_grid)^2) # 3 x n_grid
  Beta_pred <- A_k %*% Z_grid # p x n_grid
  
  # Predictions: X_va (n_val x p) * Beta_pred (p x n_grid) -> (n_val x n_grid)
  Preds <- X_va %*% Beta_pred
  
  # Compute MSE per lambda
  # y_va is vector of length n_val. sweep subtracts vector from each column? 
  # No, sweep(x, 1, stat) subtracts stat[i] from x[i, ]. Correct for rows.
  Resids <- sweep(Preds, 1, y_va, "-")
  total_gen_mse <- total_gen_mse + colMeans(Resids^2)
}
cv_gen_vals <- total_gen_mse / K
end_time_gen <- Sys.time()
time_gen <- as.numeric(end_time_gen - start_time_gen, units = "secs")

lam_gen_cv <- lam_grid[which.min(cv_gen_vals)]
# Final refit at selected lambda (exact)
beta_gen_cv <- ridge_fit(X, y, lam_gen_cv)
mse_gen_cv <- mean((yte - ridge_pred(beta_gen_cv, Xte))^2)

# Print timings
cat(sprintf("Standard CV time: %.2f s\n", time_cv))
cat(sprintf("Amortized CV time: %.2f s\n", time_gen))

# ---- Figure: tuning curves ----
pdf("fig/ridge_tuning_demo.pdf", width = 6.7, height = 4.5)
par(mar = c(4.2, 4.2, 1.2, 1))
plot(lam_grid, cv_vals, log = "x", type = "l", lwd = 2,
     xlab = expression(lambda), ylab = "Estimated validation MSE",
     ylim = range(c(cv_vals, cv_gen_vals)))
lines(lam_grid, cv_gen_vals, lwd = 2, lty = 2)
abline(v = lam_cv, col = "gray40")
abline(v = lam_gcv, col = "gray40", lty = 3)
abline(v = lam_gen_cv, col = "gray40", lty = 2)
legend("topright",
       legend = c("5-fold CV (exact refits)", "5-fold CV (amortized proxy)",
                  expression(paste("Selected ", lambda, ": CV")),
                  expression(paste("Selected ", lambda, ": GCV")),
                  expression(paste("Selected ", lambda, ": amortized"))),
       lwd = c(2, 2, 1, 1, 1), lty = c(1, 2, 1, 3, 2),
       col = c("black", "black", "gray40", "gray40", "gray40"),
       bty = "n", cex = 0.85)
dev.off()

# ---- Table (LaTeX) ----
fmt <- function(x) formatC(x, digits = 3, format = "f")
tab <- c(
  "\\begin{table}[H]",
  "\\centering",
  "\\caption{Ridge tuning illustration (simulated data). We compare generalized cross-validation (GCV), $K$-fold cross-validation (CV), and a simple amortized generator proxy that approximates the CV curve without repeated refits across $\\lambda$. Test MSE is computed on an independent hold-out test set.}",
  "\\label{tab:ridge-tuning}",
  "\\begin{tabular}{lcc}",
  "\\hline",
  "Method & Selected $\\lambda$ & Test MSE\\\\",
  "\\hline",
  paste0("GCV & ", fmt(lam_gcv), " & ", fmt(mse_gcv), " \\\\"),
  paste0("5-fold CV & ", fmt(lam_cv), " & ", fmt(mse_cv), " \\\\"),
  paste0("Amortized CV proxy (refit at selected $\\lambda$) & ", fmt(lam_gen_cv), " & ", fmt(mse_gen_cv), " \\\\"),
  "\\hline",
  "\\end{tabular}",
  "\\end{table}"
)
writeLines(tab, con = "fig/ridge_tuning_table.tex")

cat("Wrote fig/ridge_tuning_demo.pdf and fig/ridge_tuning_table.tex\n")


