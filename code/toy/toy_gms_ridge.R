#!/usr/bin/env Rscript

# Toy illustration (Shin-style setup): weighted ridge regression
# Compare two generator training modes within a simple linear generator family:
#   (A) supervised regression to optimizer labels theta_hat(w, lambda)
#   (B) criterion-based training that minimizes the integrated objective value
#
# Outputs:
#   fig/toy_gms_ipl_vs_B.pdf   - IPL curves vs B
#   fig/toy_gms_ipl_vs_time.pdf - IPL curves vs training time (compute-matched view)
#   fig/toy_gms_table.tex      - LaTeX table of IPL at selected B values
#
# Run:
#   Rscript code/toy/toy_gms_ridge.R

set.seed(123)
options(stringsAsFactors = FALSE)

ensure_dir <- function(path) if (!dir.exists(path)) dir.create(path, recursive = TRUE)
ensure_dir("fig")

# ---- Simulation setup ----
n <- 200
p <- 20
S <- 25  # number of blocks / subgroup weights (dimension reduction for w)

sigma <- 1.0

# Correlated design (simple AR1 covariance)
rho <- 0.3
Sigma <- rho ^ abs(outer(1:p, 1:p, "-"))
X <- matrix(rnorm(n * p), n, p) %*% chol(Sigma)

theta0 <- c(rep(1, 5), rep(0, p - 5))
y <- as.numeric(X %*% theta0 + rnorm(n, sd = sigma))

# Fixed partition into S blocks (roughly equal size)
block_id <- sample(rep(1:S, length.out = n))
idx_by_block <- split(seq_len(n), block_id)

# Weight distribution on blocks: alpha ~ S * Dirichlet(1,...,1) => alpha sums to S, E[alpha]=1
rdirichlet_1 <- function(S) {
  g <- rgamma(S, shape = 1, rate = 1)
  g / sum(g)
}

sample_alpha <- function(S) S * rdirichlet_1(S)

alpha_to_w <- function(alpha, idx_by_block, n) {
  w <- numeric(n)
  for (s in seq_along(idx_by_block)) w[idx_by_block[[s]]] <- alpha[s]
  w
}

sample_lambda <- function() exp(runif(1, log(1e-3), log(1.0)))

make_z <- function(alpha, lambda) c(1, log(lambda), alpha) # k = 2 + S
k <- 2 + S

# Numerical stabilization (tiny ridge for matrix inversions)
ridge_eps <- 1e-8

# Closed-form weighted ridge optimizer
theta_hat <- function(X, y, w, lambda) {
  # Solve (X' W X + n lambda I) theta = X' W y
  WX <- X * w
  XtWX <- crossprod(X, WX)
  rhs <- crossprod(X, w * y)
  A <- XtWX + (n * lambda) * diag(ncol(X))
  solve(A, rhs)
}

# Linear generator: theta = A %*% z, where A is p x k
pred_theta <- function(A, z) as.numeric(A %*% z)

# Training set generator
draw_training_set <- function(B) {
  alphas <- vector("list", B)
  lambdas <- numeric(B)
  ws <- vector("list", B)
  zs <- matrix(0, k, B)
  thetas <- matrix(0, p, B)
  for (b in seq_len(B)) {
    alpha <- sample_alpha(S)
    lambda <- sample_lambda()
    w <- alpha_to_w(alpha, idx_by_block, n)
    th <- theta_hat(X, y, w, lambda)
    z <- make_z(alpha, lambda)
    alphas[[b]] <- alpha
    lambdas[b] <- lambda
    ws[[b]] <- w
    zs[, b] <- z
    thetas[, b] <- th
  }
  list(ws = ws, zs = zs, thetas = thetas, lambdas = lambdas, alphas = alphas)
}

# Mode A: supervised fit of A by multivariate OLS on labels theta_hat
fit_supervised <- function(zs, thetas) {
  # A = Theta * Z^T * (Z Z^T)^{-1}
  ZZt <- zs %*% t(zs)
  thetas %*% t(zs) %*% solve(ZZt + ridge_eps * diag(nrow(ZZt)))
}

# Mode B: criterion-based fit of A by minimizing sum_b L(A z_b; w_b, lambda_b)
fit_criterion <- function(ws, zs, lambdas) {
  # Quadratic form in a = vec(A) (column-major)
  # L_b(A) = (1/n)(y - M_b a)' W_b (y - M_b a) + lambda_b * ||A z_b||^2
  # where M_b = (z_b^T ⊗ X)
  Q <- matrix(0, p * k, p * k)
  q <- numeric(p * k)
  I_p <- diag(p)
  for (b in seq_along(ws)) {
    w <- ws[[b]]
    z <- zs[, b, drop = FALSE] # k x 1
    lambda <- lambdas[b]
    # Compute M_b^T W y and M_b^T W M_b efficiently without forming W or M explicitly
    # M = (z^T ⊗ X) => (p*k x n)?? but we'll build using kronecker with manageable sizes
    M <- kronecker(t(z), X)  # n x (p*k)? Actually kronecker(t(z), X) gives n x (p*k)
    # Weighting by w: W y = w * y, W M = (w * 1_n) elementwise on rows
    Wy <- w * y
    WM <- M * w
    Q <- Q + (crossprod(M, WM) / n) + as.numeric(lambda) * kronecker(z %*% t(z), I_p)
    q <- q + (crossprod(M, Wy) / n)
  }
  a_hat <- solve(Q + ridge_eps * diag(nrow(Q)), q)
  matrix(a_hat, nrow = p, ncol = k, byrow = FALSE)
}

# Criterion-based training with Monte Carlo draws (no optimizer labels needed)
fit_criterion_mc <- function(M_mc) {
  Q <- matrix(0, p * k, p * k)
  q <- numeric(p * k)
  I_p <- diag(p)
  for (m in seq_len(M_mc)) {
    alpha <- sample_alpha(S)
    lambda <- sample_lambda()
    w <- alpha_to_w(alpha, idx_by_block, n)
    z <- make_z(alpha, lambda)
    z <- matrix(z, ncol = 1)
    Mmat <- kronecker(t(z), X)  # n x (p*k)
    Wy <- w * y
    WM <- Mmat * w
    Q <- Q + (crossprod(Mmat, WM) / n) + as.numeric(lambda) * kronecker(z %*% t(z), I_p)
    q <- q + (crossprod(Mmat, Wy) / n)
  }
  a_hat <- solve(Q + ridge_eps * diag(nrow(Q)), q)
  matrix(a_hat, nrow = p, ncol = k, byrow = FALSE)
}

# Evaluate IPL on a fresh test set
eval_ipl <- function(A, M_test) {
  se <- numeric(M_test)
  for (m in seq_len(M_test)) {
    alpha <- sample_alpha(S)
    lambda <- sample_lambda()
    w <- alpha_to_w(alpha, idx_by_block, n)
    th_true <- theta_hat(X, y, w, lambda)
    z <- make_z(alpha, lambda)
    th_pred <- pred_theta(A, z)
    se[m] <- sum((th_pred - th_true)^2)
  }
  mean(se)
}

# ---- Experiment: IPL vs B ----
B_grid <- c(25, 50, 100, 200, 400, 800)
R_rep <- 12
M_test <- 300
Mmc_grid <- c(200, 500, 1000, 2000, 5000, 10000)  # unlabeled draws for criterion-based training

ipl_sup <- matrix(NA_real_, nrow = length(B_grid), ncol = R_rep)
ipl_cri <- matrix(NA_real_, nrow = length(Mmc_grid), ncol = R_rep)

time_sup <- matrix(NA_real_, nrow = length(B_grid), ncol = R_rep)
time_cri <- matrix(NA_real_, nrow = length(Mmc_grid), ncol = R_rep)

for (r in seq_len(R_rep)) {
  set.seed(1000 + r)
  # criterion-based training does not require optimizer labels; vary Monte Carlo budget
  for (j in seq_along(Mmc_grid)) {
    M_mc <- Mmc_grid[j]
    t0 <- proc.time()[["elapsed"]]
    A_cri <- fit_criterion_mc(M_mc)
    t1 <- proc.time()[["elapsed"]]
    time_cri[j, r] <- t1 - t0
    ipl_cri[j, r] <- eval_ipl(A_cri, M_test)
  }
  for (j in seq_along(B_grid)) {
    B <- B_grid[j]
    t0 <- proc.time()[["elapsed"]]
    dat <- draw_training_set(B)  # includes B optimizer solves (labels)
    A_sup <- fit_supervised(dat$zs, dat$thetas)
    t1 <- proc.time()[["elapsed"]]
    time_sup[j, r] <- t1 - t0
    ipl_sup[j, r] <- eval_ipl(A_sup, M_test)
  }
}

mean_sup <- rowMeans(ipl_sup)
mean_cri <- rowMeans(ipl_cri)
se_sup <- apply(ipl_sup, 1, sd) / sqrt(R_rep)
se_cri <- apply(ipl_cri, 1, sd) / sqrt(R_rep)

# time means
mean_time_sup <- rowMeans(time_sup)
mean_time_cri <- rowMeans(time_cri)
se_time_sup <- apply(time_sup, 1, sd) / sqrt(R_rep)
se_time_cri <- apply(time_cri, 1, sd) / sqrt(R_rep)

# ---- Figure ----
pdf("fig/toy_gms_ipl_vs_B.pdf", width = 6.5, height = 4.2)
par(mar = c(4.2, 4.2, 1.2, 1))
plot(
  B_grid, mean_sup, log = "x", type = "b", pch = 16, lwd = 2,
  xlab = "Number of training pairs B (log scale)",
  ylab = "Integrated prediction loss (IPL)",
  ylim = range(c(mean_sup - 2 * se_sup, mean_sup + 2 * se_sup))
)
if (any(se_sup > 0)) {
  ok <- se_sup > 0
  arrows(B_grid[ok], mean_sup[ok] - 2 * se_sup[ok], B_grid[ok], mean_sup[ok] + 2 * se_sup[ok],
         angle = 90, code = 3, length = 0.04)
}
legend(
  "topright",
  legend = c("Supervised match-to-optimizer"),
  pch = c(16), lwd = 2, bty = "n"
)
dev.off()

# ---- Compute-matched figure: IPL vs training time (log-log) ----
pdf("fig/toy_gms_ipl_vs_time.pdf", width = 6.5, height = 4.2)
par(mar = c(4.2, 4.2, 1.2, 1))
all_ipl <- c(mean_sup, mean_cri)
all_time <- c(mean_time_sup, mean_time_cri)
plot(
  mean_time_sup, mean_sup, log = "xy", type = "b", pch = 16, lwd = 2,
  xlab = "Training time (seconds, log scale)",
  ylab = "Integrated prediction loss (IPL, log scale)",
  ylim = range(all_ipl[all_ipl > 0]) * c(0.7, 1.5),
  xlim = range(all_time[all_time > 0])
)
lines(mean_time_cri, mean_cri, type = "b", pch = 17, lwd = 2)
if (any(se_sup > 0)) {
  ok <- se_sup > 0
  arrows(mean_time_sup[ok], mean_sup[ok] - 2 * se_sup[ok],
         mean_time_sup[ok], mean_sup[ok] + 2 * se_sup[ok],
         angle = 90, code = 3, length = 0.04)
}
if (any(se_cri > 0)) {
  ok <- se_cri > 0
  arrows(mean_time_cri[ok], mean_cri[ok] - 2 * se_cri[ok],
         mean_time_cri[ok], mean_cri[ok] + 2 * se_cri[ok],
         angle = 90, code = 3, length = 0.04)
}
text(mean_time_sup, mean_sup, labels = B_grid, pos = 3, cex = 0.75)
text(mean_time_cri, mean_cri, labels = Mmc_grid, pos = 1, cex = 0.75)
legend(
  "topright",
  legend = c("Supervised (labels B)", "Criterion-based (MC draws M)"),
  pch = c(16, 17), lwd = 2, bty = "n"
)
dev.off()

# ---- Table (LaTeX) ----
# For supervised: report at B = 50, 200, 800 (sel_idx_sup in B_grid)
sel_B <- c(50, 200, 800)
sel_idx_sup <- match(sel_B, B_grid)

# For criterion-based: report at M = 500, 2000, 10000 (sel_idx_cri in Mmc_grid)
sel_M <- c(500, 2000, 10000)
sel_idx_cri <- match(sel_M, Mmc_grid)

fmt <- function(x) formatC(x, digits = 3, format = "f")

tab_lines <- c(
  "\\begin{table}[H]",
  "\\centering",
  "\\caption{Toy weighted-ridge example: integrated prediction loss (IPL) for a linear generator trained either by supervised matching to optimizer labels ($B$ optimizer solves) or by criterion-based (GMS-style) training ($M$ Monte Carlo draws). Values are means over 12 replications with Monte Carlo standard errors in parentheses.}",
  "\\label{tab:toy-gms-ipl}",
  "\\begin{tabular}{lccc}",
  "\\hline",
  " & \\multicolumn{3}{c}{Training budget}\\\\",
  paste0("Method & ", paste(c("$B{=}50$", "$B{=}200$", "$B{=}800$"), collapse = " & "), "\\\\"),
  "\\hline"
)

row_sup <- character(length(sel_idx_sup))
for (t in seq_along(sel_idx_sup)) {
  j <- sel_idx_sup[t]
  row_sup[t] <- paste0(fmt(mean_sup[j]), " (", fmt(se_sup[j]), ")")
}

tab_lines <- c(
  tab_lines,
  paste0("Supervised match-to-optimizer & ", paste(row_sup, collapse = " & "), " \\\\"),
  "\\hline",
  paste0(" & ", paste(c("$M{=}500$", "$M{=}2{,}000$", "$M{=}10{,}000$"), collapse = " & "), "\\\\"),
  "\\hline"
)

row_cri <- character(length(sel_idx_cri))
for (t in seq_along(sel_idx_cri)) {
  j <- sel_idx_cri[t]
  row_cri[t] <- paste0(fmt(mean_cri[j]), " (", fmt(se_cri[j]), ")")
}

tab_lines <- c(
  tab_lines,
  paste0("Criterion-based (GMS-style) & ", paste(row_cri, collapse = " & "), " \\\\"),
  "\\hline",
  "\\end{tabular}",
  "\\end{table}"
)

writeLines(tab_lines, con = "fig/toy_gms_table.tex")

cat("Wrote fig/toy_gms_ipl_vs_B.pdf, fig/toy_gms_ipl_vs_time.pdf and fig/toy_gms_table.tex\n")


