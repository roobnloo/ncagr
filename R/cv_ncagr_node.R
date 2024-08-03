cv_ncagr_node <- function(y, uw, p, q, nlambda, lambda_factor, alpha,
                          maxit, tol, nfolds) {
  n <- length(y)
  nvars <- q + (p - 1) * (q + 1)
  nalpha <- length(alpha)

  foldid <- cut(sample(seq_len(n)), nfolds, labels = FALSE)
  groupid <- c(rep(0, q), rep(1:(q + 1), each = p - 1)) + 1
  cvm_mx <- matrix(0, nrow = nlambda, ncol = nalpha)
  coefs <- matrix(nrow = nvars, ncol = nalpha)
  lambda <- numeric(nlambda)
  mse <- numeric(nalpha)
  # df <- numeric(nalpha)
  uws <- uw %*% diag(1 / sqrt(colSums(uw^2)))
  lam_max <- max(abs(t(uws) %*% y))
  lambda <- lam_max * exp(seq(log(1), log(lambda_factor), length = nlambda))

  for (asid in seq_along(alpha)) {
    asparse <- alpha[asid]
    pf_group <- c(0, 0, rep(1, q))
    sgl1 <- sparsegl::cv.sparsegl(
      uw, y,
      group = groupid,
      foldid = foldid,
      # nlambda = nlambda,
      # lambda.factor = lambda_factor,
      lambda = lambda,
      pf_group = pf_group,
      asparse = asparse,
      standardize = TRUE, intercept = FALSE
    )
    cvm_mx[, asid] <- sgl1$cvm
    lambda_min_ind <- which.min(sgl1$cvm)
    fit <- sgl1$sparsegl.fit
    # risk <- sparsegl::estimate_risk(fit, uw)
    # df[asid] <- fit$df[lambda_min_ind]
    coefs[, asid] <- as.numeric(fit$beta[, lambda_min_ind])
    mse[asid] <- fit$mse[lambda_min_ind]
    # lambda <- sgl1$lambda
  }

  cv_ind <- arrayInd(which.min(cvm_mx), dim(cvm_mx))
  alpha_min_ind <- cv_ind[2]
  gamma <- coefs[1:q, alpha_min_ind]
  beta <- coefs[(q + 1):nvars, alpha_min_ind]
  nnz <- sum(abs(beta) > 0) - 1
  if (nnz >= n) {
    sigma2 <- 1
  } else {
    sigma2 <- mse[alpha_min_ind] * n / abs(n - nnz)
  }

  return(list(
    gamma = gamma,
    beta = beta,
    sigma2 = sigma2,
    lambda = lambda,
    mse = mse[alpha_min_ind],
    cvm = cvm_mx,
    cv_lambda_idx = cv_ind[1],
    cv_alpha_idx = cv_ind[2]
  ))
}
