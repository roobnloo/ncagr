cv_ncagr_node <- function(y, uw, p, q, nlambda, lambda_factor, gmixpath, sglmixpath,
                          maxit, tol, nfolds, adaptive, sigma2 = NULL) {
  n <- length(y)
  nvars <- q + (p - 1) * (q + 1)
  lam_max <- max(abs(t(uw) %*% y))
  lambda <- lam_max * exp(seq(log(1), log(lambda_factor), length = nlambda))

  # nlambda <- length(lambdapath)
  # nsglmix <- length(sglmixpath)
  # ngmix <- length(gmixpath)

  foldid <- cut(sample(seq_len(n)), nfolds, labels = FALSE)
  groupid <- c(rep(0, q), rep(1:(q + 1), each = p - 1)) + 1
  cvm_mx <- matrix(0, nrow = length(lambda), ncol = length(gmixpath))
  coefs <- matrix(nrow = nvars, ncol = length(gmixpath))
  mse <- numeric(length(gmixpath))

  wl1 <- rep(1, nvars)
  wl2 <- rep(1, q)
  if (adaptive) {
    fit_ridge <- glmnet::cv.glmnet(
      uw, y,
      alpha = 0, standardize = FALSE, intercept = FALSE
    )
    co <- stats::coef(fit_ridge, s = fit_ridge$lambda.min)@x
    wl1 <- 1 / abs(co)
    # indices of U, X, without interactions
    ux_id <- 1:(q + p - 1)
    wgroups <- co[-ux_id]
    # Compute weights for each group
    wl2 <- 1 / sapply(split(wgroups, rep(seq_len(q), each = p - 1)), \(x) sqrt(sum(x^2)))
  }

  for (agid in seq_along(gmixpath)) {
    alpha_g <- gmixpath[agid]
    pf_sparse <- c(rep(alpha_g, q), rep(1 - alpha_g, (p - 1) * (q + 1))) * wl1
    anorm <- sum(pf_sparse) / nvars
    pf_group <- c(0, 0, rep((1 - alpha_g) / anorm, q) * wl2)
    sgl1 <- sparsegl::cv.sparsegl(
      uw, y,
      group = groupid,
      foldid = foldid,
      lambda = anorm * lambda,
      pf_group = pf_group, pf_sparse = pf_sparse,
      asparse = sglmixpath, standardize = FALSE, intercept = FALSE
    )
    lambda_min_ind <- which.min(sgl1$cvm)
    fit <- sgl1$sparsegl.fit
    coefs[, agid] <- as.numeric(fit$beta[, lambda_min_ind])
    mse[agid] <- fit$mse[lambda_min_ind]
    cvm_mx[, agid] <- sgl1$cvm
  }

  cv_ind <- arrayInd(which.min(cvm_mx), dim(cvm_mx))
  gamma <- coefs[1:q, cv_ind[2]]
  beta <- coefs[(q + 1):nvars, cv_ind[2]]
  nnz <- sum(abs(beta) > 1e-10)

  if (is.null(sigma2)) {
    sigma2 <- mse[cv_ind[2]] * n / nnz
  }

  return(list(
    gamma = gamma,
    beta = beta,
    sigma2 = sigma2,
    mse = mse[cv_ind[2]],
    lambda = lambda,
    cvm = cvm_mx,
    cv_lambda_idx = cv_ind[1],
    cv_gmix_idx = cv_ind[2],
    wl1 = wl1,
    wl2 = wl2
  ))
}
