cv_ncagr_node <- function(y_centered, intx_scale_node, sglmixpath,
                          lambdapath, gmixpath, wl1, wl2,
                          maxit, tol, nfolds,
                          verbose = FALSE) {
  q <- length(wl2)
  p <- (ncol(intx_scale_node) - q) / (q + 1) + 1
  n <- length(y_centered)
  nvars <- q + (p - 1) * (q + 1)
  # nlambda <- length(lambdapath)
  # nsglmix <- length(sglmixpath)
  # ngmix <- length(gmixpath)

  foldid <- cut(sample(seq_len(n)), nfolds, labels = FALSE)
  groupid <- c(rep(0, q), rep(1:(q + 1), each = p - 1)) + 1
  cvm_mx <- matrix(0, nrow = length(lambdapath), ncol = length(gmixpath))
  coefs <- matrix(nrow = nvars, ncol = length(gmixpath))
  rss <- numeric(length(gmixpath))

  for (agid in seq_along(gmixpath)) {
    alpha_g <- gmixpath[agid]
    pf_sparse <- c(rep(alpha_g, q), rep(1 - alpha_g, (p - 1) * (q + 1))) * wl1
    anorm <- sum(pf_sparse) / nvars
    pf_group <- c(0, 0, rep((1 - alpha_g) / anorm, q) * wl2)
    sgl1 <- sparsegl::cv.sparsegl(
      intx_scale_node, y_centered,
      group = groupid,
      foldid = foldid,
      lambda = anorm * lambdapath,
      pf_group = pf_group, pf_sparse = pf_sparse,
      asparse = sglmixpath, standardize = FALSE, intercept = FALSE
    )
    lambda_min_ind <- which.min(sgl1$cvm)
    fit <- sgl1$sparsegl.fit
    coefs[, agid] <- as.numeric(fit$beta[, lambda_min_ind])
    rss[agid] <- fit$mse[lambda_min_ind] * n
    cvm_mx[, agid] <- sgl1$cvm
  }

  cv_ind <- arrayInd(which.min(cvm_mx), dim(cvm_mx))
  gamma <- coefs[1:q, cv_ind[2]]
  beta <- coefs[(q + 1):nvars, cv_ind[2]]

  return(list(
    gamma = gamma,
    beta = beta,
    rss = rss[cv_ind[2]],
    cvm = cvm_mx,
    cv_lambda_idx = cv_ind[1],
    cv_gmix_idx = cv_ind[2]
  ))
}
