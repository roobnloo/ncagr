cv_ncagr_node <- function(y, responses, covariates, sglmixpath,
                          lambdapath, gmixpath, wl1, wl2,
                          maxit, tol, nfolds,
                          verbose = FALSE) {
  p <- ncol(responses) + 1
  q <- ncol(covariates)
  n <- nrow(responses)
  nlambda <- length(lambdapath)
  nsglmix <- length(sglmixpath)
  ngmix <- length(gmixpath)

  intx <- intxmx(responses, covariates)
  foldsplit <- split(
    seq_len(n), cut(sample(seq_len(n)), nfolds, labels = FALSE)
  )
  mses <- matrix(nrow = nfolds, ncol = nlambda * nsglmix * ngmix)
  for (i in seq_len(nfolds)) {
    testids <- foldsplit[[i]]
    y_train <- y[-testids]
    covariates_train <- covariates[-testids, ]
    intx_train <- intx[-testids, ]

    y_mean <- mean(y_train)
    cov_means <- Matrix::colMeans(covariates_train)
    cov_sds <- apply(covariates_train, 2, stats::sd)
    intx_means <- Matrix::colMeans(intx_train)
    intx_sds <- apply(intx_train, 2, stats::sd)

    ntest <- length(testids)
    y_test <- y[testids]
    y_test <- matrix(rep(y_test, times = nlambda * nsglmix * ngmix),
      nrow = ntest, ncol = nlambda * nsglmix * ngmix
    )
    covariates_test <- covariates[testids, ]
    intx_test <- intx[testids, ]

    y_train <- y_train - y_mean
    covariates_train <- scale(covariates_train)
    intx_train <- scale(intx_train)
    nodereg <- NodewiseRegression(
      y_train, covariates_train, intx_train, gmixpath, sglmixpath, wl1, wl2,
      lambdaPath = lambdapath, maxit = maxit, tol = tol
    )

    gamma <- nodereg["gamma"][[1]]
    dim(gamma) <- c(q, nlambda * nsglmix * ngmix)
    gamma <- gamma / cov_sds
    beta <- nodereg["beta"][[1]]
    dim(beta) <- c((p - 1) * (q + 1), nlambda * nsglmix * ngmix)
    beta <- beta / intx_sds
    y_fitted <- covariates_test %*% gamma + intx_test %*% beta
    y_fitted <- y_fitted + y_mean
    y_fitted <- sweep(y_fitted, 2, Matrix::colSums(gamma * cov_means), "-")
    y_fitted <- sweep(y_fitted, 2, Matrix::colSums(beta * intx_means), "-")
    resid_test <- y_test - y_fitted

    # resid_test <- y_test - covariates_test %*% gamma - intx_test %*% beta
    # resid_test <- resid_test + y_mean
    # resid_test <- sweep(resid_test, 2, Matrix::colSums(gamma * cov_means), "-")
    # resid_test <- sweep(resid_test, 2, Matrix::colSums(beta * intx_means), "-")
    mses[i, ] <- apply(resid_test, 2, \(x) mean(sum(x^2)) / ntest)
    if (verbose) {
      message("Finished CV fold ", i)
    }
  }

  dim(mses) <- c(nfolds, nlambda, nsglmix, ngmix)
  cv_mse <- apply(mses, c(2, 3, 4), mean)
  return(cv_mse)
}
