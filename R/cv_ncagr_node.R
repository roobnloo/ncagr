cv_ncagr_node <- function(y, responses, covariates, sglmixpath,
                          lambdapath, gmixpath,
                          maxit, tol, nfolds,
                          verbose = FALSE) {
  p <- ncol(responses) + 1
  q <- ncol(covariates)
  n <- nrow(responses)
  nlambda <- length(lambdapath)
  nsglmix <- length(sglmixpath)
  ngmix <- length(gmixpath)

  intx <- cbind(responses, intxmx(responses, covariates))
  foldsplit <- split(
    seq_len(n), cut(sample(seq_len(n)), nfolds, labels = FALSE))
  mses <- matrix(nrow = nfolds, ncol = nlambda * nsglmix * ngmix)
  for (i in seq_len(nfolds)) {
    testids <- foldsplit[[i]]
    y_train <- y[-testids]
    responses_train <- responses[-testids, ]
    covariates_train <- covariates[-testids, ]

    ntest <- length(testids)
    y_test <- y[testids]
    y_test <- matrix(
      rep(y_test, times = nlambda * nsglmix * ngmix),
      nrow = ntest, ncol = nlambda * nsglmix * ngmix)
    intx_test <- intx[testids, ]
    covariates_test <- covariates[testids, ]

    nodereg <- NodewiseRegression(
      y_train, responses_train, covariates_train, gmixpath, sglmixpath,
      lambdaPath = lambdapath, maxit = maxit, tol = tol)

    gamma <- nodereg["gamma"][[1]]
    dim(gamma) <- c(q, nlambda * nsglmix * ngmix)
    beta <- nodereg["beta"][[1]]
    dim(beta) <- c((p - 1) * (q + 1), nlambda * nsglmix * ngmix)
    resid_test <- y_test - covariates_test %*% gamma - intx_test %*% beta
    mses[i, ] <- apply(resid_test, 2, \(x) sum(x^2) / (ntest - 1))
    if (verbose)
      message("Finished CV fold ", i)
  }

  dim(mses) <- c(nfolds, nlambda, nsglmix, ngmix)
  cv_mse <- apply(mses, c(2, 3, 4), mean)
  return(cv_mse)
}
