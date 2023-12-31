cv_ncagr_node <- function(y, responses, covariates, asparse,
                          lambdapath, regmeanpath,
                          maxit, tol, nfolds,
                          verbose = FALSE) {
  p <- ncol(responses) + 1
  q <- ncol(covariates)
  n <- nrow(responses)
  nlambda <- length(lambdapath)
  nregmean <- length(regmeanpath)

  intx <- cbind(responses, intxmx(responses, covariates))
  foldsplit <- split(
    seq_len(n), cut(sample(seq_len(n)), nfolds, labels = FALSE))
  mses <- matrix(nrow = nfolds, ncol = nlambda * nregmean)
  for (i in seq_len(nfolds)) {
    testids <- foldsplit[[i]]
    y_train <- y[-testids]
    responses_train <- responses[-testids, ]
    covariates_train <- covariates[-testids, ]

    ntest <- length(testids)
    y_test <- y[testids]
    y_test <- matrix(
      rep(y_test, times = nlambda * nregmean),
      nrow = ntest, ncol = nlambda * nregmean)
    intx_test <- intx[testids, ]
    covariates_test <- covariates[testids, ]

    nodereg <- NodewiseRegression(
      y_train, responses_train, covariates_train, asparse,
      regmeanPath = regmeanpath, lambdaPath = lambdapath,
      maxit = maxit, tol = tol)

    gamma <- nodereg["gamma"][[1]]
    dim(gamma) <- c(q, nlambda * nregmean)
    beta <- nodereg["beta"][[1]]
    dim(beta) <- c((p - 1) * (q + 1), nlambda * nregmean)
    resid_test <- y_test - covariates_test %*% gamma - intx_test %*% beta
    mses[i, ] <- apply(resid_test, 2, \(x) sum(x^2) / (ntest - 1))
    if (verbose)
      message("Finished CV fold ", i)
  }

  dim(mses) <- c(nfolds, nlambda, nregmean)
  cv_mse <- apply(mses, c(2, 3), mean)
  return(cv_mse)
}
