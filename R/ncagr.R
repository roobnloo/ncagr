#' Run Natural Covariate-adjusted Graphical Regression
#'
#' @param responses \eqn{n x p} matrix of responses
#' @param covariates \eqn{n x q} matrix of covariates
#' @param asparse The sparse-group lasso mixing parameter with \eqn{0\leq \alpha \leq 1}
#' @param nregmean The number of eta values - default is 10
#' @param nlambda The number of lambda values - default is 100
#' @param lambdafactor The smallest value of lambda as a fraction of the maximum lambda.
#' @param maxit The maximum number of iterations. Default is \eqn{3\times 10^6}
#' @param tol The convergence threshhold for optimization. Default is \eqn{10^{-8}}
#' @param nfolds Number of folds for cross-validation. Default is 5.
#' @param verbose If TRUE, prints progress messages. Default is FALSE.
#' @param ncores Runs the nodewise regressions in parallel using that many cores. Default is 1.
#' @useDynLib ncagr
#' @importFrom Rcpp sourceCpp
#' @importFrom abind abind
#' @import parallel
#' @export
ncagr <- function(responses, covariates, asparse,
                 nregmean = 10, nlambda = 100, lambdafactor = 1e-4,
                 maxit = 3e6, tol = 1e-8, nfolds = 5,
                 verbose = FALSE, ncores = 1) {

  stopifnot(is.matrix(responses), is.matrix(covariates),
            nrow(responses) == nrow(covariates),
            nregmean > 0)

  p <- ncol(responses)
  q <- ncol(covariates)
  n <- nrow(responses)
  bveclength <- (p - 1) * (q + 1)

  # Initial run to get lambdas for each response
  lambdas <- matrix(nrow = nlambda, ncol = p)
  regmeanpaths <- matrix(nrow = nregmean, ncol = p)
  beta <- array(dim = c(bveclength, nlambda, nregmean, p))
  gamma <- array(dim = c(q, nlambda, nregmean, p))
  varhat <- array(dim = c(nlambda, nregmean, p))
  resid <- array(dim = c(n, nlambda, nregmean, p))
  objval <- array(dim = c(nlambda, nregmean, p))

  nodewise <- function(node) {
    # if (verbose)
    #   print(paste("Starting initial run for node", node))
    y <- responses[, node] - mean(responses[, node])
    nodereg <- NodewiseRegression(
      y, responses[, -node], covariates, asparse,
      nregmean = nregmean, nlambda = nlambda, lambdaFactor = lambdafactor,
      maxit = maxit, tol = tol)
    if (verbose)
      message("Finished initial run for node ", node)
    return(list(
      lambdas = nodereg["lambdas"][[1]],
      regmeanpath = nodereg["regmeans"][[1]],
      beta = nodereg["beta"][[1]],
      gamma = nodereg["gamma"][[1]],
      varhat = nodereg["varhat"][[1]],
      resid = nodereg["resid"][[1]],
      objval = nodereg["objval"][[1]]
    ))
  }

  if (verbose) {
    message("Begin initial run...\n")
  }
  if (ncores > 1) {
    reg_result <- parallel::mclapply(seq_len(p), nodewise, mc.cores = ncores)
  } else {
    reg_result <- lapply(seq_len(p), nodewise)
  }

  for (node in seq_len(p)) {
    lambdas[, node] <- reg_result[[node]]$lambdas
    regmeanpaths[, node] <- reg_result[[node]]$regmeanpath
    beta[, , , node] <- reg_result[[node]]$beta
    gamma[, , , node] <- reg_result[[node]]$gamma
    varhat[, , node] <- reg_result[[node]]$varhat
    resid[, , , node] <- reg_result[[node]]$resid
    objval[, , node] <- reg_result[[node]]$objval
  }

  if (verbose) {
    message("Finished initial run")
    message("Begin cross-validation...")
  }

  cv_lambda_idx <- vector(length = p)
  cv_regmean_idx <- vector(length = p)
  cv_mse <- array(dim = c(p, nlambda, nregmean))

  cv_node <- function(node) {
    y <- responses[, node] - mean(responses[, node])
    cv_result <- cv_ncagr_node(y, responses[, -node], covariates, asparse,
                               lambdas[, node], regmeanpaths[, node],
                               maxit, tol, nfolds)
    if (verbose)
      message("Done cross-validating node ", node)
    return(cv_result)
  }

  if (ncores > 1) {
    cv_results <- parallel::mclapply(seq_len(p), cv_node, mc.cores = ncores)
  } else {
    cv_results <- lapply(seq_len(p), cv_node)
  }

  # if (any(sapply(cv_results, inherits, what = "try-error"))) {
  #   browser()
  # }

  for (node in seq_len(p)) {
    cv_mse[node, , ] <- cv_results[[node]]
    minind <- arrayInd(which.min(cv_results[[node]]), dim(cv_results[[node]]))
    cv_lambda_idx[node] <- minind[1]
    cv_regmean_idx[node] <- minind[2]
  }

  if (verbose) {
    message("Finished cross validating all nodes")
  }

  ghat_select <- cbind(rep(seq(q), times = p),
                       rep(cv_lambda_idx, each = q),
                       rep(cv_regmean_idx, each = q),
                       rep(seq(p), each = q))
  ghat_mx <- t(matrix(gamma[ghat_select], nrow = q, ncol = p))

  varhat <- varhat[cbind(cv_lambda_idx, cv_regmean_idx, seq(p))]

  bhat_select <- cbind(rep(seq(bveclength), times = p),
                       rep(cv_lambda_idx, each = bveclength),
                       rep(cv_regmean_idx, each = bveclength),
                       rep(seq(p), each = bveclength))
  bhat_mx <- matrix(beta[bhat_select], nrow = bveclength, ncol = p)
  bhat_tens <-  array(0, dim = c(p, p, q + 1))
  for (i in seq_len(p)) {
    bhat_tens[i, -i, ] <- bhat_mx[, i]
  }

  bhat_symm <- abind::abind(
    apply(bhat_tens, 3, symmetrize, simplify = FALSE), along = 3)

  outlist <- list(gamma = ghat_mx,
                  beta = bhat_symm,
                  sigma2 = varhat,
                  lambda = lambdas,
                  regmean = regmeanpaths,
                  cv_lambda_idx = cv_lambda_idx,
                  cv_regmean_idx = cv_regmean_idx)
  class(outlist) <- "ncagr"

  return(outlist)
}