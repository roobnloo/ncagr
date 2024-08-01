#' Run Natural Covariate-adjusted Graphical Regression
#'
#' @param responses \eqn{n \times p} matrix of responses
#' @param covariates \eqn{n \times q} matrix of covariates
#' @param gmixpath A path of ncagr mixing parameters. Default is 0.1, 0.2, ..., 0.9.
#' @param sglmixpath A path of sparse-group lasso mixing parameter with \eqn{0\leq \alpha \leq 1}. Default is 0.75.
#' @param nlambda The number of lambda values to use for cross-validation - default is 100.
#' @param lambdafactor The smallest value of lambda as a fraction of the maximum lambda.
#' @param maxit The maximum number of iterations. Default is \eqn{3\times 10^6}.
#' @param tol The convergence threshhold for optimization. Default is \eqn{10^{-6}}.
#' @param nfolds Number of folds for cross-validation. Default is 5.
#' @param verbose If TRUE, prints progress messages. Default is TRUE.
#' @param ncores Runs the nodewise regressions in parallel using that many cores. Default is 1.
#' @param adaptive Use adaptive weights when fitting nodewise regressions. Default is FALSE.
#' @useDynLib ncagr
#' @importFrom Rcpp sourceCpp
#' @importFrom Matrix colMeans colSums
#' @importFrom stats sd
#' @import parallel
#' @export
ncagr <- function(responses, covariates, gmixpath = seq(0, 1, by = 0.1),
                  sglmixpath = 0.75, nlambda = 100,
                  lambdafactor = 1e-6, maxit = 3e6, tol = 1e-6, nfolds = 5,
                  verbose = TRUE, ncores = 1, adaptive = FALSE) {
  stopifnot(
    is.matrix(responses), is.matrix(covariates),
    nrow(responses) == nrow(covariates),
    all(gmixpath >= 0), all(gmixpath <= 1),
    all(sglmixpath >= 0), all(sglmixpath <= 1)
  )

  p <- ncol(responses)
  q <- ncol(covariates)
  n <- nrow(responses)
  bveclength <- (p - 1) * (q + 1)

  # Initial run to get lambdas for each response
  ngmix <- length(gmixpath)
  nsglmix <- length(sglmixpath)
  lambdas <- matrix(nrow = nlambda, ncol = p)
  beta <- array(dim = c(bveclength, nlambda, nsglmix, ngmix, p))
  gamma <- array(dim = c(q, nlambda, nsglmix, ngmix, p))
  varhat <- array(dim = c(nlambda, nsglmix, ngmix, p))
  resid <- array(dim = c(n, nlambda, nsglmix, ngmix, p))
  objval <- array(dim = c(nlambda, nsglmix, ngmix, p))
  l1_weights <- matrix(nrow = q + bveclength, ncol = p)
  l2_weights <- matrix(nrow = q, ncol = p)

  cov_sds <- apply(covariates, 2, stats::sd)
  cov_scale <- scale(covariates)
  intx <- intxmx(responses, covariates)
  intx_sds <- apply(intx, 2, sd)
  intx_scale <- scale(intx)

  nodewise <- function(node, adaptive = FALSE) {
    y <- responses[, node] - mean(responses[, node])
    intx_scale_node <- intx_scale[, -(seq(0, q) * p + node)]

    wl1 <- c(rep(1, q), rep(1 / (p - 1), bveclength))
    wl1 <- (q + bveclength) * wl1 / sum(wl1)
    wl2 <- rep(1, q)
    if (adaptive) {
      fit_ridge <- glmnet::glmnet(
        cbind(cov_scale, intx_scale_node), y,
        alpha = 0, standardize = FALSE, intercept = FALSE
      )
      co <- stats::coef(fit_ridge, s = fit_ridge$lambda[100], exact = TRUE)@x
      wl1 <- 1 / abs(co)
      wl1[wl1 > 2000] <- 2000
      # indices of U, X, without interactions
      ux_id <- 1:(q + p - 1)
      wgroups <- co[-ux_id]
      # Compute weights for each group
      wl2 <- 1 / sapply(split(wgroups, rep(seq_len(q), each = p - 1)), \(x) sqrt(sum(x^2)))
      wl2[wl2 > 2000] <- 2000
    }

    nodereg <- NodewiseRegression(
      y, cov_scale, intx_scale_node, gmixpath, sglmixpath,
      wl1 = wl1, wl2 = wl2,
      nlambda = nlambda, lambdaFactor = lambdafactor,
      maxit = maxit, tol = tol
    )
    if (verbose) {
      message(paste(node, " "), appendLF = FALSE)
    }
    return(list(
      lambdas = nodereg$lambdapath,
      beta = nodereg$beta / intx_sds[-(seq(0, q) * p + node)],
      gamma = nodereg$gamma / cov_sds,
      varhat = nodereg$varhat,
      resid = nodereg$resid,
      objval = nodereg$objval,
      wl1 = wl1,
      wl2 = wl2
    ))
  }

  message("Begin initial run...")

  if (ncores > 1) {
    reg_result <- parallel::mclapply(seq_len(p), nodewise, mc.cores = ncores)
  } else {
    reg_result <- lapply(seq_len(p), nodewise)
  }

  for (node in seq_len(p)) {
    lambdas[, node] <- reg_result[[node]]$lambdas
    beta[, , , , node] <- reg_result[[node]]$beta
    gamma[, , , , node] <- reg_result[[node]]$gamma
    varhat[, , , node] <- reg_result[[node]]$varhat
    resid[, , , , node] <- reg_result[[node]]$resid
    objval[, , , node] <- reg_result[[node]]$objval
    l1_weights[, node] <- reg_result[[node]]$wl1
    l2_weights[, node] <- reg_result[[node]]$wl2
  }

  message("\nFinished initial run")
  message("Begin cross-validation...")

  cv_lambda_idx <- vector(length = p)
  cv_sglmix_idx <- vector(length = p)
  cv_gmix_idx <- vector(length = p)
  cv_mse <- array(dim = c(p, nlambda, nsglmix, ngmix))

  cv_node <- function(node) {
    cv_result <- cv_ncagr_node(
      responses[, node], responses[, -node], covariates, sglmixpath,
      lambdas[, node], gmixpath, l1_weights[, node], l2_weights[, node],
      maxit, tol, nfolds
    )
    message(paste(node, " "), appendLF = FALSE)
    return(cv_result)
  }

  if (ncores > 1) {
    cv_results <- parallel::mclapply(seq_len(p), cv_node, mc.cores = ncores)
  } else {
    cv_results <- lapply(seq_len(p), cv_node)
  }

  for (node in seq_len(p)) {
    cv_mse[node, , , ] <- cv_results[[node]]
    minind <- arrayInd(which.min(cv_results[[node]]), dim(cv_results[[node]]))
    cv_lambda_idx[node] <- minind[1]
    cv_sglmix_idx[node] <- minind[2]
    cv_gmix_idx[node] <- minind[3]
  }

  message("\nFinished cross validating all nodes")

  varhat <- varhat[cbind(cv_lambda_idx, cv_sglmix_idx, cv_gmix_idx, seq(p))]

  if (adaptive) {
    message("Beginning adaptive run")
    if (ncores > 1) {
      reg_result <- parallel::mclapply(seq_len(p), nodewise, adaptive = TRUE, mc.cores = ncores)
    } else {
      reg_result <- lapply(seq_len(p), nodewise, adaptive = TRUE)
    }
    message("\nFinished initial adaptive run")

    for (node in seq_len(p)) {
      lambdas[, node] <- reg_result[[node]]$lambdas
      beta[, , , , node] <- reg_result[[node]]$beta
      gamma[, , , , node] <- reg_result[[node]]$gamma
      resid[, , , , node] <- reg_result[[node]]$resid
      objval[, , , node] <- reg_result[[node]]$objval
      l1_weights[, node] <- reg_result[[node]]$wl1
      l2_weights[, node] <- reg_result[[node]]$wl2
    }

    message("Begin cross-validation...")

    if (ncores > 1) {
      cv_results <- parallel::mclapply(seq_len(p), cv_node, mc.cores = ncores)
    } else {
      cv_results <- lapply(seq_len(p), cv_node)
    }

    message("\nFinished adaptive run")

    for (node in seq_len(p)) {
      cv_mse[node, , , ] <- cv_results[[node]]
      minind <- arrayInd(which.min(cv_results[[node]]), dim(cv_results[[node]]))
      cv_lambda_idx[node] <- minind[1]
      cv_sglmix_idx[node] <- minind[2]
      cv_gmix_idx[node] <- minind[3]
    }
  }

  ghat_select <- cbind(
    rep(seq(q), times = p),
    rep(cv_lambda_idx, each = q),
    rep(cv_sglmix_idx, each = q),
    rep(cv_gmix_idx, each = q),
    rep(seq(p), each = q)
  )
  ghat_mx <- t(matrix(gamma[ghat_select], nrow = q, ncol = p))

  bhat_select <- cbind(
    rep(seq(bveclength), times = p),
    rep(cv_lambda_idx, each = bveclength),
    rep(cv_sglmix_idx, each = bveclength),
    rep(cv_gmix_idx, each = bveclength),
    rep(seq(p), each = bveclength)
  )
  bhat_mx <- matrix(beta[bhat_select], nrow = bveclength, ncol = p)
  bhat_tens <- array(0, dim = c(p, p, q + 1))
  for (i in seq_len(p)) {
    bhat_tens[i, -i, ] <- bhat_mx[, i]
  }

  bhat_symm <- array(0, dim = c(p, p, q + 1))
  for (h in seq_len(q + 1)) {
    bhat_symm[, , h] <- symmetrize(-diag(1 / varhat) %*% bhat_tens[, , h])
  }

  outlist <- list(
    gamma = ghat_mx,
    beta = bhat_symm,
    beta_raw = bhat_tens,
    sigma2 = varhat,
    lambdapath = lambdas,
    sglmixpath = sglmixpath,
    gmixpath = gmixpath,
    cv_lambda_idx = cv_lambda_idx,
    cv_sglmix_idx = cv_sglmix_idx,
    cv_gmix_idx = cv_gmix_idx,
    cv_mse = cv_mse
  )
  class(outlist) <- "ncagr"

  return(outlist)
}
