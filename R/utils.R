#' @return n x q(p-1) matrix representing interactions btw responses and covs
#' @noRd
intxmx <- function(responses, covariates) {
  p <- ncol(responses) + 1
  q <- ncol(covariates)
  idx_mat <- as.matrix(expand.grid(seq_len(p - 1), seq_len(q)))

  foo <- function(i) {
    responses[, idx_mat[i, 1]] * covariates[, idx_mat[i, 2]]
  }

  result <- Reduce(
    cbind,
    Map(foo, seq_len(nrow(idx_mat)))
  )
  return(result)
}

#' @return symmetrized version of matrix mx
#' result_ij = result_ji is nonzero iff both mx_ij and mx_ji are nonzero,
#' in which case we choose the smaller value in magnitude.
#' @noRd
symmetrize <- function(mx, rule = "and") {
  if (rule == "and") {
    result <- mx * (abs(mx) < t(abs(mx))) + t(mx) * (t(abs(mx)) < abs(mx))
  } else {
    result <- mx * (abs(mx) >= t(abs(mx))) + t(mx) * (t(abs(mx)) >= abs(mx))
  }
  return(result)
}
