#include <RcppArmadillo.h>
#include <vector>
#include <algorithm>

extern "C"
{
  #include "_utils_mat.h"
}


double cpp_LL
  (const arma::mat&, const arma::mat&, const arma::uword, const arma::uword);

/* S      covariance matrix 
 * n      number of observations
 * K      starting value for concentration matrix
 * nvar   number of variables
 * glen   length of each cliques
 * gg     concatenated clique indices
 * iter   maximum number of iterations
 * eps    convergence threshold
 */

static const char U_char = 'U';

//[[Rcpp::export]]
Rcpp::List cpp_ggmfit
(const arma::mat &S, const unsigned int n, arma::mat K, 
 const unsigned int nvar, const arma::uvec &glen, const arma::uvec &gg, 
 const unsigned int iter, const double eps)
{
  /* setup vector with clique indices */
  std::size_t 
    ngen = gg.size(), /* number of cliques */
    max_clique_size = 0L, 
    max_resid_size  = 0L;
  std::vector<arma::uvec> clique_indices;
  clique_indices.reserve(ngen);
  /* TODO: avoid O(nvar * ngen) storage and do a bit extra computation? */
  std::vector<arma::uvec> residual_indices;
  residual_indices.reserve(ngen);
  {
    auto gg_i = gg.begin();
    for(auto n_members : glen){
      arma::uvec clique(gg_i, n_members), residual(nvar - n_members);
      std::sort(clique.begin(), clique.end());
      
      auto r_i = residual.begin(), c_i = clique.begin();
      for(unsigned int i = 0; i < nvar and r_i != residual.end(); ++i){
        if(c_i != clique.end() and i == *c_i){
          ++c_i;
          continue;
        }
        
        *(r_i++) = i;
        
      }
      
      if(clique.size() > max_clique_size)
        max_clique_size = clique.size();
      if(residual.size() > max_resid_size)
        max_resid_size = residual.size();
      
      clique_indices.emplace_back(std::move(clique));
      residual_indices.emplace_back(std::move(residual));
      gg_i += n_members;
    }
  }
  
  /* allocate memory to use */
  std::unique_ptr<double[]> 
    K_res_res_mem(new double[max_resid_size * max_resid_size]),
    K_res_cli_mem(new double[max_resid_size * max_clique_size]);
  
  /* compute and store inverse covariance matrices for cliques. 
   * This can be done in parallel but it is not the bottleneck */
  std::vector<arma::mat> inv_covars;
  inv_covars.reserve(ngen);
  for(auto clique : clique_indices){
    arma::mat tmp = S(clique, clique);
    int dim = tmp.n_cols, info;
    
    dpotrf_wrap(&U_char, &dim, tmp.memptr(), &dim, &info);
    if(info != 0L)
      Rcpp::stop("'dpotrf' failed with INFO: " + std::to_string(info));
    
    dpotri_wrap(&U_char, &dim, tmp.memptr(), &dim, &info);
    if(info != 0L)
      Rcpp::stop("'dportri' failed with INFO: " + std::to_string(info));
    inv_covars.emplace_back(arma::symmatu(tmp));
  }
  
  /* run iterative proportional scaling */
  if(ngen == 1)
    Rcpp::stop("Not implemented with ngen == 1");
  unsigned int i = 0;
  for(; i < iter; ++i){
    Rcpp::checkUserInterrupt();
    const arma::mat old_K = K;
    auto resiaul_it = residual_indices.begin();
    auto S_cli_cli_inv = inv_covars.begin();
    
    for(auto clique : clique_indices){
      const arma::uvec &residuals = *(resiaul_it++);
      size_t n_res = residuals.size();
      arma::mat 
        K_res_cli(K_res_cli_mem.get(), n_res, clique.size(), false), 
        K_res_res(K_res_res_mem.get(), n_res, n_res        , false);
      K_res_cli = K(residuals, clique);
      K_res_res = K(residuals, residuals);
      
      int n_res_i = n_res, nrhs = clique.size(), info;
      dposv_wrap(
        &U_char, &n_res_i, &nrhs, K_res_res.memptr(), 
        &n_res_i, K_res_cli.memptr(), &n_res_i, &info);
      if(info != 0L)
        Rcpp::stop("'dposv' failed with INFO: " + std::to_string(info));
      
      K(clique, clique) =  *(S_cli_cli_inv++) +
        K(clique, residuals) * K_res_cli;
    }
    
    /* TODO: change convergence criteria to match w/ C version? */
    if(arma::norm(K - old_K) / arma::norm(K) < eps)
      break;
  }
  
  return Rcpp::List::create(
    Rcpp::Named("logL") = cpp_LL(S, K, n, nvar), /* TODO: compute */
    Rcpp::Named("K") = std::move(K),
    Rcpp::Named("iter") = i + 1L);
}

double cpp_LL
  (const arma::mat &S, const arma::mat &K, const arma::uword nobs, 
   const arma::uword nvar)
{
  double l_det, l_sign, trace = 0.;
  arma::log_det(l_det, l_sign, K);
  auto *k = K.begin();
  for(auto s : S)
    trace += s * *(k++);
  
  return - (double)nobs * nvar * log(2. * M_PI) / 2. + nobs / 2. * (l_det - trace);
}
