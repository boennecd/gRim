#include "arma.h"
#include <vector>
#include <algorithm>
#include <cmath> 
extern "C"
{
  #include "_utils_mat.h"
}


double cpp_LL
  (const arma::mat&, const arma::mat&, const arma::uword, const arma::uword);

inline Rcpp::List cpp_ggmfit_return
  (const arma::mat &S, arma::mat &K, const unsigned int n, 
   const unsigned int nvar, const unsigned int iter)
{
  return Rcpp::List::create(
    Rcpp::Named("logL") = cpp_LL(S, K, n, nvar),
    Rcpp::Named("K") = std::move(K),
    Rcpp::Named("iter") = iter);
}

bool conv_criteria(const arma::mat&, const arma::mat&, const double);

static const char U_char = 'U';

std::tuple<
  std::vector<arma::uvec>, std::vector<arma::uvec>, 
  size_t, size_t> get_clique_n_residual_indices
  (const unsigned int, const arma::uvec&, const arma::uvec&, const bool);

template<typename T>
arma::mat sym_mat_inv(T X){
  arma::mat Z = X; /* copy */
  int dim = Z.n_cols, info;
  
  dpotrf_wrap(&U_char, &dim, Z.memptr(), &dim, &info);
  if(info != 0L)
    Rcpp::stop("'dpotrf' failed with INFO: " + std::to_string(info));
  
  dpotri_wrap(&U_char, &dim, Z.memptr(), &dim, &info);
  if(info != 0L)
    Rcpp::stop("'dportri' failed with INFO: " + std::to_string(info));
  
  Z = arma::symmatu(Z);
  
  return Z;
}

/* S      covariance matrix 
 * n      number of observations
 * K      starting value for concentration matrix
 * nvar   number of variables
 * glen   length of each cliques
 * gg     concatenated clique indices
 * iter   maximum number of iterations
 * eps    convergence threshold
 */

//[[Rcpp::export]]
Rcpp::List cpp_ggmfit
(const arma::mat &S, const unsigned int n, arma::mat K, 
 const unsigned int nvar, const arma::uvec &glen, const arma::uvec &gg, 
 const unsigned int iter, const double eps, const unsigned int details)
{
  const std::size_t ngen = glen.size(); /* number of cliques */
  
  /* return quickly if possible */
  if(ngen == 1){
    K = sym_mat_inv(S);
    return cpp_ggmfit_return(S, K, n, nvar, 0L);
    
  }
  
  /* setup vector with clique indices */
  std::size_t max_clique_size, max_resid_size;
  std::vector<arma::uvec> clique_indices, residual_indices;
  {
    auto tmp = get_clique_n_residual_indices(nvar, glen, gg, true);
    clique_indices   = std::move(std::get<0>(tmp));
    residual_indices = std::move(std::get<1>(tmp));
    max_clique_size  = std::get<2>(tmp);
    max_resid_size   = std::get<3>(tmp);
  }
  
  /* allocate memory to use */
  std::unique_ptr<double[]> 
    K_res_res_mem(new double[max_resid_size * max_resid_size]),
    K_res_cli_mem(new double[max_resid_size * max_clique_size]);
  
  /* compute and store inverse covariance matrices for cliques. 
   * This can be done in parallel but it is not the bottleneck */
  std::vector<arma::mat> inv_covars;
  inv_covars.reserve(ngen);
  for(auto clique : clique_indices)
    inv_covars.emplace_back(sym_mat_inv(S(clique, clique)));
  
  /* run iterative proportional scaling */
  double prev_ll = 0., ll = 0.;
  if(details > 0L){
    ll = cpp_LL(S, K, n, nvar);
    Rprintf("Initial logL: %14.6f \n", ll);
  }
  unsigned int i = 0;
  arma::mat S_working = sym_mat_inv(K);
  for(; i < iter; ++i){
    Rcpp::checkUserInterrupt();
    
    const arma::mat old_K = K, S_working_old = std::move(S_working);
    prev_ll = ll;
    
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
    
    S_working = sym_mat_inv(K);
    if(conv_criteria(S_working, S_working_old, eps))
      break;
    
    if(details > 0L){
      ll = cpp_LL(S, K, n, nvar);
      Rprintf("Iteration: %3i logL: %14.6f diff logL: %20.13f\n", 
              i + 1L, ll, ll - prev_ll);
    }
  }
  
  return cpp_ggmfit_return(S, K, n, nvar, i + 1L);
}

//[[Rcpp::export]]
Rcpp::List cpp_ggmfit_wood
  (const arma::mat &S, const unsigned int n, arma::mat K, 
   const unsigned int nvar, const arma::uvec &glen, const arma::uvec &gg, 
   const unsigned int iter, const double eps, const unsigned int details)
{
  const std::size_t ngen = glen.size(); /* number of cliques */
  
  /* return quickly if possible */
  if(ngen == 1){
    K = sym_mat_inv(S);
    return cpp_ggmfit_return(S, K, n, nvar, 0L);
    
  }
  
  /* setup vector with clique indices */
  std::size_t max_clique_size;
  std::vector<arma::uvec> clique_indices;
  {
    auto tmp = get_clique_n_residual_indices(nvar, glen, gg, false);
    clique_indices  = std::move(std::get<0>(tmp));
    max_clique_size = std::move(std::get<2>(tmp));
  }
  
  /* allocate memory to use */
  std::unique_ptr<double[]> 
    S_clique_mem      (new double[max_clique_size * nvar]),
    Delta_S_clique_mem(new double[max_clique_size * nvar]);
  
  /* compute and store inverse covariance matrices for cliques. 
   * This can be done in parallel but it is not the bottleneck */
  std::vector<arma::mat> inv_covars;
  inv_covars.reserve(ngen);
  for(auto clique : clique_indices)
    inv_covars.emplace_back(sym_mat_inv(S(clique, clique)));
  
  /* run iterative proportional scaling */
  unsigned int i = 0;
  arma::mat S_working = sym_mat_inv(K);
  for(; i < iter; ++i){
    const arma::mat S_working_old = S_working;
    
    auto S_cli_cli_inv = inv_covars.begin();
    for(auto clique : clique_indices){
      /* Use Binomial inverse theorem as Delta may be singular
       * https://en.wikipedia.org/wiki/Woodbury_matrix_identity#Binomial_inverse_theorem */
      arma::mat S_clique(clique.size(), nvar);
      S_clique = S_working.rows(clique);
      arma::mat Delta = 
        *(S_cli_cli_inv++) - sym_mat_inv(S_working(clique, clique));
      
      arma::mat Delta_S_clique(clique.size(), nvar); 
      Delta_S_clique = Delta * S_clique;
      arma::mat H = Delta_S_clique.cols(clique);
      H.diag() += 1.;
      
      S_working -= S_clique.t() * arma::solve(H, Delta_S_clique);
      
    }
    
    if(conv_criteria(S_working, S_working_old, eps))
      break;
  }
  
  K = sym_mat_inv(S_working);
  return cpp_ggmfit_return(S, K, n, nvar, i + 1L);
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

bool conv_criteria(
    const arma::mat &new_mat, const arma::mat &old_mat, const double eps){
  const arma::uword dim = new_mat.n_cols; /* assume symmetrical */
  
  for(arma::uword i = 0; i < dim; ++i)
    for(arma::uword j = 0; j <= i; ++j){
      double num = std::abs(new_mat(i, j) - old_mat(i, j));
      double den = std::sqrt(
        new_mat(i, i) * new_mat(j, j) + new_mat(i, j) * new_mat(i, j));
      
      if(num/den > eps)
        return false;
    }
  
  return true;
}

std::tuple<
  std::vector<arma::uvec>, std::vector<arma::uvec>, 
  size_t, size_t> get_clique_n_residual_indices
  (const unsigned int nvar, const arma::uvec &glen, const arma::uvec &gg, 
   const bool with_residuals)
  {
    const std::size_t ngen = glen.size();
    std::size_t max_clique_size = 0L, max_resid_size = 0L;
    std::vector<arma::uvec> clique_indices;
    clique_indices.reserve(ngen);
    /* TODO: avoid O(nvar * ngen) storage and do a bit extra computation? */
    std::vector<arma::uvec> residual_indices;
    if(with_residuals)
      residual_indices.reserve(ngen);
    {
      auto gg_i = gg.begin();
      for(auto n_members : glen){
        arma::uvec clique(gg_i, n_members), residual(nvar - n_members);
        std::sort(clique.begin(), clique.end());
        
        if(with_residuals){
          auto r_i = residual.begin(), c_i = clique.begin();
          for(unsigned int i = 0; i < nvar and r_i != residual.end(); ++i){
            if(c_i != clique.end() and i == *c_i){
              ++c_i;
              continue;
            }
            
            *(r_i++) = i;
            
          }
        }
        
        if(clique.size() > max_clique_size)
          max_clique_size = clique.size();
        if(with_residuals and residual.size() > max_resid_size)
          max_resid_size = residual.size();
        
        clique_indices.emplace_back(std::move(clique));
        if(with_residuals)
          residual_indices.emplace_back(std::move(residual));
        gg_i += n_members;
      }
    }
    
    return std::make_tuple(
      std::move(clique_indices), std::move(residual_indices), 
      max_clique_size, max_resid_size);
  }

