// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// cpp_ggmfit
Rcpp::List cpp_ggmfit(const arma::mat& S, const unsigned int n, arma::mat K, const unsigned int nvar, const arma::uvec& glen, const arma::uvec& gg, const unsigned int iter, const double eps);
RcppExport SEXP _gRim_cpp_ggmfit(SEXP SSEXP, SEXP nSEXP, SEXP KSEXP, SEXP nvarSEXP, SEXP glenSEXP, SEXP ggSEXP, SEXP iterSEXP, SEXP epsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type S(SSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type n(nSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type K(KSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type nvar(nvarSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type glen(glenSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type gg(ggSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type iter(iterSEXP);
    Rcpp::traits::input_parameter< const double >::type eps(epsSEXP);
    rcpp_result_gen = Rcpp::wrap(cpp_ggmfit(S, n, K, nvar, glen, gg, iter, eps));
    return rcpp_result_gen;
END_RCPP
}
// ghk2pmsParms_
List ghk2pmsParms_(List parms);
RcppExport SEXP _gRim_ghk2pmsParms_(SEXP parmsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type parms(parmsSEXP);
    rcpp_result_gen = Rcpp::wrap(ghk2pmsParms_(parms));
    return rcpp_result_gen;
END_RCPP
}
// pms2ghkParms_
List pms2ghkParms_(List parms);
RcppExport SEXP _gRim_pms2ghkParms_(SEXP parmsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type parms(parmsSEXP);
    rcpp_result_gen = Rcpp::wrap(pms2ghkParms_(parms));
    return rcpp_result_gen;
END_RCPP
}
// updateA
arma::mat updateA(arma::mat A, arma::mat E, arma::uvec row, arma::uvec col);
RcppExport SEXP _gRim_updateA(SEXP ASEXP, SEXP ESEXP, SEXP rowSEXP, SEXP colSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type A(ASEXP);
    Rcpp::traits::input_parameter< arma::mat >::type E(ESEXP);
    Rcpp::traits::input_parameter< arma::uvec >::type row(rowSEXP);
    Rcpp::traits::input_parameter< arma::uvec >::type col(colSEXP);
    rcpp_result_gen = Rcpp::wrap(updateA(A, E, row, col));
    return rcpp_result_gen;
END_RCPP
}
// update_ghkParms_
List update_ghkParms_(List Cparms, IntegerVector dgen_idx, IntegerVector cgen_idx, List ghk_obs, List pms_obs, List ghk_fit, List pms_fit, double scale, double details);
RcppExport SEXP _gRim_update_ghkParms_(SEXP CparmsSEXP, SEXP dgen_idxSEXP, SEXP cgen_idxSEXP, SEXP ghk_obsSEXP, SEXP pms_obsSEXP, SEXP ghk_fitSEXP, SEXP pms_fitSEXP, SEXP scaleSEXP, SEXP detailsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type Cparms(CparmsSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type dgen_idx(dgen_idxSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type cgen_idx(cgen_idxSEXP);
    Rcpp::traits::input_parameter< List >::type ghk_obs(ghk_obsSEXP);
    Rcpp::traits::input_parameter< List >::type pms_obs(pms_obsSEXP);
    Rcpp::traits::input_parameter< List >::type ghk_fit(ghk_fitSEXP);
    Rcpp::traits::input_parameter< List >::type pms_fit(pms_fitSEXP);
    Rcpp::traits::input_parameter< double >::type scale(scaleSEXP);
    Rcpp::traits::input_parameter< double >::type details(detailsSEXP);
    rcpp_result_gen = Rcpp::wrap(update_ghkParms_(Cparms, dgen_idx, cgen_idx, ghk_obs, pms_obs, ghk_fit, pms_fit, scale, details));
    return rcpp_result_gen;
END_RCPP
}
// normalize_ghkParms_
List normalize_ghkParms_(List parms);
RcppExport SEXP _gRim_normalize_ghkParms_(SEXP parmsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type parms(parmsSEXP);
    rcpp_result_gen = Rcpp::wrap(normalize_ghkParms_(parms));
    return rcpp_result_gen;
END_RCPP
}
