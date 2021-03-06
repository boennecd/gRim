#include <R.h>
#include <Rinternals.h>
#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>

/* FIXME: 
   Check these declarations against the C/Fortran source code.
*/

/* .C calls */
extern void Cggmfit(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);

/* .Call calls */
extern SEXP C_ghk2pms(SEXP);
extern SEXP C_pms2ghk(SEXP);
extern SEXP _gRim_ghk2pmsParms_(SEXP);
extern SEXP _gRim_normalize_ghkParms_(SEXP);
extern SEXP _gRim_pms2ghkParms_(SEXP);
extern SEXP _gRim_updateA(SEXP, SEXP, SEXP, SEXP);
extern SEXP _gRim_update_ghkParms_(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _gRim_cpp_ggmfit(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _gRim_cpp_ggmfit_wood(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _gRim_cpp_ggmfit_reg(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);

static const R_CMethodDef CEntries[] = {
    {"Cggmfit", (DL_FUNC) &Cggmfit, 14},
    {NULL, NULL, 0}
};

static const R_CallMethodDef CallEntries[] = {
    {"C_ghk2pms",                 (DL_FUNC) &C_ghk2pms,                 1},
    {"C_pms2ghk",                 (DL_FUNC) &C_pms2ghk,                 1},
    {"_gRim_ghk2pmsParms_",       (DL_FUNC) &_gRim_ghk2pmsParms_,       1},
    {"_gRim_normalize_ghkParms_", (DL_FUNC) &_gRim_normalize_ghkParms_, 1},
    {"_gRim_pms2ghkParms_",       (DL_FUNC) &_gRim_pms2ghkParms_,       1},
    {"_gRim_updateA",             (DL_FUNC) &_gRim_updateA,             4},
    {"_gRim_update_ghkParms_",    (DL_FUNC) &_gRim_update_ghkParms_,    9},
    {"_gRim_cpp_ggmfit",          (DL_FUNC) &_gRim_cpp_ggmfit,          9},
    {"_gRim_cpp_ggmfit_wood",     (DL_FUNC) &_gRim_cpp_ggmfit_wood,     9},
    {"_gRim_cpp_ggmfit_reg",      (DL_FUNC) &_gRim_cpp_ggmfit_reg,      9},
    {NULL, NULL, 0}
};

void R_init_gRim(DllInfo *dll)
{
    R_registerRoutines(dll, CEntries, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
