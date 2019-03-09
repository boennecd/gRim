context("Testing cmod")

test_that("different 'cmod' version gives the same", {
  data(carcass)
  obj <- cm1 <- cmod(
    ~ Fat11 * Fat12 * Fat13 + Meat11 + Meat12 + Meat13 + LeanMeat, 
    data = carcass, fit = FALSE)
  
  f_c   <- fit(obj, engine = "ggmfit")
  f_cpp <- fit(obj, engine = "ggmfit-cpp")
  
  expect_equal(f_c, f_cpp)
  
  to_test <- f_c[names(f_c) != "datainfo"]
  expect_known_value(to_test, "carcass-test.RDS")
})