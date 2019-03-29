context("Testing cmod")

test_that("different 'cmod' version gives the same", {
  data(carcass)
  obj <- cmod(
    ~ Fat11 * Fat12 * Fat13 + Meat11 + Meat12 + Meat13 + LeanMeat, 
    data = carcass, fit = FALSE)
  
  f_c        <- fit(obj, engine = "ggmfit")
  f_cpp      <- fit(obj, engine = "ggmfit-cpp")
  f_cpp_wood <- fit(obj, engine = "ggmfit-cpp-wood")
  f_cpp_reg  <- fit(obj, engine = "ggmfit-cpp-reg")
  
  expect_equal(f_c, f_cpp)
  expect_equal(f_c, f_cpp_wood)
  expect_equal(f_c, f_cpp_reg)
  
  to_test <- f_c[names(f_c) != "datainfo"]
  expect_known_value(to_test, "carcass-test.RDS")
  
  # dense case
  obj <- cmod(~ .^., data = carcass, fit = FALSE)
  
  f_c        <- fit(obj, engine = "ggmfit")
  f_cpp      <- fit(obj, engine = "ggmfit-cpp")
  f_cpp_wood <- fit(obj, engine = "ggmfit-cpp-wood")
  f_cpp_reg  <- fit(obj, engine = "ggmfit-cpp-reg")
  
  expect_equal(f_c, f_cpp)
  expect_equal(f_c, f_cpp_wood)
  expect_equal(f_c, f_cpp_reg)
})

