module RuiExamples

using IntervalCensoredEventTimes, Ipopt, KNITRO, MAT, Test

@info "load n=200, p=1000 example data"
datafile = "/Users/huazhou/Desktop/interval_censoring_examples/n200p1000_Identity.mat"
file = matopen(datafile)
Z = read(file, "Z")
βtrue = read(file, "beta_tr")
L = vec(read(file, "L"))
R = vec(read(file, "R"))
close(file)

# form a data set with with p=30 for benchmark
Zbm = [ones(size(Z, 1)) Z[:, [1, 50, 100, 150, 200]] Z[:, 2:25]]
βbm = [0; βtrue[[1, 50, 100, 150, 200]]; zeros(24)]
@show βbm
# form model object
icm = IntervalCensoredModel(Zbm, L, R)
display(icm); println()
# fit model
# KNITRO options:
# algorithm (1, IP/Direct; 2, IP/CG; 3, Active Set; 4, SQP (fastest); 5, try all)
# hessopt (2, BFGS; 6, L-BFGS)
# derivcheck (1, gradient; 2, hessian; 3, both)
# derivcheck_tol
# derivcheck_type (1, forward; 2, central)
# maxit
# solver = KNITRO.KnitroSolver(outlev = 3, derivcheck = 3, derivcheck_tol = 1e-4, maxit = 10)
solver = KNITRO.KnitroSolver(outlev = 3, hessopt = 2)
# IPOPT options:
# hessian_approximation = "limited-memory"
# solver = Ipopt.IpoptSolver(print_level = 3, hessian_approximation = "limited-memory")
fit!(icm, solver, init=initialize_uniform!(icm))
display(icm); println()
end
