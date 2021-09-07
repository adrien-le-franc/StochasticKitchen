# developed with julia 1.5.3
#
# Quasi Newton method with Ipopt solver


DIR = @__DIR__
include(joinpath(DIR, "..", "functions.jl"))
include("model.jl")

using Ipopt
using Statistics


function optimize(model::PM.ParametricMultistageModel)

    # dimension
    n = horizon

    # search space
    x_L = zeros(horizon)
    x_U = ones(horizon)*peak_power

    # number of constraints
    m = 0

    # constraints bounds
    g_L = Float64[]
    g_U = Float64[]

    # objective
    function eval_f(p::Array{Float64})
        model.parameter = p
        return PM.parallel_compute_cost_to_go(model)
    end

    elapsed_per_oracle_call = Float64[]

    # evaluate grad objective
    function eval_grad_f(p::Array{Float64,1}, grad_f::Array{Float64,1})
        model.parameter = p
        t = @elapsed grad_f[:] = PM.parallel_compute_gradient(model, false)
        push!(elapsed_per_oracle_call, t)

        return
    end

    # evaluate constraint functions
    function eval_g(x, g)
        return
    end

    # evaluate jacobian of constraint function
    # see https://coin-or.github.io/Ipopt/IMPL.html#TRIPLET
    function eval_jac_g(x, mode, rows, cols, values)
        return
    end

    # problem
    prob = createProblem(
        n,
        x_L,
        x_U,
        m,
        g_L,
        g_U,
        0,
        0,
        eval_f,
        eval_g,
        eval_grad_f,
        eval_jac_g,
        nothing,
    )

    # quasi-newton
    addOption(prob, "hessian_approximation", "limited-memory")

    # stopping test
    addOption(prob, "acceptable_iter", 5)
    addOption(prob, "acceptable_tol", 1.)
    addOption(prob, "acceptable_compl_inf_tol", 1.)
    addOption(prob, "acceptable_obj_change_tol", 0.005)
    addOption(prob, "max_iter", 100)

    # verbose
    addOption(prob, "print_level", 0)   
    
    # callback

    n_steps = 0
    f_values = Float64[]

    function intermediate(
        alg_mod::Int,
        iter_count::Int,
        obj_value::Float64,
        inf_pr::Float64,
        inf_du::Float64,
        mu::Float64,
        d_norm::Float64,
        regularization_size::Float64,
        alpha_du::Float64,
        alpha_pr::Float64,
        ls_trials::Int,
    )
        
        n_steps += 1
        push!(f_values, obj_value)

        return true 
    end

    setIntermediateCallback(prob, intermediate)

    # initialize
    prob.x = zeros(n)

    # solve !
    overall_time = @elapsed status = solveProblem(prob)

    # log
    return  Dict(
        "x_opt" => prob.x,
        "f_values" => f_values,
        "n_steps" => n_steps,
        "overall_time" => overall_time,
        "average_time_per_gradient_call" => mean(elapsed_per_oracle_call[2:end]))

end