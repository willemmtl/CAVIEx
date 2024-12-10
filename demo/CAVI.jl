using Optim, Distributions, Gadfly

include("../precipFramework/iGMRF.jl");
include("../precipFramework/utils.jl");
include("model.jl");


"""
    runCAVI(n_epoch, epoch_size; y)

Run the CAVI algorithm over the given data and spatial scheme.

# Arguments :
- `n_epoch::Integer`: Number of epochs to compute i.e. number of type we'll compute the convergence criterion.
- `epoch_size::Integer`: Size of each epoch i.e. number of iterations before computing the convergence criterion.
- `Hθ₀::DenseVector`: Initial values of the hyper-parameters.
- `Y::Vector{Float64}`: Sample (xᵢ)ᵢ.
"""
function runCAVI(n_epoch::Integer, epoch_size::Integer, Hθ₀::DenseVector; Y::Vector{Float64})

    ### ---- Initialization ---- ###
    Hθ = Hθ₀;
    
    ### ---- Ascent ---- ###
    MCKL = Float64[];
    duration = @elapsed begin
        for _ = 1:n_epoch
            push!(MCKL, MonteCarloKL(Hθ, Y=Y))
            for _ = 1:epoch_size
                updateHyperParams!(Hθ, Y=Y);
            end
        end
    end
    println("Minimization made in ", duration, " s")
    
    return MCKL, Hθ
end;


"""
    MonteCarloKL(Hθ; Y)

Compute the convergence criterion with current hyper-parameters.

# Arguments :
- `Hθ::DenseVector`: Hyper-parameters [α, β, m, s²].
- `Y::Vector{Float64}`:: Sample.
"""
function MonteCarloKL(Hθ::DenseVector; Y::Vector{Float64})
    
    N = 1000;
    Θ = zeros(4, N);

    α, β, m, s² = Hθ;

    # Unpacking
    Hθ[1] = compute_α(Y=Y);
    Hθ[2] = compute_β(m, s², Y=Y);
    Hθ[3] = compute_m(Y=Y);
    Hθ[4] = compute_s²(α, β, Y=Y);
    
    Θ[1, :] = rand(Normal(m, sqrt(s²)), N);
    Θ[2, :] = rand(Gamma(α, β), N);
    
    logTarget = evaluateLogMvDensity(x -> logTargetDensity(x, Y=Y), Θ);
    logApprox = evaluateLogMvDensity(x -> logApproxDensity(x, Hθ=Hθ), Θ);
    
    return sum(logApprox .- logTarget) / N
end;


"""
    updateHyperParams!(Hθ; Y)

Update approximation parameters β and s².

Correspond to one iteration in an epoch of the CAVI algorithm.

# Arguments :
- `Hθ::DenseVector`: Last updated hyper-parameters [α, β, m, s²].
- `Y::Vector{Float64}`: Sample.
"""
function updateHyperParams!(Hθ::DenseVector; Y::Vector{Float64})
    
    α, β, m, s² = Hθ;

    Hθ[2] = compute_β(m, s², Y=Y);
    Hθ[4] = compute_s²(α, β, Y=Y);
    
end;