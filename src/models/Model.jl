using OrderedCollections

include("../Param.jl");
include("../HyperParam.jl");


"""
    Model

Generic model for the CAVIEx algorithm.

# Attributes :
- `params::OrderedDict{Symbol, Param}`: Parameters that we're trying to estimate.
- `nParams::Integer`: Number of parameters.
- `hyperParams::OrderedDict{Symbol, HyperParam}`: Hyper-parameters that define approximation densities.
- `nHyperParams::Integer`: Number of hyper-parameters.
- `pHp::OrderedDict{Symbol, OrderedDict{Symbol, Symbol}}`: Relation between parameters and hyperParameters.
- `logTargetDensity::Function`: Density that we're trying to approximate.
    Must be of the forme θ::Vector -> f(θ)
- `logApproxDensity::Function`: Approximating density.
    Must be of the forme (θ::Vector, Hθ::Vector) -> f(θ, Hθ)
- `mcmcSampler::Function`: MCMC algorithm to imitate the target density.
    Must be of the form n_iter -> f(n_iter)
"""
struct Model
    params::OrderedDict{Symbol, Param}
    nParams::Integer
    hyperParams::OrderedDict{Symbol, HyperParam}
    nHyperParams::Integer
    pHp::OrderedDict{Symbol, OrderedDict{Symbol, Symbol}}
    logTargetDensity::Function
    logApproxDensity::Function
    mcmcSampler::Function

    """
    Constructor.

    # Arguments
    - `parameters::OrderedDict{Symbol, Tuple{UnionAll, OrderedDict{Symbol, Tuple{Int64, Symbol, Function}}}}`:
        Ordered dictionnary containing { 
            :param's symbol in our model => (
                The law of its marginal approximating distribution (e.g. Normal),
                Another ordered dict containing the hyperparameters' information i.e. {
                    :the conventional symbol for the hyper-parameter in the Distribution package => (
                        numero of the hyper-parameter,
                        :hyper-param's symbol in our model,
                        its refining function,
                    )
                }
            )
        )}
        WARNING : the order in which parameters are placed in the dictionnary 
        reflects the order in which they will be refined during the refining process (see CAVI).
    - `logTargetDensity::Function`: Density that we're trying to approximate.
        Must be of the forme θ::Vector -> f(θ)
    - `logApproxDensity::Function`: Approximating density.
        Must be of the forme (θ::Vector, Hθ::Vector) -> f(θ, Hθ)
    - `mcmcSampler::Function`: MCMC algorithm to imitate the target density.
        Must be of the form n_iter -> f(n_iter)                    
    """
    function Model(
        parameters::OrderedDict{Symbol, Tuple{UnionAll, OrderedDict{Symbol, Tuple{Int64, Symbol, Function}}}},
        logTargetDensity::Function,
        logApproxDensity::Function,
        mcmcSampler::Function,
    )
        
        params = isolateParams(parameters);
        nParams = length(keys(params));
        hyperParams = isolateHyperParams(parameters);
        nHyperParams = length(keys(hyperParams));
        pHp = mapParamsToHyperParams(parameters);

        new(
            params,
            nParams,
            hyperParams,
            nHyperParams,
            pHp,
            logTargetDensity,
            logApproxDensity,
            mcmcSampler,
        )
    
    end


    """
        isolateParams(parameters)

    Gather all the parameters from the `parameters` dictionnary.

    # Arguments
    - `parameters`: Ordered dict containing all the information on parameters and hyper-parameters.
        See details in the constructor signature.
    
    # Return
    A dictionnary with the form name => Parameter(approxMarginal, num=num).
    """
    function isolateParams(parameters::OrderedDict{Symbol, Tuple{UnionAll, OrderedDict{Symbol, Tuple{Int64, Symbol, Function}}}})
        return OrderedDict(name => Param(props[1], num=num) for (num, (name, props)) in enumerate(parameters))
    end


    """
        isolateHyperParams(parameters)

    Gather all the hyper-parameters from the `parameters` dictionnary.
    
    # Arguments
    - `parameters`: Ordered dict containing all the information on parameters and hyper-parameters.
        See details in the constructor signature.

    # Return :
    A dictionnary with the form hpName => HyperParameter(refiningFunction, num=num).
    """
    function isolateHyperParams(parameters::OrderedDict{Symbol, Tuple{UnionAll, OrderedDict{Symbol, Tuple{Int64, Symbol, Function}}}})
        
        hyperParams = OrderedDict{Symbol, HyperParam}();
        for (_, subdict) in parameters
            for (_, (num, hpName, hpFunc)) in subdict[2]
                hyperParams[hpName] = HyperParam(hpFunc, num=num);
            end
        end

        return hyperParams
    end

    """
        mapParamsToHyperParams(parameters)

    Map the parameters with the according hyper-parameters from the `parameters` dictionnary.

    # Arguments
    - `parameters`: Ordered dict containing all the information on parameters and hyper-parameters.
        See details in the constructor signature.

    # Return :
    A nested dictionnary of the form name => Dict(fieldName => hpName).
    """
    function mapParamsToHyperParams(parameters::OrderedDict{Symbol, Tuple{UnionAll, OrderedDict{Symbol, Tuple{Int64, Symbol, Function}}}})
        return Dict(
            param => Dict(
                fieldName => hyperParam[2] 
                for (fieldName, hyperParam) in props[2]
            ) 
            for (param, props) in parameters
        )
    end

end


"""
    initialize!(model, hp0)

Assign the given initial values to the given hyper-parameters.

Synchronize the parameters' approximating marginals.

# Arguments :
- `hp0::DenseVector`: Initial values to assign IN THE SAME ORDER as the hyperparameters' NUMERO.
"""
function initialize!(model::Model, hp0::DenseVector)

    for (_, hyperParam) in model.hyperParams
        hyperParam.trace = [hp0[hyperParam.numero]];
    end

    syncMarginals!(model);

end


"""
    generateApproxSample(model, N)

Draw samples from the approximating distribution with current hyper-parameters.

Use the mean-field approximation by generating each variable independantly.
Will be used to compute KL divergence.

# Arguments :
- `N::Integer`: Sample size.
"""
function generateApproxSample(model::Model, N::Integer)
    
    supp = zeros(model.nParams, N);

    for (_, param) in model.params
        supp[param.numero, :] = draw(param, N);
    end
    
    return supp
end


"""
    syncMarginals!(model)

Update each approximating marginal distribution with the current value of the hyper-parameters.
"""
function syncMarginals!(model::Model)

    for (name, param) in model.params
        conventionalFieldNames = getConventionalFieldNames(param.approxDistribution);
        hpNames = get.(Ref(model.pHp[name]), conventionalFieldNames, nothing);
        if any(isnothing.(hpNames))
            error("The placeholders for the law of $name are wrong !")
        else
            hpValues = [current.(get.(Ref(model.hyperParams), hpNames, nothing))...];
            adaptHp!(param.approxDistribution, hpValues);
        end
        param.approxMarginal = param.approxDistribution(hpValues...);
    end

end


"""
    getConventionalFieldNames(dist)

Give the symbols of a distribution's parameters as they are in the Distributions package.

# Arguments :
- `dist::UnionAll`: the Distribution whose field names we want.
"""
function getConventionalFieldNames(dist::UnionAll)
    return fieldnames(dist)
end


"""
    getHpValues(model)

Collect the current value of each hyper-parameter in the right order (i.e. numero in ascending order).

# Return
Vector containing the current values.
"""
function getHpValues(model::Model)

    values = zeros(model.nHyperParams);

    for (_, hp) in model.hyperParams
        values[hp.numero] = current(hp);
    end

    return values;
end


"""
    runMCMC!(model)

Generate samples from the real target density through the model's MCMC algorithm.

Store each sample in its corresponding param.

# Arguments :
- `niter::Integer`: Number of MCMC iterations.
"""
function runMCMC!(model::Model, niter::Integer)
    chains = model.mcmcSampler(niter)
    for (name, param) in model.params
        param.mcmcSample = chains[:, String(name), 1].value[:, 1];
        param.warmingSize = Int(0.1 * niter)
    end
end