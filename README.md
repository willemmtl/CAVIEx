CAVI algorithm for extremes.

# General usage

## How to run CAVIEx on a given model

1. Go to `playground.ipynb`. Let the **first cell untouched**.
2. Instantiate a model in the **second cell**. To do so, you can see what arguments are required in the according instance's file within the `src/instances` folder.
*Example for a Precip Mean-Field model :*
```
m₁ = 10; # Number of rows of the grid.
m₂ = 10; # Number of columns of the grid.
m = m₁ * m₂;
seed = 400;
biasmu = 10.0;
realkappa = 10.0;

instance = InstancePMF(m₁, m₂, seed=seed, biasmu=biasmu, realkappa=realkappa);
```
3. Specify the number of epochs, the epoch size and the initial values of the hyper-parameters in the **third cell**. Then use the `runCAVI` function to run the algorithm on your model.
**NB :** For now the `runCAVI` function is taking a `CAVIEx.Model` structure as argument, see example bellow.
*Example for a Precip Mean-Field model :*
```
n_epoch = 2;
epoch_size = 10;
hp0 = [zeros(m)..., ones(m)..., 1.0, 1.0];

MCKL = runCAVI(n_epoch, epoch_size, hp0, instance.model);
```

## How to see the results

Several functions have been implemented for post analysis :
- `plotConvergenceCriterion(MCKL)`: Allow you to see the convergence criterion value (in our case the KL divergence) through the **epochs**.
- `plotTrace(model, hyperParam)`: Allow you to see the trace of a given hyper-parameter through the **iterations**.
- `plotApproxVSMCMC(model, param, a, b, step)`: Allow you to compare the approximation density of a given parameter with its MCMC simulation.
**NB :** For this last function, you MUST run the `runMCMC!` function on your model beforehand.

# Development for new models

## How to create a new instance (i.e. a new specific model)

Create a `YourInstance.jl` file in `src/instances` containing a structure named `YourInstance` which **must have** a `model` attribute with the type CAVIEx.Model.

The model can now be built in the constructor of your new instance structure.

Every specific function regarding the new model must be implemented in the `src/models` folder and be gathered in its own **module**.

See for example `src/instances/InstanceDemo.jl` and `src/models/Demo.jl`.

## Other organizational aspects

To alleviate the code, all functions regarding fake data generation and mcmc simulation must be respectively implemented in `src/dataGen` and `src/mcmc`