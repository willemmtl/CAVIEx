# ----- FOR MONTE CARLO TEST -----

σ² = 100;
N1 = Normal(0, 1);
N2 = Normal(0, sqrt(σ²));

paramsMC = OrderedDict(
    :μ => (
        Normal,
        OrderedDict(
            :μ => (1, :η, () -> nothing),
            :σ => (2, :s, () -> nothing),
        )
    ),
);

modelMC = CAVIEx.Model(
    paramsMC,
    θ -> logpdf(N2, θ[1]),
    (θ, Hθ) -> logpdf(N1, θ[1]),
    () -> nothing,
);

hp0 = [0.0, 1.0];

initialize!(modelMC, hp0);


# ----- FOR REFINING TEST -----

m₁ = 2;
m₂ = 2;
m = m₁ * m₂;
F = iGMRF(m₁, m₂, 1, NaN);
Y = [float.([i*2, i*2+1]) for i = 0:m-1];

params = OrderedDict(
    :κᵤ => (
        Gamma,
        OrderedDict(
            :α => (2*m+1, :aᵤ, Hθ -> NormalMeanField.refine_aᵤ(F=F)),
            :θ => (2*m+2, :bᵤ, Hθ -> NormalMeanField.refine_bᵤ(Hθ, F=F)),
        ),
    ),
    (
        Symbol("μ$k") => (
            Normal,
            OrderedDict(
                :μ => (k, Symbol("η$k"), Hθ -> NormalMeanField.refine_η(Hθ, k, F=F, Y=Y)),
                :σ => (m+k, Symbol("s²$k"), Hθ -> NormalMeanField.refine_s²(Hθ, k, F=F, Y=Y)),
            )
        )
        for k = 1:m
    )...,
);

refiningModel = CAVIEx.Model(
    params,
    () -> nothing,
    () -> nothing,
    () -> nothing,
)

hp0 = [zeros(m)..., ones(m)..., 1.0, 1.0];

initialize!(refiningModel, hp0);