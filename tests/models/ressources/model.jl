# ----- FOR MOST OF THE TESTS -----

paramsBase = OrderedDict(
    :μ => (
        Normal,
        OrderedDict(
            :μ => (1, :η, x -> x + 1),
            :σ => (2, :s, x -> x + 2),
        )
    ),
    :σ => (
        Gamma,
        OrderedDict(
            :α => (3, :a, x -> x + 3),
            :θ => (4, :b, x -> x + 4),
        )
    ),
);

modelBase = CAVIEx.Model(
    paramsBase,
    () -> nothing,
    () -> nothing,
    () -> nothing,
);


# ----- FOR PARSING TESTS -----

m = 9;

paramsToParse = OrderedDict(
    (
        Symbol("μ$k") => (
            Normal,
            OrderedDict(
                :μ => (k, Symbol("η$k"), Hθ -> PrecipMeanField.refine_η(Hθ, k, F=F, Y=data)),
                :σ => (m+k, Symbol("s²$k"), Hθ -> PrecipMeanField.refine_s²(Hθ, k, F=F, Y=data)),
            )
        )
        for k = 1:m
    )...
)


# ----- FOR SYNC MARGINAL TEST -----

modelSync = CAVIEx.Model(
    paramsBase,
    () -> nothing,
    () -> nothing,
    () -> nothing,
);

modelSync.hyperParams[:η].trace = [0.0];
modelSync.hyperParams[:s].trace = [4.0];
modelSync.hyperParams[:a].trace = [1.0];
modelSync.hyperParams[:b].trace = [2.0];

