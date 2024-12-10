using Test

# Chemin vers le dossier contenant les tests
test_dir = "tests"

# Lister tous les fichiers se terminant par `.jl`
test_files = filter(f -> endswith(f, ".jl"), readdir(test_dir))

# Parcourir et exécuter chaque fichier de test
for file in test_files
    println("Exécution de $file...")
    include(joinpath(test_dir, file))
end

println("Tous les tests ont été exécutés.")