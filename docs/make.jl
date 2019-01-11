using Documenter, MultivariateFunctions

makedocs(
    format = Documenter.HTML(),
    sitename = "MultivariateFunctions",
    modules = [MultivariateFunctions],
    pages = ["index.md",
             "1_structs_and_limitations.md",
             "2_Interpolation_and_splines.md",
             "3_examples_algebra.md",
             "4_examples_interpolation.md",
             "5_examples_approximation.md",
             "99_refs.md"]
)

deploydocs(
    repo   = "github.com/s-baumann/MultivariateFunctions.jl.git",
    target = "build",
    deps   = nothing,
    make   = nothing
)
