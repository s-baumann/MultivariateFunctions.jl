using Documenter, MultivariateFunctions

makedocs(
    format = :html,
    sitename = "MultivariateFunctions",
    modules = [MultivariateFunctions]
)

deploydocs(
    repo   = "github.com/s-baumann/MultivariateFunctions.jl.git",
    julia  = "1.0",
    target = "build",
    deps   = nothing,
    make   = nothing
)
