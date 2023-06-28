module ICCA

using Manifolds
using LinearAlgebra
using Statistics
using CSV
using DataFrames

include("function.jl")
include("data.jl")
include("extrinsicCCA.jl")
include("intrinsicCCA.jl")
SO3 = Manifolds.SpecialOrthogonal(3)
SO2 = Manifolds.SpecialOrthogonal(2)


end