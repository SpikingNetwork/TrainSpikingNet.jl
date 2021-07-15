using Test, JLD

mkdir(joinpath(@__DIR__,"cpu-data"))
cp(joinpath(@__DIR__,"param.jl"), joinpath(@__DIR__,"cpu-data","param.jl"))
init_out = readlines(pipeline(`$(Base.julia_cmd())
                               $(joinpath(@__DIR__,"..","src","main-initialization.jl"))
                               $(joinpath(@__DIR__,"cpu-data"))`))
for iHz in findall(contains("Hz"), init_out)
  @test 0 < parse(Float64, match(r"([.0-9]+) Hz", init_out[iHz]).captures[1]) < 10
end

cp(joinpath(@__DIR__,"cpu-data"), joinpath(@__DIR__,"gpu-data"))
cpu_out = readlines(pipeline(`$(Base.julia_cmd())
                              $(joinpath(@__DIR__,"..","src","cpu","main-train.jl"))
                              $(joinpath(@__DIR__,"cpu-data"))`))
gpu_out = readlines(pipeline(`$(Base.julia_cmd())
                              $(joinpath(@__DIR__,"..","src","gpu","main-train.jl"))
                              $(joinpath(@__DIR__,"gpu-data"))`))

iHz = findlast(contains("Hz"), cpu_out)
@test 0 < parse(Float64, match(r"([.0-9]+) Hz", cpu_out[iHz]).captures[1]) < 10
iHz = findlast(contains("Hz"), gpu_out)
@test 0 < parse(Float64, match(r"([.0-9]+) Hz", cpu_out[iHz]).captures[1]) < 10

cpu_wpWeightIn = load(joinpath(@__DIR__,"cpu-data","wpWeightIn-trained.jld"))["wpWeightIn"]
cpu_wpWeightOut = load(joinpath(@__DIR__,"cpu-data","wpWeightOut-trained.jld"))["wpWeightOut"]
gpu_wpWeightIn = load(joinpath(@__DIR__,"gpu-data","wpWeightIn-trained.jld"))["wpWeightIn"]
gpu_wpWeightOut = load(joinpath(@__DIR__,"gpu-data","wpWeightOut-trained.jld"))["wpWeightOut"]
@test isapprox(reshape(transpose(cpu_wpWeightIn), 40, 1, 2000), gpu_wpWeightIn)
@test isapprox(cpu_wpWeightOut, gpu_wpWeightOut)
