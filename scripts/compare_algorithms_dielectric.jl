using MoM_Basics, MoM_Kernels
using Lebedev_Interpolations
using MoM_Kernels:calZfarI!
using BenchmarkTools, Printf

SimulationParams.SHOWIMAGE = true
setPrecision!(Float64)
include("antenna_array.jl")

filenames = ["meshfiles/missileHT10mm.nas", 
                ]#"meshfiles/missileHT7mm.nas", "meshfiles/missileHT5mm.nas"

fs = [15e8, ]#20e8, 30e8

interp_algorithms = [:Lagrange2Step, :Lagrange1Step, :LbTrained1Step]

for (filename, f) in zip(filenames, fs)
for interp_algo in interp_algorithms
    inputParameters(;frequency = f, ieT = :EFIE)
    set_Interpolation_Method!(interp_algo)

    meshData, εᵣs   =  getMeshData(filename; meshUnit=:mm);
    ngeo, nbf, geosInfo, bfsInfo =  getBFsFromMeshData(meshData; sbfT = :nothing, vbfT = :PWC)

    setGeosPermittivity!(geosInfo, 2.81(1-0.001im))

    # εᵣmean = map(geos -> mapreduce(g -> g.ε, +, geos)/length(geos), geos)

    # MoM_Kernels.set_leafCubeSize!(0.25Params.λ_0/sqrt(real(εᵣmean)))
    nLevels, octree     =   getOctreeAndReOrderBFs!(geosInfo, bfsInfo; nInterp = 4);

    # 叶层
    leafLevel   =   octree.levels[nLevels];
    # 计算近场矩阵CSC
    ZnearCSC     =   calZnearCSC(leafLevel, geosInfo, bfsInfo);

    # 构建矩阵向量乘积算子
    Zopt    =   MLMFAIterator(ZnearCSC, octree, geosInfo, bfsInfo);

    ## 根据近场矩阵和八叉树计算 SAI 左预条件
    Zprel   =   sparseApproximateInversePl(ZnearCSC, leafLevel)

    # 源
    source  =   dipolediffArray(0.; arraysize = 0.32)
    # source  =   MagneticDipole(;Iml = 1., phase = 0., orient = (π/2, π/2, 0))
    V    =   getExcitationVector(geosInfo, size(ZnearCSC, 1), source);

    ICoeff, ch   =   solve(Zopt, V; solverT = :gmres, Pl = Zprel);
    ## 观测角度
    θs_obs  =   LinRange{Precision.FT}(-π/2,  π/2,  1441)
    ϕs_obs  =   LinRange{Precision.FT}(   0,  π/2,     3)
    
    # 远场电场
    _, _, farEDe, farEDedB = farField(θs_obs, ϕs_obs, ICoeff, geosInfo, source)

    # 比较远场矩阵向量乘积计算时间 
    b = @benchmark calZfarI!($Zopt, $ICoeff)

    open(joinpath(SimulationParams.resultDir, "InputArgs.txt"), "a+") do io
        bmean = mean(b.times)/1e9
        @printf(io,  "%-20s %10s\n", "插值方式", String(interp_algo))
        @printf(io,  "%-20s %10.3f\n", "远场矩阵向量乘积", bmean)
    end
    
end
end