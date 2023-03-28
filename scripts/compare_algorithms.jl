using MoM_Basics, MoM_Kernels
using Lebedev_Interpolations
using MoM_Kernels:calZfarI!
using BenchmarkTools, Printf

SimulationParams.SHOWIMAGE = true
setPrecision!(Float64)

filenames = ["meshfiles/f22_100MHz.nas", "meshfiles/f22_300MHz.nas", "meshfiles/f22_600MHz.nas"
                ]#"meshfiles/sphere_r1m_metal_300MHz.nas", "meshfiles/sphere_r1m_metal_600MHz.nas", "meshfiles/sphere_r1m_metal_1dot2GHz.nas", 
                # "meshfiles/helicopter_300MHz.nas", "meshfiles/sphere_r1m_metal_2dot4GHz.nas", "meshfiles/sphere_r1m_metal_4dot8GHz.nas", 
fs = [1e8, 3e8, 6e8]#3e8, 6e8, 1.2e9, 3e8, 2.4e9, 4.8e9, 

interp_algorithms = [:Lagrange2Step, :Lagrange1Step, :LbTrained1Step]

for (filename, f) in zip(filenames, fs)
for interp_algo in interp_algorithms
    inputParameters(;frequency = f, ieT = :CFIE)
    set_Interpolation_Method!(interp_algo)

    meshData, εᵣs   =  getMeshData(filename; meshUnit=:m);
    ngeo, nbf, geosInfo, bfsInfo =  getBFsFromMeshData(meshData; sbfT = :RWG)


    MoM_Kernels.set_leafCubeSize!(0.25Params.λ_0)
    nLevels, octree     =   getOctreeAndReOrderBFs!(geosInfo, bfsInfo; nInterp = 4);

    # 叶层
    leafLevel   =   octree.levels[nLevels];
    # 计算近场矩阵CSC
    ZnearCSC     =   calZnearCSC(leafLevel, geosInfo, bfsInfo);

    # 构建矩阵向量乘积算子
    Zopt    =   MLMFAIterator(ZnearCSC, octree, geosInfo, bfsInfo);

    ## 根据近场矩阵和八叉树计算 SAI 左预条件
    Zprel   =   sparseApproximateInversePl(ZnearCSC, leafLevel)

    source  =   PlaneWave(π/2, 0, 0f0, 1f0)
    # source  =   MagneticDipole(;Iml = 1., phase = 0., orient = (π/2, π/2, 0))
    V    =   getExcitationVector(geosInfo, size(ZnearCSC, 1), source);

    ICoeff, ch   =   solve(Zopt, V; solverT = :gmres, Pl = Zprel);
    ## 观测角度
    θs_obs  =   LinRange{Precision.FT}(-π/2,  π/2,  1441)
    ϕs_obs  =   LinRange{Precision.FT}(   0,  π/2,     3)
    
    # RCS
    radarCrossSection(θs_obs, ϕs_obs, ICoeff, geosInfo)

    # 比较远场矩阵向量乘积计算时间 
    b = @benchmark calZfarI!($Zopt, $ICoeff)

    open(joinpath(SimulationParams.resultDir, "InputArgs.txt"), "a+") do io
        bmean = mean(b.times)/1e9
        @printf(io,  "%-20s %10s\n", "插值方式", String(interp_algo))
        @printf(io,  "%-20s %10.3f\n", "远场矩阵向量乘积", bmean)
    end
    
end
end