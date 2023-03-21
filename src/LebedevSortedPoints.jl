# 该模组用于预处理 Lebedev 求积点文件
"""
该模组用于预处理 Lebedev 求积点文件，得到直角坐标下，经排序的点文件
格式为 x, y, z w, 文件名为 “阶数.点数”
"""
# module LebedevSorted


# export getlbSortedData, visualizeScatterOnSphere

"""
构建 lebedev nodes 的 t 值与文件名的字典
"""
function lbnt2fnDictConstruct(filedirs::String = "deps/sphere_lebedev/")
    # 读取所有文件名
    sphlebeFileNames     =   readdir(filedirs)
    # 初始化 t 值 → 文件名的字典
    t2fnDict    =   Dict{Int, String}()
    # 循环写入字典
    for filename in sphlebeFileNames
        try 
            p    =   parse(Int, split(filename, ".")[1])
            t2fnDict[p] =   filename
        catch
            nothing
        end
    end

    return t2fnDict
end


const lbnP2FILEDict    =   lbnt2fnDictConstruct()
const TargetDir = "deps/sphere_lebedev/nodesSorted/"


"""
根据 p 值寻找对应的文件名称
"""
function modiTgetFileName(p::Int, T2FILEDict::Dict)
    p > max(keys(T2FILEDict)...) && throw("t值过大，没有对应文件")
    let filename::String
        try
            filename = T2FILEDict[p]
        catch KeyError
            filename = modiTgetFileName(p + 1, T2FILEDict)
        end
        return filename
    end
end


"""
根据文件名读取 lebedev nodes 文件
"""
function readlbndata(filename::String, lebedevFileDir = "deps/sphere_lebedev/")

    # 打开文件读取
    file = open(lebedevFileDir*filename, "r")

    # 先解析有多少点
    nNodes  =   countlines(file)
    # 重新把指针定位到开头
    seekstart(file)
    # 创建保存这些点的数组
    nodes   =   zeros(Float64, 3, nNodes)
    # 权重
    weights =   zeros(Float64, nNodes)  

    # 找出行数
    lineID  =   0
    for line in eachline(file)
        lineID += 1
        # 将三个坐标点拆分
        contents = split(line)
        # 读取数据
        θ   =   parse(Float64, contents[1])/180*π
        ϕ   =   parse(Float64, contents[2])/180*π

        # 计算 θ ϕ 正余弦
        sinθ, cosθ  =   sincos(θ)
        sinϕ, cosϕ  =   sincos(ϕ)
        # 写入数据
        nodes[1, lineID]    =   cosθ*sinϕ
        nodes[2, lineID]    =   sinθ*sinϕ
        nodes[3, lineID]    =   cosϕ
        weights[lineID]     =   parse(Float64, contents[3])*4π
    end

    close(file)

    return nodes, weights
end

"""
修正浮点数误差引起的数值最后一位不同导致排序失败的问题
"""
function modiFloatPointError!(nodes)

    nodesData   =   reshape(copy(nodes), :)
    sortp   =   sortperm(nodesData, by = abs)
    for ii in 1:(length(sortp) - 1)
        vii     =   nodesData[sortp[ii]]
        viip    =   nodesData[sortp[ii+1]]
        abs(abs(vii) - abs(viip)) < 1e-10 && begin
            nodesData[sortp[ii+1]] = sign(nodesData[sortp[ii+1]])*abs(vii)
        end
    end

    nodes[:] .= nodesData

    return nothing

end

"""
根据 p 值读取 lb 数据并更正t值后返回
"""
function getlbdata(p::Int)
    # 找到存在的 t 值和文件名
    filename = modiTgetFileName(p, lbnP2FILEDict)
    # 读取数据
    nodes, weights = readlbndata(filename)

    # 剔除冗余元素
    for ij in eachindex(nodes)
        (abs(nodes[ij]) < 3eps(eltype(nodes))) &&  begin nodes[ij] = 0; end
    end

    # 修正浮点数误差导致的对称点差异
    modiFloatPointError!(nodes)

    # 跟权重一起排序令数据对称
    nodesWeightsSorted = sortslices([nodes; reshape(weights,1, :)], dims = 2, alg = Base.Sort.MergeSort,  rev=true)
    # 提取排序后的数据
    nodesSorted =   nodesWeightsSorted[1:3, :]
    weights    .=   nodesWeightsSorted[4, :]

    nNodes = size(nodesSorted, 2)
    # 是否是偶数个点？
    @assert iseven(nNodes)
    for icol in 1:(nNodes ÷ 2)
        @assert isapprox(nodesSorted[:, icol], -nodesSorted[:, end+1 - icol], rtol = 1e-4) "第$(icol)列对不上"
    end

    return nodesSorted, weights
end

"""
根据得到的修正并排序后的点写入文件
"""
function writeLebedevNodes(p::Int, nodes::Matrix{FT}, weights::Vector{FT}; targetdir   =   TargetDir) where {FT<:Real}
    # 点数量
    nNodes  =   size(nodes, 2)
    # 目标文件夹
    ~isdir(targetdir) && mkpath(targetdir)
    # 写入文件
    open(targetdir*"$(p).$(nNodes).txt", "w+") do file
        for ii in 1:nNodes
            nodeCol = view(nodes,:, ii)
            write(file, "$(nodeCol[1]) $(nodeCol[2]) $(nodeCol[3]) $(weights[ii])\n")
        end
    end

    nothing
end

"""
初始化
"""
function __init__(;targetdir   =   TargetDir)
    for p in keys(lbnP2FILEDict)

        # 读取修正、排序后数据
        nodesSorted, weights =  getlbdata(p)
        # 写入文件
        writeLebedevNodes(p, nodesSorted, weights; targetdir   =   TargetDir)

    end
end

function lbnSorted2fnDictConstruct(filedirs::String = TargetDir)
    # 读取所有文件名
    sphlebeFileNames     =   readdir(filedirs)
    # 初始化 t 值 → 文件名的字典
    t2fnDict    =   Dict{Int, String}()
    # 循环写入字典
    for filename in sphlebeFileNames
        p    =   parse(Int, split(filename, ".")[1])
        t2fnDict[p] =   filename 
    end

    return t2fnDict
end

const lbnSortedP2FILEDict    =   lbnSorted2fnDictConstruct()


"""
返回排序后的点、权重数据
"""
function getlbSortedData(p::Int)

    # 找到存在的 p 值和文件名
    filename = modiTgetFileName(p, lbnSortedP2FILEDict)
    # 点数
    nNodes  =   parse(Int, split(filename, ".")[2])
    # 预分配内存
    nodes   =   zeros(Float64, 3, nNodes)
    # 权重
    weights =   zeros(Float64, nNodes)  
    # 读取数据
    open(TargetDir*filename, "r") do file
        for ii in 1:nNodes
            contents = split(readline(file), " ")
            for ixyz in 1:3
                nodes[ixyz, ii] = parse(Float64, contents[ixyz])
            end
            weights[ii] = parse(Float64, contents[4])
        end
    end

    return nodes, weights

end


# """
# 简单地可视化三维散点
# """
# function visualizeScatterOnSphere(nodes::Matrix{FT}; filename = "nodes", reDir = "results/") where {FT<:Real}
    

#     fig = Plots.scatter(nodes[1,:], nodes[2,:], nodes[3,:], label = nothing, 
#                     markersize=1, color = :blue, colorbar = false, show = true)
#     nϕ  =   721
#     nθ  =   361
#     u = range(0, 2π, length = nϕ)
#     v = range(0,  π, length = nθ)
#     x = 0.999 .* (cos.(u) * sin.(v)')
#     y = 0.999 .* (sin.(u) * sin.(v)')
#     z = 0.999 .* repeat(cos.(v)',outer=[nϕ, 1])

#     fig = Plots.surface!(fig, x, y, z, color = :gold, dpi = 600)

#     Plots.plot!(fig, xlabel = "x", ylabel = "y", zlabel = "z",  dpi = 600)

#     !ispath(reDir) && mkpath(reDir)
#     for fmt in ["pdf", "svg"]# , "html"
#         Plots.savefig(fig, joinpath(reDir*  filename  * ".$fmt"))
#     end

#     return fig
# end

!ispath(TargetDir) && __init__()


# end # module