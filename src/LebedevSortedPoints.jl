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

"""
多项式阶数 p 到对应采样点文件、p到点数、点数到p的字典
"""
function lbnSorted2fnDictConstruct(filedirs::String = TargetDir)
    # 读取所有文件名
    sphlebeFileNames     =   readdir(filedirs)
    # 初始化 t 值 → 文件名的字典
    t2fnDict    =   Dict{Int, String}()
    # p → n 字典
    p2nDict     =   Dict{Int, Int}()
    # n → p 字典
    n2pDict     =   Dict{Int, Int}()
    # 循环写入字典
    for filename in sphlebeFileNames
        contents = split(filename, ".")
        p    =   parse(Int, contents[1])
        n    =   parse(Int, contents[2])
        t2fnDict[p] =   filename
        p2nDict[p] = n
        n2pDict[n] = p
    end

    return t2fnDict, p2nDict, n2pDict
end

const lbnSortedP2FILEDict, p2nDict, n2pDict   =   lbnSorted2fnDictConstruct()


"""
返回排序后的点、权重数据
"""
function getlbSortedData(p::Int; FT = Precision.FT)

    # 找到存在的 p 值和文件名
    filename = modiTgetFileName(p, lbnSortedP2FILEDict)
    # 点数
    nNodes  =   parse(Int, split(filename, ".")[2])
    # 预分配内存
    nodes   =   zeros(FT, 3, nNodes)
    # 权重
    weights =   zeros(FT, nNodes)  
    # 读取数据
    open(TargetDir*filename, "r") do file
        for ii in 1:nNodes
            contents = split(readline(file), " ")
            for ixyz in 1:3
                nodes[ixyz, ii] = parse(FT, contents[ixyz])
            end
            weights[ii] = parse(FT, contents[4])
        end
    end

    return nodes, weights

end

using MoM_Kernels:octreeXWNCal

function get_t_nodes(t; FT = Precision.FT)

    p = 2t+1

    nodes = if p <= maximum(keys(p2nDict))
        getlbSortedData(p; FT = FT)[1]
    else
        # θ方向
        Xcosθs, Wθs   =   octreeXWNCal(one(FT), -one(FT), t, :glq)
        # 将θ方向高斯-勒让德求积坐标从 [1.,-1.] 转换到 [0,π]
        Xθs      =   acos.(Xcosθs)
        # ϕ方向
        Xϕs, Wϕs = octreeXWNCal(zero(FT), convert(FT, 2π), t, :uni)
        
        # 将数据保存在 levelsPoles 中，按照 θ 方向连续的顺序，将所有采样点信息保存为一向量
        # 计算所有极子的信息
        reduce(hcat, [r̂θϕInfo{FT}(θ, ϕ).r̂ for ϕ in Xϕs for θ in Xθs])
    end

    return nodes

end

!ispath(TargetDir) && __init__()
