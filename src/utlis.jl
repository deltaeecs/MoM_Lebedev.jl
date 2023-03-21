# 本文件用于读取spherical t-Design 和 lebedev nodes 文件


"""
将球面散点转化为角度信息
"""
function nodes2Poles(nodes::Matrix{FT}) where {FT}
    # 点数
    nr  =   size(nodes, 2)
    # 初始化
    r̂sθsϕs  =   Vector{r̂θϕInfo{FT}}(undef, nr)

    # 循环计算
    for ii in 1:nr
        r̂sθsϕs[ii] = r̂θϕInfo{FT}(view(nodes, :, ii))
    end

    return r̂sθsϕs

end

"""
在两层采样点之间初始化插值矩阵（采用反距离权重）
输入
tNodes::Matrix{T} 大小为 (3, nt) 的矩阵，表示 nt 个插值点集
pNodes::Matrix{T} 大小为 (3, np) 的矩阵，表示 np 个待插值点集
nInterp::Int, 插值点数
"""
function interpWeightsInitial(tNodes::Matrix{T}, pNodes::Matrix{T}; nInterp::Integer=10) where {T}
    # 点数
    ptNodes =   size(pNodes, 2)

    # 最近 nInterp 个结点计算
    kdtree  =   KDTree(tNodes)
    # 最近的 nInterp 个点的id, 笛卡尔距离
    idxs, dists =   knn(kdtree, pNodes, nInterp, true)
    # 
    idxs    =   hcat(idxs...)
    dists   =   hcat(dists...)

    # 转换为球面距离
    sphdists    =   2asin.(dists/2)

    # 反距离插值权重
    interpWeits =   1 ./ sphdists
    for i in 1:size(interpWeits, 2)
        interpWeits[:, i] ./= sum(interpWeits[:, i])
    end
    interpWeitsDiagonal =   deepcopy(interpWeits)
    # nan值由自插值引起，对同向 ( θ → θ, ϕ → ϕ ) 插值，变为1，对异向( θ → ϕ, ϕ → θ )插值变为 0 
    for ij in eachindex(interpWeits)
        isnan(interpWeits[ij]) && begin 
            interpWeits[ij] = 1
            interpWeitsDiagonal[ij] = 0
        end
    end


    raws = repeat(1:ptNodes; inner = nInterp)

    # 同向 ( θ → θ, ϕ → ϕ ) 插值矩阵
    @views interpWeitsCSC  =   sparse(raws, idxs[:], interpWeits[:])
    dropzeros!(interpWeitsCSC)
    # 异向( θ → ϕ, ϕ → θ ) 插值矩阵
    @views interpWeitsCSCDiagonal = sparse(raws, idxs[:], interpWeitsDiagonal[:])
    dropzeros!(interpWeitsCSCDiagonal)
    # 总的插值矩阵
    interpWeits = [ interpWeitsCSC  interpWeitsCSCDiagonal;
                    interpWeitsCSCDiagonal  interpWeitsCSC]

    return interpWeits
end
