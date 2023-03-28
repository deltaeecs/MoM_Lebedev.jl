using MoM_Basics:r̂θϕInfo
using MoM_Kernels:PolesInfo, GLPolesInfo, InterpInfo, truncationLCal, levelIntegralInfoCal

"""
多极子的极信息，即角谱空间采样信息，基于 Lebedev 采样点
Wθϕs    ::Vector{FT}， 权重向量
r̂sθsϕs  ::Vector{r̂θϕInfo{FT}}， 球面采样信息向量
"""
struct LbPolesInfo{FT<:Real} <:PolesInfo{FT}
    Wθϕs::Vector{FT}
    r̂sθsϕs::Vector{r̂θϕInfo{FT}}
end

"""
计算八叉树的积分相关信息，包括截断项、各层积分点和求积权重数据
输入:
levelCubeEdgel::FT,  层盒子边长, 一般叶层为0.25λ，其中 λ 为区域局部波长。
返回值
L           ::IT， 层 截断项
levelsPoles ::Vector{GLPolesInfo{FT}}，从叶层到第 “2” 层的角谱空间采样信息
"""
function MoM_Kernels.levelIntegralInfoCal(levelCubeEdgel::FT, ::Val{:LbTrained1Step}) where{FT<:Real}
    ## 计算截断项
    truncL  =   truncationLCal(levelCubeEdgel)
    if 2truncL+1 < maximum(keys(p2nDict))
        # 读取 球 t 采样点信息并返回更新的 truncL
        nodes, weights  =   getlbSortedData(2truncL + 1)

        # 创建Poles实例保存
        r̂sθsϕs  =   nodes2Poles(nodes)
        poles   =   LbPolesInfo{FT}(weights, r̂sθsϕs)
            
        return truncL, poles
    else
        @warn "本层大小超出 Lebedev 求积极限，换回球面高斯求积。"
        return levelIntegralInfoCal(levelCubeEdgel::FT, Val(:Lagrange2Step))
    end
end

"""
保存整个方向的稀疏插值矩阵，存储形式定为稠密阵，因为稀疏矩阵元素密度较高时不如直接计算稠密阵乘积
θϕCSC   ::AbstractMatrix{FT} 插值矩阵，用于左乘本层多极子矩阵插值
θϕCSCT  ::AbstractMatrix{FT} 插值矩阵的转置，用于左乘本层多极子矩阵反插值
"""
mutable struct LbTrainedInterp1tepInfo{IT, FT<:Real} <: InterpInfo{IT, FT}
    θϕCSC    ::SparseMatrixCSC{FT, IT}
    θϕCSCT   ::SparseMatrixCSC{FT, IT}
    LbTrainedInterp1tepInfo{IT, FT}() where {IT, FT<:Real}  =   new{IT, FT}()
    LbTrainedInterp1tepInfo{IT, FT}(θϕCSC::AbstractArray, θϕCSCT::AbstractArray) where {IT, FT<:Real}  =   new{IT, FT}(θϕCSC, θϕCSCT)
end

LbTrainedInterp1tepInfo(θϕCSC::AbstractArray, θϕCSCT::AbstractArray) = LbTrainedInterp1tepInfo{Int, eltype(θϕCSC)}(θϕCSC, θϕCSCT)

function MoM_Kernels.get_Interpolation_Method(method::Val{:LbTrained1Step})
    if method == Val(:LbTrained1Step)
        return LbTrainedInterp1tepInfo
    else
        throw("插值方法选择出错")
    end
end

"""
带参数的构造函数
"""
function LbTrainedInterp1tepInfo(pk::Int, pt::Int, FT = Precision.FT)

    w   =   load("deps/InterpolationWeights/$(pk)to$(pt).jld2", "data")
    θϕCSC   =   convert(SparseMatrixCSC{FT, Int}, w)
    θϕCSCT  =   sparse(transpose(θϕCSC))

    return LbTrainedInterp1tepInfo{Int, FT}(θϕCSC, θϕCSCT)

end

"""
球 t 设计一步插值
"""
function MoM_Kernels.interpolate(weights::LbTrainedInterp1tepInfo{IT, FT}, data::AbstractArray) where {IT, FT}
    target  =   zeros(eltype(data), size(weights.θϕCSC, 1))
    interpolate!(target, weights, data)
    reshape(target, :, 2)
end

"""
球 t 设计一步反插值
"""
function MoM_Kernels.anterpolate(weights::LbTrainedInterp1tepInfo{IT, FT}, data::AbstractArray) where {IT, FT}
    target  =   zeros(eltype(data), size(weights.θϕCSCT, 1))
    anterpolate!(target, weights, data)
    reshape(target, :, 2)
end

"""
球 t 设计一步插值
"""
function MoM_Kernels.interpolate!(target::AbstractArray, weights::LbTrainedInterp1tepInfo{IT, FT}, data::AbstractArray) where {IT, FT}
    mul!(reshape(target, :), weights.θϕCSC, reshape(data, :))
    return target
end

"""
球 t 设计一步反插值
"""
function MoM_Kernels.anterpolate!(target::AbstractArray, weights::LbTrainedInterp1tepInfo{IT, FT}, data::AbstractArray) where {IT, FT}
    mul!(reshape(target, :), weights.θϕCSCT, reshape(data, :))
    return target
end

function MoM_Kernels.interpolationCSCMatCal(tLevelPoles::LbPolesInfo{FT}, kLevelPoles::LbPolesInfo{FT}, ::IT=8) where {IT<:Integer, FT<:Real}
    # 多极子数
    nt = length(tLevelPoles.Wθϕs)
    nk = length(kLevelPoles.Wθϕs)
    # 多项式阶数
    pk = n2pDict[nk]
    pt = n2pDict[nt]
    # 插值矩阵
    LbTrainedInterp1tepInfo(pk, pt)
end


function MoM_Kernels.interpolationCSCMatCal(tLevelPoles::GLPolesInfo{FT}, kLevelPoles::LbPolesInfo{FT}, ::IT=8) where {IT<:Integer, FT<:Real}
    # 多极子数
    nk = length(kLevelPoles.Wθϕs)
    # 多项式阶数
    pk = n2pDict[nk]
    pt = 2(length(tLevelPoles.Xθs) - 1) + 1
    # 插值矩阵
    LbTrainedInterp1tepInfo(pk, pt)
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
