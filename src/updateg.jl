function updateg(Ex::AbstractVector, Ey::AbstractVector, rx::Real, ry::Real)
    gx = abs(rx) > 1.0e-12 ? Ex ./ rx : zero(Ex)
    gy = abs(ry) > 1.0e-12 ? Ey ./ ry : zero(Ex)

    return gx, gy
end

function updateg!(
        gx::Vector{T}, gy::Vector{T}, Ex::Vector{T}, Ey::Vector{T}, rx::T, ry::T
    ) where {T}
    gx .= abs(rx) > 1.0e-12 ? Ex ./ rx : zero(Ex)
    return gy .= abs(ry) > 1.0e-12 ? Ey ./ ry : zero(Ex)
end
