# specifies Fourier ONS on [0,1]
# Example 
function unit_fourier(k)
  sqrt_two = sqrt(2.0) 
  if k == 0
    return x -> 1.0
  elseif isodd(k)
    return x -> sqrt(2.0) * sinpi(float(k+1)*x)
  else
    return x -> sqrt(2.0) * cospi(float(k)*x)
  end
end

# specifies Fourier ONS on [0, 2pi]
function rad_fourier(k)
  if k == 0
    return x -> 1/ sqrt(2pi)
  elseif isodd(k)
    return x -> sin(float(k+1)*x / 2.0) / sqrt(pi)
  else
    return x -> cos(float(k)*x / 2.0) / sqrt(pi)
  end
end
