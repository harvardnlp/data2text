--[[ Recursively call `func()` on all tensors within `out`. ]]
local function recursiveApply(out, func, ...)
  local res
  if torch.type(out) == 'table' then
    res = {}
    for k, v in pairs(out) do
      res[k] = recursiveApply(v, func, ...)
    end
    return res
  end
  if torch.isTensor(out) then
    res = func(out, ...)
  else
    res = out
  end
  return res
end

--[[ Recursively call `clone()` on all tensors within `out`. ]]
local function recursiveClone(out)
  if torch.isTensor(out) then
    return out:clone()
  else
    local res = {}
    for k, v in ipairs(out) do
      res[k] = recursiveClone(v)
    end
    return res
  end
end

local function recursiveSet(dst, src)
  if torch.isTensor(dst) then
    dst:set(src)
  else
    for k, _ in ipairs(dst) do
      recursiveSet(dst[k], src[k])
    end
  end
end

--[[ Clone any serializable Torch object. ]]
local function deepClone(obj)
  local mem = torch.MemoryFile("rw"):binary()
  mem:writeObject(obj)
  mem:seek(1)
  local clone = mem:readObject()
  mem:close()
  return clone
end

--[[
Reuse Tensor storage and avoid new allocation unless any dimension
has a larger size.

Parameters:

  * `t` - the tensor to be reused
  * `sizes` - a table or tensor of new sizes

Returns: a view on zero-tensor `t`.

--]]
local function reuseTensor(t, sizes)
  assert(t ~= nil, 'tensor must not be nil for it to be reused')

  if torch.type(sizes) == 'table' then
    sizes = torch.LongStorage(sizes)
  end

  return t:resize(sizes):zero()
end

--[[
Reuse all Tensors within the table with new sizes.

Parameters:

  * `tab` - the table of tensors
  * `sizes` - a table of new sizes

Returns: a table of tensors using the same storage as `tab`.

--]]
local function reuseTensorTable(tab, sizes)
  local newTab = {}

  for i = 1, #tab do
    local size = sizes -- if just one size
    if torch.type(sizes) == 'table' and torch.type(sizes[1]) == 'table' then
        size = sizes[i]
    end
    table.insert(newTab, reuseTensor(tab[i], size))
  end

  return newTab
end

--[[
Initialize a table of tensors with the given sizes.

Parameters:

  * `tab` - the table of tensors
  * `proto` - tensor to be clone for each index
  * `sizes` - a table of new sizes

Returns: an initialized table of tensors.

--]]
local function initTensorTableOrig(size, proto, sizes)
  local tab = {}

  local base = reuseTensor(proto, sizes)

  for _ = 1, size do
    table.insert(tab, base:clone())
  end

  return tab
end

local function initTensorTable(size, proto, sizes)
  local tab = {}

  --local base = reuseTensor(proto, sizes)

  for i = 1, size do
    local size = sizes -- if just one size
    if torch.type(sizes) == 'table' and torch.type(sizes[1]) == 'table' then
        size = sizes[i]
    end
    table.insert(tab, proto:clone():resize(torch.LongStorage(size)):zero()) --base:clone())
  end

  return tab
end

--[[
Copy tensors from `src` reusing all tensors from `proto`.

Parameters:

  * `proto` - the table of tensors to be reused
  * `src` - the source table of tensors

Returns: a copy of `src`.

--]]
local function copyTensorTable(proto, src)
  local tab = reuseTensorTable(proto, src[1]:size())

  for i = 1, #tab do
    tab[i]:copy(src[i])
  end

  return tab
end

local function copyTensorTableHalfRul(proto, src)
    local tab = {}
    assert(#proto == #src)
    for i = 1, #proto do
        proto[i]:resize(src[i]:size(1), proto[i]:size(2)):zero() -- unnecessary
        if src[i]:size(2) < proto[i]:size(2) then
            assert(proto[i]:size(2) == src[i]:size(2)*2)
            proto[i]:narrow(2,1,src[i]:size(2)):copy(src[i])
        elseif src[i]:size(2) == proto[i]:size(2) then
            proto[i]:copy(src[i])
        else
            assert(false)
        end
        table.insert(tab, proto[i])
    end
    return tab
end

local function copyTensorTableHalf(proto, src)
    local tab = {}
    assert(#proto == #src)
    for i = 1, #proto do
        proto[i]:resize(src[i]:size(1), proto[i]:size(2)):zero() -- unnecessary
        if src[i]:size(2) < proto[i]:size(2) then
            assert(proto[i]:size(2) == src[i]:size(2)*2)
            proto[i]:narrow(2,1,src[i]:size(2)):copy(src[i])
            proto[i]:narrow(2,src[i]:size(2)+1,src[i]:size(2)):copy(src[i])
        elseif src[i]:size(2) == proto[i]:size(2) then
            proto[i]:copy(src[i])
        else
            assert(false)
        end
        table.insert(tab, proto[i])
    end
    return tab
end


return {
  recursiveApply = recursiveApply,
  recursiveClone = recursiveClone,
  recursiveSet = recursiveSet,
  deepClone = deepClone,
  reuseTensor = reuseTensor,
  reuseTensorTable = reuseTensorTable,
  initTensorTable = initTensorTable,
  copyTensorTable = copyTensorTable,
  copyTensorTableHalf = copyTensorTableHalf
}
