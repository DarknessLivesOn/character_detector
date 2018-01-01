require 'nn';
require 'pretty-nn'
cutorch = require 'cutorch'
require 'cunn'
net=nn.Sequential();

-- First Layer
net:add(nn.SpatialConvolution(3,6,5,5,1,1,2,2));
net:add(nn.ReLU());
net:add(nn.SpatialMaxPooling(2,2,2,2))

-- Second Layer
net:add(nn.SpatialConvolution(6,6,5,5,1,1,2,2));
net:add(nn.ReLU());
net:add(nn.SpatialMaxPooling(2,2,2,2));

-- Third Layer
net:add(nn.SpatialConvolution(6,6,5,5,1,1,2,2));
net:add(nn.ReLU());
net:add(nn.SpatialMaxPooling(2,2,2,2));


-- Output
net:add(nn.View(6*4*4))
net:add(nn.Linear(6*4*4,62));
net:add(nn.LogSoftMax())

net:cuda()

input=torch.rand(3,32,32)
input=input:cuda()
output=net:forward(input)
net:zeroGradParameters()

criterion=nn.ClassNLLCriterion():cuda()

loss=criterion:forward(output,3)   -- 3 is target value 
gradients=criterion:backward(output,3)  -- 3 is target value
gradInput=net:backward(input,gradients)

m=nn.SpatialConvolution(1,3,2,2)


train_set=torch.load("newnewtrain.t7")
test_set=torch.load("newnewval.t7")
classes={0,1,2,3,4,5,6,7,8,9,'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'}

setmetatable(train_set,
{
	__index=function(t,i)
		return {t.data[i],t.label[i]}
	end
});

train_set.data=train_set.data:cuda()
test_set.data=test_set.data:cuda()
function train_set:size()
	return self.data:size(1)
end

mean={}
stdv={}

for i=1,3 do
	mean[i]=train_set.data[{{},{i},{},{}}]:mean()
	train_set.data[{{},{i},{},{}}]:add(-mean[i])
	stdv[i]=train_set.data[{{},{i},{},{}}]:std()
	train_set.data[{{},{i},{},{}}]:div(stdv[i])
end


-- train the data

trainer=nn.StochasticGradient(net,criterion)
trainer.learningRate=0.001
trainer.maxIteration=25

trainer:train(train_set)

test_set.data=test_set.data
for i=1,3 do
	test_set.data[{{},{i},{},{}}]:add(-mean[i])
	test_set.data[{{},{i},{},{}}]:div(stdv[i])
end

-- validation 

plabel=torch.DoubleTensor(1631,1):zero()
for i=1,1631 do
	predicted=net:forward(test_set.data[i])
	predicted:exp()
	maxs,indices=torch.max(predicted,1)
	--plabel[i]=indices
end

-- just to write result on validation set

f=torch.DiskFile('temp.txt','w')
f:writeObject(plabel)
f:writeObject(test_set.label)
f:close()

-- saving the trained network
torch.save('trained_net.pt',net)






























