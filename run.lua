require 'nn'
require 'pretty-nn'
require 'image'

-- load the image
x=image.load('test.png')
input=image.scale(x,32,32)
net=torch.load('trained_net.pt')
output=net:forward(input)
output:exp()
maxs,indices=torch.max(output,1)
print(indices)
