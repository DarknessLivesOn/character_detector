require 'nn'
require 'pretty-nn'
require 'image'
require 'cunn'
require 'cutorch'

-- load the image
x=image.load('test.png')
input=image.scale(x,32,32)
input=input:cuda()
net=torch.load('trained_net.pt')
output=net:forward(input)
output:exp()
maxs,indice=torch.max(output,1)
indices=indice[1]
if indices==1 then
    print(0)
elseif indices==2 then
    print(1)
elseif indices==3 then
    print(2)
elseif indices==4 then
    print(3)
elseif indices==5 then
    print(4)
elseif indices==6 then
    print(5)
elseif indices==7 then
    print(6)
elseif indices==8 then
    print(7)
elseif indices==9 then
    print(8)
elseif indices==10 then
    print(9)
elseif indices==11 then
    print('A')
elseif indices==12 then
    print('B')
elseif indices==13 then
    print('C')
elseif indices==14 then
    print('D')
elseif indices==15 then
    print('E')
elseif indices==16 then
    print('F')
elseif indices==17 then
    print('G')
elseif indices==18 then
    print('H')
elseif indices==19 then
    print('I')
elseif indices==20 then
    print('J')
elseif indices==21 then
    print('K')
elseif indices==22 then
    print('L')
elseif indices==23 then
    print('M')
elseif indices==24 then
    print('N')
elseif indices==25 then
    print('O')
elseif indices==126 then
    print('P')
elseif indices==27 then
    print('Q')
elseif indices==28 then
    print('R')
elseif indices==29 then
    print('S')
elseif indices==30 then
    print('T')
elseif indices==31 then
    print('U')
elseif indices==32 then
    print('V')
elseif indices==33 then
    print('W')
elseif indices==34 then
    print('X')
elseif indices==35 then
    print('Y')
elseif indices==36 then
    print('Z')
elseif indices==37 then
    print('a')
elseif indices==38 then
    print('b')
elseif indices==39 then
    print('c')
elseif indices==40 then
    print('d')
elseif indices==41 then
    print('e')
elseif indices==42 then
    print('f')
elseif indices==43 then
    print('g')
elseif indices==44 then
    print('h')
elseif indices==45 then
    print('i')
elseif indices==46 then
    print('j')
elseif indices==47 then
    print('k')
elseif indices==48 then
    print('l')
elseif indices==49 then
    print('m')
elseif indices==50 then
    print('n')
elseif indices==51 then
    print('o')
elseif indices==52 then
    print('p')
elseif indices==53 then
    print('q')
elseif indices==54 then
    print('r')
elseif indices==55 then
    print('s')
elseif indices==56 then
    print('t')
elseif indices==57 then
    print('u')
elseif indices==58 then
    print('v')
elseif indices==59 then
    print('w')
elseif indices==60 then
    print('x')
elseif indices==61 then
    print('y')
elseif indices==62 then
    print('z')
end


