# character_detector
Project to build a cnn based character detector model

Disclaimer: It is a simple project done for learning CNN, So it sucks most of the time and no optimization of parameters are done. Lel

******************************************** REQUIREMENTS *******************************************************
*****************************************************************************************************************

Installed torch: http://torch.ch/docs/getting-started.html
Installed image package : $ luarocks install image
Installed pretty-nn package : $ luarocks install pretty-nn


************************************************* FILES **********************************************************
******************************************************************************************************************

1. conv.lua : Model description and Training is done using this file. Training dataset is originally taken from http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/. And it is then modified accordinaly, the one used for training and validation can be found here. 
Training: https://www.dropbox.com/s/rt0brmr9vvre1ga/newnewtrain.t7?dl=0
Validation: https://www.dropbox.com/s/a0l7xxoc00cs83l/newnewval.t7?dl=0

2. trained_net.pt: Already trained network

3. run.lua : For using an image 'test.png' as input to pretrained network.

4. temp.txt : random output result file obtained using the validation dataset.

5. test.png, trained_001_1.png : images for testing.


************************************************* NETWORK DESCRIPTION **********************************************
********************************************************************************************************************

There are 62 categories: 0-9, A-Z, a-z => (10+26+26=62). 0=>1, 1=>2,.....9=>10,A=>11, B=>12 ..........Z=>36, a=>37,....z=>62.

write some boring stuff about network zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
