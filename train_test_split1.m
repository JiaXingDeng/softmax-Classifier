function [train_X,test_X,train_y,test_y]=train_test_split1(x,y,r)
%%
%�ú������ڻ���softmax�����ѵ�����Ͳ��Լ�
% xָ����,yָ��ǩ��rָ���Լ�ռ���������ı���
% ����У�train_X��train_y��ʾѵ���������ݺͱ�ǩ��test_X��test_y��ʾ���Լ������ݺͱ�ǩ
% EXAMPLE��
% load('softmax.mat');
% [train_X,test_X,train_y,test_y]=train_test_split1(x,y,0.3)
x=mapminmax(x,0,1);
data=[x',y'];
trainindex = crossvalind('HoldOut',size(data,1),r);
testindex = ~trainindex;
traindata = data (trainindex,:)';
testdata = data (testindex,:)';
train_X = traindata(1:4,:);
test_X = testdata(1:4,:);
train_y = traindata(5:7,:);
test_y = testdata(5:7,:);