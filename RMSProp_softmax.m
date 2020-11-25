function[wt,t,s,accuracy]=RMSProp_softmax(train_X,train_y,a,w0,u,e,test_X,test_y)
%%
%使用RMSProp法进行softmax分类
% train_X指训练集数据,train_y指训练集对应标签，a指梯度下降学习率，u指动量参数，w0为初始值，test_X指测试集数据,test_y指测试集对应标签
% e是为维持数据稳定性而添加的常数
% 输出中wt指最终得出的直线参数，t指迭代次数，s指当前loss函数值，accuracy指准确率
% EXAMPLE：
% load('softmax.mat');
% [train_X,test_X,train_y,test_y]=train_test_split1(x,y,0.3)
%[wt,t,s,accuracy]=RMSProp_softmax(train_X,train_y,0.01,zeros(3,5),0.9,10^(-6),test_X,test_y)
l=length(train_y(1,:));
x1=[train_X;ones(1,l)];
z=w0*x1;
p=(exp(z)'./sum(exp(z))')';%计算p
s=-sum(sum(train_y.*log(p)));%计算第一次损失函数
t=0;
ttest=0;
v=zeros(3,5);%动量初值为0
loss=[];
while s>0.00001 && t<5000
        gammax=(p-train_y)*x1';%梯度计算
        vt=u*v+(1-u)*gammax.*gammax;%动量迭代
        v=vt;
        wt=w0-a*(gammax./(v.^(1/2)+e));%参数迭代 
        w0=wt;
        z=w0*x1;
        p=(exp(z)'./sum(exp(z))')';%更新p
        s=-sum(sum(train_y.*log(p)));%计算损失函数
        t=t+1;
        loss(t)=s;

end
%%
%绘图
subplot(211);
lm=length(test_y(1,:));
x2=[test_X;ones(1,lm)];
z2=w0*x2;
p2=(exp(z2)'./sum(exp(z2))')';
[~,index_test_output]=max(p2,[],1);
[~,index_test_y]=max(test_y,[],1);
error=sum(index_test_y~=index_test_output);
accuracy=1-error/lm;
ttt=1:1:45;
stem(ttt,index_test_output,'ro');
hold on;
stem(ttt,index_test_y,'bo'); 
set(gca,'position',[0.04 0.55 0.94 0.43])
hold off;
subplot(212);
tt = 1:10:t;
plot(tt,loss(tt),'b-','linewidth',1.5);
xlabel('epoch')
ylabel('loss')
set(gca,'position',[0.04 0.07 0.94 0.42])
axis([0 5000 0 100])
grid on;
