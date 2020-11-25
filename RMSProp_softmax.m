function[wt,t,s,accuracy]=RMSProp_softmax(train_X,train_y,a,w0,u,e,test_X,test_y)
%%
%ʹ��RMSProp������softmax����
% train_Xָѵ��������,train_yָѵ������Ӧ��ǩ��aָ�ݶ��½�ѧϰ�ʣ�uָ����������w0Ϊ��ʼֵ��test_Xָ���Լ�����,test_yָ���Լ���Ӧ��ǩ
% e��Ϊά�������ȶ��Զ���ӵĳ���
% �����wtָ���յó���ֱ�߲�����tָ����������sָ��ǰloss����ֵ��accuracyָ׼ȷ��
% EXAMPLE��
% load('softmax.mat');
% [train_X,test_X,train_y,test_y]=train_test_split1(x,y,0.3)
%[wt,t,s,accuracy]=RMSProp_softmax(train_X,train_y,0.01,zeros(3,5),0.9,10^(-6),test_X,test_y)
l=length(train_y(1,:));
x1=[train_X;ones(1,l)];
z=w0*x1;
p=(exp(z)'./sum(exp(z))')';%����p
s=-sum(sum(train_y.*log(p)));%�����һ����ʧ����
t=0;
ttest=0;
v=zeros(3,5);%������ֵΪ0
loss=[];
while s>0.00001 && t<5000
        gammax=(p-train_y)*x1';%�ݶȼ���
        vt=u*v+(1-u)*gammax.*gammax;%��������
        v=vt;
        wt=w0-a*(gammax./(v.^(1/2)+e));%�������� 
        w0=wt;
        z=w0*x1;
        p=(exp(z)'./sum(exp(z))')';%����p
        s=-sum(sum(train_y.*log(p)));%������ʧ����
        t=t+1;
        loss(t)=s;

end
%%
%��ͼ
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
