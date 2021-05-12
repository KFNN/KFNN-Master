clear all
flag=1;
N=2000;
D=10
R1=0.0702*eye(D);
R2=0.0684*eye(D);
%n=2;
fai=eye(D);
%L=2;
randn('seed',10);%10
v1=sqrt(R1)*randn(D,N+10);%构造v1(t)函数
randn('seed',99);%99
v2=sqrt(R2)*randn(D,N+10);%构造v2(t)函数


%gama=0;
H=eye(D);%H1=H2=H
x(:,1)=zeros(D,1);
y1(:,1)=H*x(:,1)+v1(:,1);
y2(:,1)=H*x(:,1)+v2(:,1);
%y3(1)=H*x(:,1)+v2(1);
for i=2:N+8
    x(:,i)=fai*x(:,i-1);
    y1(:,i)=H*x(:,i)+v1(:,i);
    y2(:,i)=H*x(:,i)+v2(:,i);
    %yc(:,i)=[y1(:,i);y2(i)];%用于集中式融合
end

%---------NO.1-----------------
xxjian1(:,1)=zeros(D,1);
pp1(:,:,1)=eye(D)+0.1;
for i=1:N
    kp1(:,:,i)=fai*pp1(:,:,i)*H'*inv(H*pp1(:,:,i)*H'+R1);%
    pusip1(:,:,i)=fai-kp1(:,:,i)*H;  
    pp1(:,:,i+1)=pusip1(:,:,i)*pp1(:,:,i)*pusip1(:,:,i)'+kp1(:,:,i)*R1*kp1(:,:,i)';
end
for  i=1:N
    xxjian1(:,i+1)=pusip1(:,:,N)*xxjian1(:,i)+kp1(:,:,N)*y1(:,i);%稳态预报器
end
trpp1=trace(pp1(:,:,N));
%---------NO.2-----------------
xxjian2(:,1)=zeros(D,1);
pp2(:,:,1)=eye(D)+0.1;
for i=1:N
    kp2(:,:,i)=fai*pp2(:,:,i)*H'*inv(H*pp2(:,:,i)*H'+R2);%
    pusip2(:,:,i)=fai-kp2(:,:,i)*H;  
    pp2(:,:,i+1)=pusip2(:,:,i)*pp2(:,:,i)*pusip2(:,:,i)'+kp2(:,:,i)*R2*kp2(:,:,i)';
end
for  i=1:N
    xxjian2(:,i+1)=pusip2(:,:,N)*xxjian2(:,i)+kp2(:,:,N)*y2(:,i);%稳态预报器
end
trpp2=trace(pp2(:,:,N));



a = (1/trpp1 + 1/trpp2)^(-1)
w1 = a/trpp1
w2 = a/trpp2
all = w1+w2
