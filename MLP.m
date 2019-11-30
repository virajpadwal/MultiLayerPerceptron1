clc;clear all; close all;

%input feature vector
x=[0.4 -0.7; 0.3 -0.5; 0.6 0.1; 0.2 0.4; 0.1 -0.2];

%target output
tr=[0.1 0.05 0.3 0.25 0.12];

%weight vectors between input and hidden layer
w=[0.1 0.2; 0.2 0.4];

%weight vector between hidden and output layer
v=[0.2 0.3];

oh=[0 0];

for epoch=1:10000
    
    
for k=1:5    % for all inputs 

    for j=1:2                
oh(1)=oh(1)+x(k,j)*w(1,j);  %output of hidden node 1
    end
    
    ojh(1)=sigmf(oh(1),[1 0]);
  oh(1)=0;


    for j=1:2                
oh(2)=oh(2)+x(k,j)*w(2,j);  %output of hidden node 1
    end
    
    oh(2)=0;
    
    ojh(2)=sigmf(oh(2),[1,0]);



    or(k)=sigmf(v(1)*ojh(1)+v(2)*ojh(2),[1,0]); %output of MLP
     if epoch==1000000
         disp(or(k));
     end


     n=(or(k)-tr(k))*or(k)*(1-or(k));  %constant 



  v(1)=v(1)-n*ojh(1);         % weights between j and r
  v(2)=v(2)-n*ojh(2);         % weights between j and r

for i=1:2
    for j=1:2
        w(i,j)=w(i,j)-n*v(j)*ojh(j)*(1-ojh(j))*x(k,i);
    end
end


end
end



