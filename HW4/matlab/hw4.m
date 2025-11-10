close all; clear all; clc
g = 9.80665;
dx = 2;
dy = 2;
CFL = 0.2;
dt = CFL*(min(dx,dy)/sqrt(g*10));
x = -300:dx:300;
y = -300:dy:300;
t = 0:dt:20;
s = 1:50:length(t);

% IC
H = 1;
L = 100;
IC.eta = @(X,Y) H.*exp(-18.*(X/L).^2).*exp(-18.*(Y/L).^2);
IC.U = @(X,Y) 0;
IC.V = @(X,Y) 0;

wave = RK2DH(@hconst,x,y,t,@Mirror,IC,s);

function out = hconst(x,y)
    out(1:length(y),1:length(x)) = 10;
end