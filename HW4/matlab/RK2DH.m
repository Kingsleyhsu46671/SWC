function data = RK2DH(h0,x,y,t,BC,IC,savet)
%---------------------------------------------------------------------------------
% This function uses SSP-RK to calculate LSWE
% h(x,y)                   = water depth
% x/y                      = spatial grid points in x/y direction
% t                        = calculation time range
% BC(x,y)                  = boundary conditions using ghost cells
% IC{"eta","U","V"}(X,Y)   = structure of initial conditions for the free surface
%                            elevation and x/y-direction mean flow velocity
% savet                    = User-specified data-saving time steps
%---------------------------------------------------------------------------------
g = 9.80665; % gravitational constant
xn = [3*x(1)-2*x(2) 2*x(1)-x(2) x 2*x(end)-x(end-1) 3*x(end)-2*x(end-1)];
yn = [3*y(1)-2*y(2) 2*y(1)-y(2) y 2*y(end)-y(end-1) 3*y(end)-2*y(end-1)];
sX = length(xn); % N+4 nodes
sY = length(yn);
h = h0(xn,yn);
[X,Y] = meshgrid(xn,yn);

a = repmat(abs(t(2)-t(1))./abs(x(2)-x(1))/12,sX,1);
A = (spdiags([a -8*a 8*a -a],[2 1 -1 -2],sX,sX)).';
C = (spdiags([a -8*a 8*a -a],[2 1 -1 -2],sX,sX).*g).';
b = repmat(abs(t(2)-t(1))./abs(y(2)-y(1))/12,sY,1);
B = spdiags([b -8*b 8*b -b],[2 1 -1 -2],sY,sY);
D = spdiags([b -8*b 8*b -b],[2 1 -1 -2],sY,sY).*g;

% BC Mirror
[Pe1,Pe2,Pu1,Pu2,Pv1,Pv2] = BC(xn,yn);

% IC
eta0 = Pe1*IC.eta(X,Y)*Pe2;
U0 = Pu1*IC.U(X,Y)*Pu2;
V0 = Pv1*IC.V(X,Y)*Pv2;

savet = [savet length(t)]; % Save the indicated time steps and the last step
count = savet(1)-1;
eta = zeros(length(x),length(y),length(savet));
U = zeros(length(x),length(y),length(savet));
V = zeros(length(x),length(y),length(savet));
for i = 1:length(savet)
%     fprintf('count = %d ',count);
%     t1 = tic;
    while count > 0
        % f* <- f^n
        eta2 = Pe1*(eta0 + (U0.*h)*A + B*(V0.*h))*Pe2;
        U2 = Pu1*(U0 + eta0*C)*Pu2;
        V2 = Pv1*(V0 + D*eta0)*Pv2;
        % f** <- f*
        eta1 = Pe1*(0.75*eta0 + 0.25*eta2 + 0.25*(U2.*h)*A + 0.25*B*(V2.*h))*Pe2;
        U1 = Pu1*(0.75*U0 + 0.25*U2 + 0.25*eta2*C)*Pu2;
        V1 = Pv1*(0.75*V0 + 0.25*V2 + 0.25*D*eta2)*Pv2;
        % f^(n+1) <- f**
        eta0 = Pe1*((1/3)*eta0 + (2/3)*eta1 + (2/3)*(U1.*h)*A + (2/3)*B*(V1.*h))*Pe2;
        U0 = Pu1*((1/3)*U0 + (2/3)*U1 + (2/3)*eta1*C)*Pu2;
        V0 = Pv1*((1/3)*V0 + (2/3)*V1 + (2/3)*D*eta1)*Pv2;
        count = count-1;
    end
%     fprintf('t = %.3f\n',toc(t1));
    eta(:,:,i) = eta0(3:end-2,3:end-2)';
    U(:,:,i) = U0(3:end-2,3:end-2)';
    V(:,:,i) = V0(3:end-2,3:end-2)';
    if i ~= length(savet) 
        count = savet(i+1)-savet(i); 
    end
end
data = struct('eta',eta,'U',U,'V',V,'t',t(savet),'X',X(3:end-2,3:end-2),'Y',Y(3:end-2,3:end-2));
end