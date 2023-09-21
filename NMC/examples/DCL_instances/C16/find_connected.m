function CM = find_connected(ins, Nx, Ny, Ns)

Nx = round(Nx);
Ny = round(Ny);
Ns = round(Ns);

J = importdata([num2str(ins,'%02d'),'.txt']);

l = size(J,1);

CM = zeros(Ny*2-1, Nx*2-1);

for ii = 1:l
    Jij = J(ii,:);
    if abs(Jij(3)) > 1e-10
        x = round(Jij(1));
        y = round(Jij(2));
        x2 = floor(x./(Ns*Nx))+1;
        x1 = mod(x, Ns*Nx);
        x1 = floor(x1./(Ns))+1;
        y2 = floor(y./(Ns*Nx))+1;
        y1 = mod(y, Ns*Nx);
        y1 = floor(y1./(Ns))+1;
        
        if (x1 == y1) && (x2==y2)
            CM(2*x1-1,2*x2-1) = 1;
        elseif (x1 > y1) && (x2==y2)
            CM(2*x1-2,2*x2-1) = -1;
        elseif (x1 < y1) && (x2==y2)
            CM(2*x1,2*x2-1) = -1;
        elseif (x1 == y1) && (x2>y2)
            CM(2*x1-1,2*x2-2) = -1;
        elseif (x1 == y1) && (x2<y2)
            CM(2*x1-1,2*x2) = -1;
        else
            disp('error')
        end
    end
end
imagesc(CM)
colorbar

end

