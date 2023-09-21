E0   = zeros(100,1);
Ns0 = zeros(100,1);

for ii = 0:99
    fid=fopen([num2str(ii,'%02d'),'_sol.txt']);
    fgetl(fid);
    fgetl(fid);
    fgetl(fid);
    fgetl(fid);
    st = fgetl(fid);
    kk = str2num(st(end-5:end));
    Ns0(ii+1) = kk(1);
    fgetl(fid);
    st = fgetl(fid);
    kk = str2num(st(end-12:end));
    E0(ii+1) = kk(1);
    fclose(fid);
end


num = 4;
E   = zeros(100,num);
Ep   = zeros(100,num);
P  = zeros(100,num);
deg  = zeros(100,num);


for nn = 1:4
    for ii = 0:99
        try
            fid=fopen(['sol_r',num2str(nn-1),'_degE_s16_cf-20_D64_L2048_beta3_Dc8,8_bc0.25,0.5_ins_',num2str(ii,'%02d'),'.txt']);
            fgetl(fid);
            st = fgetl(fid);
            kk = str2num(st);
            E(ii+1,nn) = kk(1);
            Ep(ii+1,nn) = kk(2);
            deg(ii+1,nn) = kk(3);
            P(ii+1,nn)  = kk(4);
            fclose(fid);
        catch
            disp(['missing ins=', num2str(ii),' rotation = ',num2str(nn)])
        end
    end
end



[(0:99)',min(14*min((E-E0),10),[],2)]


[x,y]=min(E,[],2);
for ii=1:100
    jj = find( abs(E(ii,:) -x(ii)) < 1e-8);
    degm(ii)=max(deg(ii,jj));
end
median(degm)
sum(degm<0)
degm(degm<0)=4e18;
median(degm)