Em0   = zeros(100,1);
Nspins0 = zeros(100,1);

for ii = 0:99
    fid=fopen([num2str(ii,'%02d'),'_sol.txt']);
    fgetl(fid);
    fgetl(fid);
    fgetl(fid);
    fgetl(fid);
    st = fgetl(fid);
    kk = str2num(st(end-5:end));
    Nspins0(ii+1) = kk(1);
    fgetl(fid);
    st = fgetl(fid);
    kk = str2num(st(end-12:end));
    Em0(ii+1) = kk(1);
    fclose(fid);
end

E   = zeros(100,4);
Ep   = zeros(100,4);
P  = zeros(100,4);
deg  = zeros(100,4);

for ii = 0:99
    for rr = 1:4
        %fid = fopen(['aferro/aferro_sol_r',num2str(rr-1),'_degE_s16_cf-20_D64_L2048_beta3_Dc8,8_bc0.25,0.5_ins_',num2str(ii,'%02d'),'.txt']);
        %if fid == -1
           fid = fopen(['aferro/aferro_sol_r',num2str(rr-1),'_degE_s12_cf-20_D64_L2048_beta2_Dc8,8_bc0.25,0.5_ins_',num2str(ii,'%02d'),'.txt']);
        %end
        fgetl(fid);
        st = fgetl(fid);
        kk = str2num(st);
        E(ii+1,rr)  = kk(1);
        Ep(ii+1,rr) = kk(2);
        deg(ii+1,rr)= kk(3);
        P(ii+1,rr)  = kk(4);
        fclose(fid);
    end
end

for ii = 0:99
    for rr = 1:4
        %fid = fopen(['aferro/aferro_sol_r',num2str(rr-1),'_degE_s16_cf-20_D64_L2048_beta3_Dc8,8_bc0.25,0.5_ins_',num2str(ii,'%02d'),'.txt']);
        %if fid == -1
           fid = fopen(['aferro/aferro_sol_r',num2str(rr-1),'_degE_s14_cf-20_D64_L2048_beta2_Dc8,8_bc0.25,0.5_ins_',num2str(ii,'%02d'),'.txt']);
        %end
        fgetl(fid);
        st = fgetl(fid);
        kk = str2num(st);
        E(ii+1,rr+4)  = kk(1);
        Ep(ii+1,rr+4) = kk(2);
        deg(ii+1,rr+4)= kk(3);
        P(ii+1,rr+4)  = kk(4);
        fclose(fid);
    end
end

[(0:99)',min(14*min((E-Em0),10),[],2)]

[x,y]=min(E,[],2);
for ii=1:100
    jj = find( abs(E(ii,:) -x(ii)) < 1e-8);
    degm(ii)=max(deg(ii,jj));
end
median(degm)
sum(degm<0)
degm(degm<0)=4e18;
median(degm)