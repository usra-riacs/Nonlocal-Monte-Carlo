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
Epp   = zeros(100,num); 
P  = zeros(100,num);
Pd   = zeros(100,num);
deg  = zeros(100,num);
Sd  = zeros(100,num);
Pmin = zeros(100,num);
NN  = zeros(100,num);
Ns = zeros(100,num);


for nn = 1:4
for ii = 0:99
    try
    fid=fopen(['unique_r',num2str(nn-1),'_degE_s12_D64_L2048_beta2_Dc64_ins',num2str(ii,'%02d'),'.txt']); 
    fgetl(fid);
    st = fgetl(fid);
    kk = str2num(st);
    E(ii+1,nn) = kk(1);
    Ep(ii+1,nn) = kk(2);
    Epp(ii+1,nn) = kk(3);
    deg(ii+1,nn) = kk(4);
    P(ii+1,nn)  = kk(5);
    fgetl(fid);
    st = fgetl(fid);
    kk = str2num(st);
    Pd(ii+1,nn) = kk(1);
    Sd(ii+1,nn)  = kk(2);
    fgetl(fid);
    st = fgetl(fid);
    kk = str2num(st);
    Pmin(ii+1,nn)= kk(1);
    NN(ii+1,nn)  = kk(2);
    fgetl(fid);
    st = fgetl(fid);
    kk = str2num(st);
    Ns(ii+1,nn) = kk(1);
    fclose(fid);   
    catch
    end
end
end



for nn = 1:4
for ii = 0:99
    try
    nnn = nn+4;
    fid=fopen(['beta2p/unique_r',num2str(nn-1),'_degE_s16_D128_L2048_beta2p_Dc64_ins',num2str(ii,'%02d'),'.txt']); 
    fgetl(fid);
    st = fgetl(fid);
    kk = str2num(st);
    E(ii+1,nnn) = kk(1);
    Ep(ii+1,nnn) = kk(2);
    Epp(ii+1,nnn) = kk(3);
    deg(ii+1,nnn) = kk(4);
    P(ii+1,nnn)  = kk(5);
    fgetl(fid);
    st = fgetl(fid);
    kk = str2num(st);
    Pd(ii+1,nnn) = kk(1);
    Sd(ii+1,nnn)  = kk(2);
    fgetl(fid);
    st = fgetl(fid);
    kk = str2num(st);
    Pmin(ii+1,nnn)= kk(1);
    NN(ii+1,nnn)  = kk(2);
    fgetl(fid);
    st = fgetl(fid);
    kk = str2num(st);
    Ns(ii+1,nnn) = kk(1);
    fclose(fid);   
    catch
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