//    Proposta de modelo para producao de celulase
//    Angarita et al. 2015
clear
// Experimental data
// 10% solidos e 10FPU/g   [tempo G G2 X]
dados=[1 4.30 0.2295 2.2004;
2 6.83 0.2981 3.2317;
6 15.05 0.4533 5.0717;
8 20.49 0.3853 4.6835;
12 21.02 0.5157 5.6440;
24 23.31 0.4363 9.6220;
48 35.19 0.7423 8.1657;
72 37.86 0.5667 7.3837;
96 38.5  0.3060 7.9050];

// 15% solidos e 10FPU/g Xilose 10 g/L [tempo G G2 X]
dados1=[1 6.32 0.00 9.91
2 9.64 0.00 10.84
6 18.84 0.00 13.33
12 28.31 0.73 15.22
24 27.93 0.72 19.69
48 51.82 1.71 17.71
72 57.96 1.92 17.81];

// Set data for the model
S0=100.0;               // Solid loading (Pretreated sugarcane straw)
Cellulose=0.66;         // Cellulose content (0-1, dimensionless)
Hemicellulose=0.083;    // Hemicellulose content (0-1, dimensionless)
Lignin=0.257;           // Lignin content (0-1, dimensionless)
Glucose=0;              // Glucose concentration (g/L)
Cellobiose=0;           // Cellobiose concentration (g/L)
Xylose=0;               // Xylose concentration (g/L)

alfa=1.0;
E_T=0.175;              // Total enzyme concentration (g/L)

// Parameters 
k_1r=0.177;
k_2r=8.81;
k_3r=201.0;
k_4r=16.34;
k_11G2=0.402;
k_11G=2.71;
k_11X=2.15;
k_21G2=119.6;
k_21G=4.69;
k_21X=0.095;
k_3M=26.6;
k_31G=11.06;
k_31X=1.023;
k_41G2=16.25;
k_41G=4.0;
k_41X=154.0;

k_ad=7.16;
E_max=8.32;

function [fE]=enzyme(xx)
E_F=xx(1);
E_B=E_T-E_F;
fE(1)=E_max*k_ad*E_F/(1+k_ad*E_F) - E_B/S
endfunction

xx(1)=[0.5];
S=S0;
[xres,v,info]=fsolve([xx],enzyme,1.d-5);
E_F=xres(1);
E_B=E_T-E_F;

// Condicoes iniciais
t0=0;           // Tempo inicial (h)
himp=1.0;       // Intervalo de impressao (para gerar o grafico)
tn=100.0;       // Tempo final (h)
t=[t0:himp:tn]; // Vetor da variavel independente
Lig=Lignin*S0;

y0=[S0*Cellulose; Cellobiose; Glucose; S0*Hemicellulose; Xylose; S0];

C=S0*Cellulose;
G2=0;
G=0;
H=S0*Hemicellulose;
X=0;
R_S=alfa*S/S0;
r1=k_1r*(E_B*C/S)*R_S*S/(1+G2/k_11G2+G/k_11G+X/k_11X);
r2=k_2r*(E_B*C/S)*R_S*S/(1+G2/k_21G2+G/k_21G+X/k_21X);
r3=k_3r*E_F*G2/(k_3M*(1+G/k_31G+X/k_31X)+G2);
r4=k_4r*(E_B*H/S)*R_S*S/(1+G2/k_41G2+G/k_41G+X/k_41X);

yd0=[-r1-r2; 1.056*r1-r3;1.111*r2+1.053*r3;-r4;1.136*r4;-r1-r2-r4];

function [r, ires]=chemres(t, y, yd)
C=y(1);    // Cellulose (g/L)
G2=y(2);   // Cellobiose (g/L)
G=y(3);    // Glucose (g/L)
H=y(4);    // Hemicellulose (g/L)
X=y(5);    // Xylose (g/L)
S=y(6);    // Biomass (g/L);

//xx(1)=[0.5];
//[xres,v,info]=fsolve([xx],enzyme,1.d-5);
//E_F=xres(1);
//E_B=E_T-E_F;

R_S=alfa*S/S0;
r1=k_1r*(E_B*C/S)*R_S*S/(1+G2/k_11G2+G/k_11G+X/k_11X);   // Eq. 5
//r1=k_1r*E_BC*R_S*S/(1+G2/k_1G2+G/k_1G+X/k_1X)        // Eq. 16

r2=k_2r*(E_B*C/S)*R_S*S/(1+G2/k_21G2+G/k_21G+X/k_21X);   // Eq. 6
//r2=k_2r*E_BC*R_S*S/(1+G2/k_1G2+G/k_1G+X/k_1X)        // Eq. 17

r3=k_3r*E_F*G2/(k_3M*(1+G/k_31G+X/k_31X)+G2);         // Eq. 7

r4=k_4r*(E_B*H/S)*R_S*S/(1+G2/k_41G2+G/k_41G+X/k_41X);   // Eq. 8
//r4=k_4r*E_B*H*R_S*S/(1+G2/k_1G2+G/k_1G+X/k_1X)        // Eq. 18

r(1) = - r1 - r2 - yd(1);             // Cellulose balance
r(2) = 1.053*r1 - r3 - yd(2);         // Cellobiose balance
r(3) = 1.111*r2 + 1.053*r3 - yd(3);   // Glucose balance
r(4) = -r4 - yd(4);
r(5) = 1.136*r4 - yd(5);
r(6) = C + H + Lig - S;
ires=0;
endfunction

// Integrando o sistema de equacoes pelo method de Dassl
y=dassl([y0,yd0],0,t,chemres);

// Graficos modelo Hydrolysis
scf(0)
plot(y(1,:),y(3,:),'r',dados(:,1),dados(:,3),'xr')
xtitle("Cellobiose")
xlabel('Time (h)')
ylabel('CG2 (g/L)')
scf(1)
plot(y(1,:),y(4,:),'r',dados(:,1),dados(:,2),'xr')
xtitle("Glucose")
xlabel('Time (h)')
ylabel('CG (g/L)')
scf(2)
plot(y(1,:),y(6,:),'r',dados(:,1),dados(:,4),'xr')
xtitle("Xylose")
xlabel('Time (h)')
ylabel('CX (g/L)')
scf(3)
plot(y(1,:),y(7,:),'r')
xtitle("Solid")
xlabel('Time (h)')
ylabel('CS (g/L)')
