%% CEN.PK.XXX STRAIN MODEL.
%% AUTHOR: Pau Cabaneros L�pez (pacalo@kt.dtu.dk)
%% PROSYS, TECHNICAL UNIVERSITY OF DENMARK Jan, 2019
%% FILE DESCRIPTION: KINETIC EQUATIONS.

% The following 9 components are considered in the model:
% 1. Glucose
% 2. Xylose
% 3. Furfural
% 4. Furfuryl alcohol
% 5. 5-HMF
% 6. HAc
% 7. Ac
% 8. Ethanol
% 9. Biomass

% According to the kinetci model, the following chemical equations describe
% the system.

% 1. Glucose  -----> Ethanol + Biomass
% 2. Xylose   -----> Ethanol + Biomass
% 3. Furfural -----> Furfuryl alcohol
% 4. HMF      -----> Acetic acid
% 5. HAc     <-----> Ac-
% 6. HAc      -----> Biomass (maintenence)



function [dxdt,rr] = kinetics(t,x,p)
%% 1. STOICHIOMETRIC MATRIX.
% The 9 components and 6 reactions are considered in this model.

%     Glu  Xyl  Fur  FA   HMF  HAc    EtOH     X
st = [-1    0    0    0    0    0     p(24)    p(32);    % Glucose uptake
       0   -1    0    0    0    0     p(25)    p(33);    % Xylose uptake
       0    0   -1    p(9) 0    0     0        0    ;    % Furfural uptake
       0    0    0    0   -1    p(21) 0        0    ;    % HMF uptake
       0    0    0    0    0   -1     0        0    ];   % HAc uptake




%% 2. DEFINE INDIVIDUAL KINETICS AND INHIBITION TERMS.
% Ph stands for phenomena, and it includes all the phenomena considered in
% the model, including reaction, inhibition and equilibria.
% 1. GLUCOSE UPTAKE RATE.
numaxG = p(1);         % max consumption rate of glucose           s^-1       [1]
KSPG   = p(2);         % affinity constant glucose                 g/L        [1]
KiPG   = p(3);         % inhibition constant glucose               g/L        [1]
Ph1 = numaxG * x(1,1) * x(8,1) / (KSPG + x(1,1) + (x(1,1)^2)/KiPG);
Ph1 = max(0,Ph1);
% 2. XYLOSE UPTAKE RATE.
numaxX = p(4);         % max consumption rate of xylose            s^-1       [1]
KSPX   = p(5);         % affinity constant xylose                  g/L        [1]
KiPX   = p(6);         % inhibition constant xylose                g/L        [1]
Ph2 = numaxX * x(2,1) * x(8,1) / (KSPX + x(2,1) + (x(2,1)^2)/KiPX);
Ph2 = max(0,Ph2);

% 3. FURFURAL UPTAKE RATE.
numaxFur = p(7);       % max uptake rate of furfural               s^-1       [2]
KSFur    = p(8);       % affinity constant furfural                g/L        [2]
Ph3 = numaxFur * x(3,1) * x(8,1) / (KSFur + x(3,1));
Ph3 = max(0,Ph3);

% 4. FURFURAL CONVERSION INTO FURFURYL ALCOHOL.
%Y_FA_Fur = p(9);       % yield coefficient FA/Fur                  gFA/gFur   [2]
%Ph4 = Ph3 * Y_FA_Fur;
%Ph4 = max(0,Ph4);

% 5. FURFURYL ALCOHOL INHIBITS GLUCOSE UPTAKE RATE.
KiFAg = p(10);         % inhibition constant of FA on Glu uptake   g/L        [2]
Ph5 = 1 / (1 + x(4,1)/KiFAg);
Ph5 = max(0,Ph5);

% 6. FURFURYL ALCOHOL INHIBITS XYLOSE UPTAKE RATE.
KiFAx = p(11);         % inhibition constant of FA on Xyl uptake   g/L        [2]
Ph6 = 1 / (1 + x(4,1)/KiFAx);
Ph6 = max(0,Ph6);

% 7. FURFURAL INHIBITS GLUCOSE UPTAKE RATE.
KiFurg = p(12);        % inhibition constant of Fur on Glu uptake  g/L        [2]
Ph7 = 1 / (1 + x(3,1)/KiFurg);
Ph7 = max(0,Ph7);

% 8. FURFURAL INHIBITS XYLOSE UPTAKE RATE.
KiFurx = p(13);        % inhibition constant of Fur on Xyl uptake  g/L        [2]
Ph8 = 1 / (1 + x(3,1)/KiFurx);
Ph8 = max(0,Ph8);

% 9. FURFURAL INHIBITS HMF UPTAKE RATE.
KiFurHMF = p(14);      % inhibition constant of Fur on HMF         g/L        [2]
Ph9 = 1 / (1 + x(3,1)/KiFurHMF);
Ph9 = max(0,Ph9);

% 10. HMF INHIBITS GLUCOSE UPTAKE RATE.
KiHMFg = p(15);        % inhibition constant of HMF on Glu         g/L        [2]
Ph10 = 1 / (1 + x(5,1)/KiHMFg);
Ph10 = max(0,Ph10);

% 11. HMF INHIBITS XYLOSE UPTAKE RATE.
KiHMFx = p(16);        % inhibition constant of HMF on Xyl         g/L        [2]
Ph11 = 1 / (1 + x(5,1)/KiHMFx);
Ph11 = max(0,Ph11);

% 12. HMF UPTAKE RATE.
numaxHMF = p(17);      % max uptake rate of HMF                    s^-1       [2]
KSHMF = p(18);         % affinity constant HMF                     g/L        [2]
Ph12 = numaxHMF * x(8,1) * x(5,1) / (KSHMF + x(5,1));
Ph12 = max(0,Ph12);

% 13. pH INFLUENCES THE PAIR HAc/Ac.
%pH = 5.5;              % pH is controlled
%pKa = 4.75;            % pKa of acetic acid.
%Ph13 = x(7,1) / x(6,1) * 1 / (1 + 10^(-pH)/10^(-pKa));
%Ph13 = max(0,Ph13);


% 14. HAc UPTAKE RATE.
numaxHAc = p(19);      % max uptake rate of HAc                    s^-1       [2]
KSHAc = p(20);         % affinity constant HAc                     g/L        [2]
Ph14 = x(8,1) * x(6,1) * numaxHAc / (KSHAc + x(6,1));
Ph14 = max(0,Ph14);


% 15. HMF CONVERSION INTO HAc
%Y_HAc_HMF = p(21);     % yield coefficient HAc/HMF                 gHAc/gHMF  [2], [3]
%Ph15 = Y_HAc_HMF * Ph12;
%Ph15 = max(0,Ph15);

% 16. HAc INHIBITS GLUCOSE UPTAKE RATE.
KiHAcg = p(22);        % inhibition constant of HAc on Glu         g/L        [2]
Ph16 = 1 / (1 + x(6,1)/KiHAcg);
Ph16 = max(0,Ph16);

% 17. HAc INHIBITS XYLOSE UPTAKE RATE.
KiHAcx = p(23);        % inhibition constant of HAc on Xyl         g/L         [2]
Ph17 = 1 / (1 + x(6,1)/KiHAcx);
Ph17 = max(0,Ph17);

% 18. PRODUCTION OF ETHANOL FORM GLUCOSE AND XYLOSE.
%YPSg = p(24);          % yield ethanol-glucose                     gEtOH/gGlu [1]
%YPSx = p(25);          % yield ethanol-xylose                      gEtOH/gGlu [1]
%Ph18a = YPSg * Ph1;
%Ph18b = YPSx * Ph2;
%Ph18a = max(0,Ph18a);
%Ph18b = max(0,Ph18b);

% 19. ETHANOL INHIBITS THE UPTAKE OF GLUCOSE AND XYLOSE.
PMPg = p(26);          % inhibition constant for glucose           g/L        [1]
gammaG = p(27);        % exponent factor inhibition glucose        g/L        [1]
PMPx = p(28);          % inhibition constant for xylose            g/L        [1]
gammaX = p(29);        % exponent factor inhibition xylose         g/L        [1]
Ph19a = 1-(x(7,1)/PMPg)^gammaG;
Ph19b = 1-(x(7,1)/PMPx)^gammaX;
Ph19a = max(0,Ph19a);
Ph19b = max(0,Ph19b);

% 20. CELL GROWTH.
%mGlu = p(30);          % maintenance constant from glucose         g/L        [1]
%mXyl = p(31);          % maintenance constant from xylose          g/L        [1]
%YXSg = p(32);          % yield X-Glu                               gX/gGlu    [1]
%YXSx = p(33);          % yield X-Xyl                               gX/gXyl    [1]
%mumaxG = p(34);        % max growth of X from Glu                  h-1        [1]
%mumaxX = p(35);        % max frowth of X from Xyl                  h-1        [1]

%Ph20a = max(0,(Ph1 + mGlu*x(8,1))*YXSg);
%Ph20b = max(0,(Ph1 + mXyl*x(8,1))*YXSx);

% 21. CATABOLITE REPRESSION.
KiGlu = p(36);         % inhibition constant of Glu to Xyl         g/L        Unknown
Ph21 = 1 / (1 + x(1,1)/KiGlu);

% 22. ACETATE IS USED FOR MAINTENANCE.
mHAc = p(37);          % maintenance constant from HAc              g/L        Unknown
YXSHAc = p(38);        % yield acetate biomass                      g/L        Unknown
Ph22 = max(0,mHAc * x(8,1) * YXSHAc);

%% 3. REACTION RATES.
% This includes the reaction rates with inhibition terms.
rr(1,1)  = Ph1 * Ph5 * Ph7 * Ph10 * Ph16 * Ph19a;                           % Glucose uptake rate considering inhibitions.
rr(2,1)  = Ph2 * Ph6 * Ph8 * Ph11 * Ph17 * Ph19b * Ph21;                    % Xylose uptake rate considering inhibitions.
rr(3,1)  = Ph3;                                                             % Furfural uptake rate.
rr(4,1)  = Ph12 *  Ph9;                                                     % 5-HMF uptake rate considering inhibitions.
rr(5,1)  = Ph14;

%% 4. MASS BALANCE FOR THE SYSTEM.
 dxdt = (rr'*st)';
%  dxdt = dxdt';



end
















