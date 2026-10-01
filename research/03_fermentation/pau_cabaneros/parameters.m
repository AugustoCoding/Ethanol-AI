%% CEN.PK.XXX STRAIN MODEL.
%% AUTHOR: Pau Cabaneros López (pacalo@kt.dtu.dk)
%% PROSYS, TECHNICAL UNIVERSITY OF DENMARK Jan, 2019
%% FILE DESCRIPTION: LIST OF PARAMETERS.
%% Cited literature:
% [1] Krishnan et al, 1999
% [2] Hanly et al, 2004
% [3] Mauricio-Iglesias et al, 


%% 1. GLUCOSE UPTAKE RATE.
numaxG = 2.6808  ;      % max consumption rate of glucose (1.0005)         h^-1       [1]
KSPG   = 0.14637;         % affinity constant glucose to ethanol      g/L        [1]
KiPG   = 4752.8;          % inhibition constant glucose to ethanol    g/L        [1]

p(1:3) = [numaxG, KSPG, KiPG];
%% 2. XYLOSE UPTAKE RATE.
numaxX = 4.4851;        % max consumption rate of xylose            h^-1       [1]
KSPX   = 0.58763;          % affinity constant xylose to ethanol       g/L        [1]
KiPX   = 21.608;         % inhibition constant xylose to ethanol     g/L        [1]

p(4:6) = [numaxX, KSPX, KiPX];
%% 3. FURFURAL UPTAKE RATE.
numaxFur = 0.14309;  % max uptake rate of furfural               h^-1       [2]
KSFur    = 0.05;       % affinity constant furfural                g/L        [2]

p(7:8) = [numaxFur, KSFur];
%% 4. FURFURAL CONVERSION INTO FURFURYL ALCOHOL.
Y_FA_Fur = 1.02;       % yield coefficient FA/Fur                  gFA/gFur   [2]

p(9) = Y_FA_Fur;
%% 5. FURFURYL ALCOHOL INHIBITS GLUCOSE UPTAKE RATE.
KiFAg = 5;             % inhibition constant of FA on Glu uptake   g/L        [2]

p(10) = KiFAg;
%% 6. FURFURYL ALCOHOL INHIBITS XYLOSE UPTAKE RATE.
KiFAx = 6;             % inhibition constant of FA on Xyl uptake   g/L        [2]

p(11) = KiFAx;
%% 7. FURFURAL INHIBITS GLUCOSE UPTAKE RATE.
KiFurg = 0.4529;         % inhibition constant of Fur on Glu uptake  g/L        [2]

p(12) = KiFurg;
%% 8. FURFURAL INHIBITS XYLOSE UPTAKE RATE.
KiFurx = 0.35;         % inhibition constant of Fur on Xyl uptake  g/L        [2]

p(13) = KiFurx;
%% 9. FURFURAL INHIBITS HMF UPTAKE RATE.
KiFurHMF = 0.25;       % inhibition constant of Fur on HMF         g/L        [2]

p(14) = KiFurHMF;
%% 10. HMF INHIBITS GLUCOSE UPTAKE RATE.
KiHMFg = 2;            % inhibition constant of HMF on Glu         g/L        [2]

p(15) = KiHMFg;
%% 11. HMF INHIBITS XYLOSE UPTAKE RATE.
KiHMFx = 10;           % inhibition constant of HMF on Xyl         g/L        [2]

p(16) = KiHMFx;
%% 12. HMF UPTAKE RATE. 
numaxHMF = 0.3154;     % max uptake rate of HMF                    h^-1       [2]
KSHMF = 0.5;           % affinity constant HMF                     g/L        [2]

p(17:18) = [numaxHMF, KSHMF];
%% 13. pH INFLUENCES THE PAIR HAc/Ac.
pH = 5.5;              % pH is controlled
pKa = 4.75;            % pKa of acetic acid.

%% 14. HAc UPTAKE RATE.
numaxHAc = 0.0443*0.001;     % max uptake rate of HAc                    h^-1       [2]
KSHAc = 2.5;           % affinity constant HAc                     g/L        [2]

p(19:20) = [numaxHAc, KSHAc];
%% 15. HMF CONVERSION INTO HAc
Y_HAc_HMF = 0.534;     % yield coefficient HAc/HMF                 gHAc/gHMF  [2], [3]

p(21) = Y_HAc_HMF;
%% 16. HAc INHIBITS GLUCOSE UPTAKE RATE.
KiHAcg = 5.1827;         % inhibition constant of HAc on Glu         g/L        [2]

p(22) = KiHAcg;
%% 17. HAc INHIBITS XYLOSE UPTAKE RATE.
KiHAcx = 0.73;       % inhibition constant of HAc on Xyl         g/L        [2]

p(23) = KiHAcx;
%% 18. PRODUCTION OF ETHANOL FORM GLUCOSE AND XYLOSE.
YPSg = 0.42;           % yield ethanol-glucose                     gEtOH/gGlu [1]
YPSx = 0.24;           % yield ethanol-xylose                      gEtOH/gGlu [1]

p(24:25) = [YPSg, YPSx];
%% 19. ETHANOL INHIBITS THE UPTAKE OF GLUCOSE AND XYLOSE.
PMPg = 32.04;           % inhibition constant for glucose           g/L        [1]
gammaG = 1.42;         % exponent factor inhibition glucose        g/L        [1]
PMPx = 29.442;           % inhibition constant for xylose            g/L        [1]
gammaX = 1.1533;        % exponent factor inhibition xylose         g/L        [1]

p(26:29) = [PMPg, gammaG, PMPx, gammaX];
%% 20. CELL GROWTH.
mGlu = 2.69E-5;        % maintenance constant from glucose         g/L        [1]
mXyl = 1.86E-5;        % maintenance constant from xylose          g/L        [1]
YXSg = 0.115;          % yield X-Glu                               gX/gGlu    [1]
YXSx = 0.162;          % yield X-Xyl                               gX/gXyl    [1]
mumaxG = 0.3308;       % max growth of X from Glu                  h-1        [1]
mumaxX = 1.0008;       % max frowth of X from Xyl                  h-1        [1]
% KSPGgr = 0.565;        % affinity constant glucose to growth       g/L        [1]
% KiPGgr = 283.7;        % inhibition constant glucose to growth     g/L        [1]
% KSPXgr = 3.4;          % affinity constant xylose to growth        g/L        [1]
% KiPXgr = 18.1;         % inhibition constant xylose to growth      g/L        [1]
% PMXg   = 95.4 ;        % max EtOH concentration for growth glucose g/L        [1]
% PMXx   = 59.04;        % max EtOH concentration for growth xylose  g/L        [1]

p(30:35) = [mGlu, mXyl,YXSg, YXSx, mumaxG, mumaxX];
%% 21. CATABOLITE REPRESSION.
KiGlu = 5;          % inhibition constant of Glu to Xyl         g/L        Unknown

p(36) = KiGlu;

%% 22. ACETIC ACID IS USED FOR MAINTENANCE.
mHAc = 0;          % maintenance constant from HAc              g/L        Unknown
YXSHAc = 0;        % yield acetate biomass                      g/L        Unknown 

p(38:39) = [mHAc, YXSHAc];


%% COLLECT PARAMETER NAMES
par_names = {'numaxG', 'KSPG', 'KiPG', 'numaxX', 'KSPX', 'KiPX', 'numaxFur', 'KSFur', 'Y_FA_Fur', 'KiFAg', 'KiFAx', 'KiFurg', ...
    'KiFurx', 'KiFurHMF', 'KiHMFg', 'KiHMFx', 'numaxHMF', 'KSHMF', 'numaxHAc', 'KSHAc', 'Y_HAC_HMF', 'KiHAcg', 'KiHAcx', ...
    'YPSg', 'YPSx', 'PMPg', 'gammaG', 'PMPx', 'gammaX', 'mGlu', 'mXyl', 'YXSg', 'YXSx', 'KiGlu'};

%% COLLECT DEPENDENT VARIABLE NAMES
var_names = {'Glucose', 'Xylose', 'Furfural','Furfuryl alcohol','5-HMF', 'Acetic acid', 'Ethanol', 'Biomass'};















