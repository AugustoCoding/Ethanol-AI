%% CEN.PK.XXX STRAIN MODEL.
%% AUTHOR: Pau Cabaneros López (pacalo@kt.dtu.dk)
%% PROSYS, TECHNICAL UNIVERSITY OF DENMARK Jan, 2019
%% FILE DESCRIPTION: INITIAL CONDITIONS OF THE PROCESS.
%% LIST OF INITIAL CONDITIONS.
% Name     Value     Index     Units
  Glu0  =  37;       % 1       g/L
  Xyl0  =  22;       % 2       g/L
  Fur0  =  0.62;     % 3       g/L
  FA0   =  0;        % 4       g/L
  HMF0  =  0.2;        % 5       g/L
  HAc0  =  3.05;    % 6       g/L
 % Ac0   =  0.0001;  % 7       g/L
  EtOH0 =  0.62;     % 8       g/L
  X0    =  1.5;      % 9       g/L
  
  init(1) = Glu0;
  init(2) = Xyl0;
  init(3) = Fur0;
  init(4) = FA0;
  init(5) = HMF0;
  init(6) = HAc0;
 % init(7) = Ac0;
  init(7) = EtOH0;
  init(8) = X0;
  