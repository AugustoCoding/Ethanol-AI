%% CEN.PK.XXX STRAIN MODEL.
%% AUTHOR: Pau Cabaneros López (pacalo@kt.dtu.dk)
%% PROSYS, TECHNICAL UNIVERSITY OF DENMARK Jan, 2019
%% FILE DESCRIPTION: SOLVE ODE SYSTEM.

function [y,t,rr] = simulation(t,p,init)

%% SOLVUTION OF THE MODEL.
options = odeset('RelTol', 1e-5, 'AbsTol', 1e-7);
[t,y,rr] = ode15s(@kinetics, t, init, options, p);

end