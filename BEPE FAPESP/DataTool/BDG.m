%% Big data generator 
% This script runs the model "Model_fedbatch" of a bioreactor for
% ethanol production a fixed number of times and collects data. 

% Clear workspace and command window. 
clc
clear

%% An interactive menu for running the program 
while true

MainMenu = menu('Please choose an action', 'Upload parameters','Generate save file' , 'Run simulations', 'Cancel');
            switch MainMenu
                case 1
                    choice = 1;
                case 2
                    choice = 2;
                case 3
                    choice = 3;
                case 4
                    choice = 4;
                otherwise                   
                    clear
                    break
            end
%-----------------------------------------------------
%% Menu item chosen

%% 1. Upload parameters
if choice == 1
    % Chose file with parameters for the simulation
    Simulation_file = uigetfile('*.xlsx*');
    if Simulation_file == 0
        break
    else
    PM = readmatrix(Simulation_file);  
    % Mark that parameters have been uploaded
    Mark1 = 1;
    end
    
end

%% 2. Generate save file
if choice == 2
    % Chose the name of the file that stores the simulation data
    Data_file = uiputfile('*.xls');   
    
    % Mark that a save file has been generated
    Mark2 = 2;
end

%% 3. Run simulations
if choice == 3

tic    
    % Make sure that step 2 and 3 has been done    
    if and(exist('Mark1','var'),exist('Mark2','var')) ~= true
    % Generate error message     
    ErrorMessage = sprintf('Please upload parameters and generate a save file before running the simulations');
    uiwait(msgbox(ErrorMessage));
    
    % If step 2 and 3 has been done then simulation can proceed
    else

    % Number of simulations
    nSimulations = PM(13,10);

    % Run simulations in a loop and collect data
    for j = 1:nSimulations
        % Load parameters
        Seed = 23341 + j;
        Parameters;
    
        % Simulate model
        sim('Model_fedbatch');
    
        % Log data
        % continuous concentrations
        Ethanol{1,j}     = logsout{2}.Values.Data;
        Glucose{1,j}     = logsout{3}.Values.Data;
        Xylose{1,j}      = logsout{4}.Values.Data;
    
        % Discrete concentrations
        EthanolD{1,j}     = logsout{5}.Values.Data;
        GlucoseD{1,j}     = logsout{6}.Values.Data;
        XyloseD{1,j}      = logsout{7}.Values.Data;
    
        % Temperature and time
        Temperature{1,j} = logsout{8}.Values.Data;
        TimeD{1,j}       = logsout{9}.Values.Data;
        Time{1,j}        = logsout{1}.Values.Data;
    
        % Export data to Excel
        % Create column number for Excel
        RN = strcat(xlscol(j),num2str(1));
    
        % Transfer data to .xls file
        % Continuous parameters
        writematrix(Glucose{1,j},Data_file,'Sheet','Glucose','Range',RN)
        writematrix(Xylose{1,j},Data_file,'Sheet','Xylose','Range',RN)
        writematrix(Ethanol{1,j},Data_file,'Sheet','Ethanol','Range',RN)
        writematrix(Temperature{1,j},Data_file,'Sheet','Temperature','Range',RN)
        writematrix(Time{1,j},Data_file,'Sheet','Time','Range',RN)
    
        % Discrete parameters
        writematrix(GlucoseD{1,j},Data_file,'Sheet','GlucoseDiscrete','Range',RN)
        writematrix(XyloseD{1,j},Data_file,'Sheet','XyloseDiscrete','Range',RN)
        writematrix(EthanolD{1,j},Data_file,'Sheet','EthanolDiscrete','Range',RN)
        writematrix(TimeD{1,j},Data_file,'Sheet','TimeDiscrete','Range',RN)
        
        % Mark that simulations has been run
        Mark3 = 3;
    end
toc
    % End Menu loop
    break
    
    end
    
end

%% 4. Cancel
if choice == 4
break    
    
end

end

% Make sure that the option to preview data only comes when simulations
% has been run.
if exist('Mark3','var') == true
% Chose whether to preview data or not
while true
PlotMenu = menu('Would you like to preview the simulated data?', 'Yes','No');

switch PlotMenu
case 1
%% Plot data
t = tiledlayout(3,2);
tile1 = nexttile;
tile2 = nexttile;
tile3 = nexttile;
tile4 = nexttile;
tile5 = nexttile;
tile6 = nexttile;
title(t,'Concentrations over time')
xlabel(t,'Time (h)')
ylabel(t,'Concentration (g/L)')

% Plot continuous glucose concentrations
for j = 1:nSimulations
 hold(tile1,'on')
 plot(tile1,Time{j},Glucose{j})
 title(tile1,'Glucose')
 hold(tile1,'off')
end

% Plot discrete glucose concentrations
for j = 1:nSimulations
 hold(tile2,'on')
 scatter(tile2,TimeD{j},GlucoseD{j},15)
 title(tile2,'GlucoseDiscrete')
 hold(tile2,'off')
end

% Plot continuous xylose concentrations
for j = 1:nSimulations
 hold(tile3,'on')
 plot(tile3,Time{j},Xylose{j})
 title(tile3,'Xylose')
 hold(tile3,'off')
end

% Plot discrete xylose concentrations
for j = 1:nSimulations
 hold(tile4,'on')
 scatter(tile4,TimeD{j},XyloseD{j},15,'filled','d')
 title(tile4,'XyloseDiscrete')
 hold(tile4,'off')
end

% Plot continuous ethanol concentrations
for j = 1:nSimulations
 hold(tile5,'on')
 plot(tile5,Time{j},Ethanol{j})
 title(tile5,'Ethanol')
 hold(tile5,'off')
end

% Plot discrete ethanol concentrations
for j = 1:nSimulations
 hold(tile6,'on')
 scatter(tile6,TimeD{j},EthanolD{j},15,'filled','v')
 title(tile6,'EthanolDiscrete')
 hold(tile6,'off') 
 
end
break

case 2
% Quit
break

otherwise                   
clear
break
end

end

end

