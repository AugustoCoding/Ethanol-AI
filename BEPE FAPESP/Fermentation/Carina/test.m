fopen('../Simulation/parameters.m');
initial
t = linspace(0,300,300);
y = simulation(t,p,init);
figure(2)
subplot(1,4,1); plot(t,y(:,1:2)); hold on;% plot(t,y(:,8)); 
subplot(1,4,2); plot(t,y(:,3:4)); hold on;
subplot(1,4,3); plot(t,y(:,5:6)); hold on;
subplot(1,4,4); plot(t,y(:,7:8)); hold on;
figure(3)
plot(t,y(:,[1:3,5])); hold on;

figure(4)
plot(t,y(:,1),'LineWidth',2,'color','b'); hold on; 
plot(t,y(:,2),'LineWidth',2,'color','r');
xlabel('Time [h]');
ylabel('Concentration [g/L]');
legend('Glucose','Xylose')

figure(5)
plot(t,y(:,3),'LineWidth',2,'color','b'); hold on; 
plot(t,y(:,4),'LineWidth',2,'color','r');
xlabel('Time [h]');
ylabel('Concentration [g/L]');
legend('Furfural','Furfuryl alcohol')

figure(6)
plot(t,y(:,5),'LineWidth',2,'color','b'); hold on; 
plot(t,y(:,6),'LineWidth',2,'color','r');
xlabel('Time [h]');
ylabel('Concentration [g/L]');
legend('5-HMF','Acetic acid')

figure(7)
plot(t,y(:,7),'LineWidth',2,'color','b'); hold on; 
plot(t,y(:,8),'LineWidth',2,'color','r');
xlabel('Time [h]');
ylabel('Concentration [g/L]');
legend('Ethanol','Biomass')