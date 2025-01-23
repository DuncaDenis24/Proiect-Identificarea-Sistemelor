load('scope46.mat')
t=scope46{:,1}
u=scope46{:,2}
y1=scope46{:,3}
y2=scope46{:,4}
figure;
plot(t,[u,y1+3,y2+6]), shg
title("Achizitionarea datelor(intrare iesire si iesire cu zero")
xlabel('Timp')
ylabel('Voltz')
legend('u(y)','y1(t)','y2(t)')
%% Identificare Sistem 1
figure
plot(t,y1)
Mr1=((max(y1)-min(y1))/2)/((max(u)-min(u))/2)
roots([4 0 -4 0 1/Mr1^2])
%zeta1=0.384
%Tr1=0.0001618;
zeta1=0.4033;
Tr1=0.00016;
Fr1=1/Tr1
wr1=2*pi/Tr1
wn1=wr1/sqrt(1-2*zeta1^2)
K1=mean(y1)/mean(u)
H1=tf(K1*wn1^2,[1 2*zeta1*wn1 wn1^2])
A=[0 1; -wn1^2 -2*zeta1*wn1];
B=[0; K1*wn1^2];
C=[1 0];
D=0;
[y1sim,~,~]=lsim(ss(A,B,C,D),u,t,[y1(1),(y1(2)-y1(1))/(t(2)-t(1))]);
figure; 
plot(t,[y1,y1sim])
title('Iesirea sistemului si iesirea simulata')
legend('y1','y1sim')
espn=norm(y1-y1sim)/norm(y1-mean(y1))
%% 
%m1.DataIndex
figure
plot(t,[u,y1])

%%
w1=pi/(t(101)-t(85))
M1=(y1(86)-y1(102))/(u(85)-u(101))
Ph1=rad2deg(2*w1*(t(85)-t(86)))
w2=pi/(t(115)-t(101))
M2=(y1(116)-y1(102))/(u(115)-u(101))
Ph2=rad2deg((w1+w2)*(t(101)-t(102)))
w3=pi/(t(127)-t(115))
M3=(y1(116)-y1(129))/(u(115)-u(127))
Ph3=rad2deg(2*w3*(t(115)-t(116)))
w4=pi/(t(140)-t(127))
M4=(y1(141)-y1(129))/(u(140)-u(127))
Ph4=rad2deg(w4*(t(127)-t(129)))
w5=pi/(t(151)-t(140))
M5=(y1(141)-y1(154))/(u(140)-u(151))
Ph5=rad2deg(w5*(t(140)-t(141)))
w6=pi/(t(162)-t(151))
M6=(y1(164)-y1(154))/(u(162)-u(151))
Ph6=rad2deg(w6*(t(151)-t(154)))
w7=pi/(t(173)-t(162))
M7=(y1(164)-y1(175))/(u(162)-u(173))
Ph7=rad2deg(w7*(t(162)-t(164)))
w8=pi/(t(183)-t(173))
M8=(y1(185)-y1(175))/(u(183)-u(173))
Ph8=rad2deg(w8*(t(173)-t(175)))
w9=pi/(t(193)-t(183))
M9=(y1(185)-y1(195))/(u(183)-u(193))
Ph9=rad2deg(w9*(t(183)-t(185)))
w10=pi/(t(486)-t(480))
M10=(y1(484)-y1(489))/(u(480)-u(486))
Ph10=rad2deg(w10*(t(480)-t(484)))
w11=pi/(t(807)-t(803))
M11=(y1(806)-y1(810))/(u(803)-u(807))
Ph11=rad2deg(w11*((t(803)-t(806))))
w12=pi/(t(812)-t(807))
M12=(y1(815)-y1(810))/(u(812)-u(807))
Ph12=rad2deg(w12*((t(807)-t(810))))
w13=pi/(t(816)-t(812))
M13=(y1(815)-y1(820))/(u(812)-u(816))
Ph13=rad2deg(w13*((t(812)-t(815))))
w14=pi/(t(821)-t(816))
M14=(y1(824)-y1(820))/(u(821)-u(816))
Ph14=rad2deg(w14*((t(816)-t(820))))
w15=pi/(t(825)-t(821))
M15=(y1(824)-y1(829))/(u(821)-u(825))
Ph15=rad2deg(w15*((t(821)-t(824))))
w16=pi/(t(830)-t(825))
M16=(y1(833)-y1(829))/(u(830)-u(825))
Ph16=rad2deg(w16*((t(825)-t(829))))
w17=pi/(t(834)-t(830))
M17=(y1(833)-y1(837))/(u(830)-u(834))
Ph17=rad2deg(w17*((t(830)-t(833))))
w18=pi/(t(839)-t(834))
M18=(y1(842)-y1(837))/(u(839)-u(834))
Ph18=rad2deg(w18*((t(834)-t(837))))
w19=pi/(t(843)-t(839))
M19=(y1(842)-y1(847))/(u(839)-u(843))
Ph19=rad2deg(w19*((t(839)-t(842))))
w20=pi/(t(848)-t(843))
M20=(y1(851)-y1(847))/(u(848)-u(843))
Ph20=rad2deg(w20*((t(843)-t(847))))
w21=pi/(t(852)-t(848))
M21=(y1(851)-y1(855))/(u(848)-u(852))
Ph21=rad2deg(w21*((t(848)-t(851))))
Ph22=rad2deg(w21*((t(852)-t(855))))
Phr1=rad2deg(wr1*(t(254)-t(257)))
w23=pi/(t(990)-t(986))
M23=(y1(989)-y1(993))/(u(986)-u(990))
Ph23=rad2deg(w23*(t(986)-t(989)))
w24=pi/(t(994)-t(990))
M24=(y1(997)-y1(993))/(u(994)-u(990))
Ph24=rad2deg(w24*(t(990)-t(993)))
md=mean([M11,M12,M13,M14,M15,M16,M17,M18,M19,M20,M21,M21])
wd=mean([w11,w12,w13,w14,w15,w16,w17,w18,w19,w20,w21,w21])
pd=mean([Ph11,Ph12,Ph13,Ph14,Ph15,Ph16,Ph17,Ph18,Ph19,Ph20,Ph21,Ph22])
figure
w=logspace(4,6);
[num,den]=tfdata(H1,'v');
[M,Ph]=bode(num,den,w);
subplot(211)
semilogx([w1,w2,w3,w4,w5,w6,w7,w8,w9,wr1,w10,wd,w23,w24], ...
    20*log10([M1,M2,M3,M4,M5,M6,M7,M8,M9,Mr1,M10,md,M23,M24]),'o');
hold on
semilogx(w, 20 * log10(M)); grid
title('Caracteristica de modul')
xlabel('Frecventa (rad/s)');
ylabel('Modul (dB)');

subplot(212)
semilogx([w1,w2,w3,w4,w5,w8,w9,wr1,w10,wd,w23,w24],[Ph1, Ph2, Ph3, Ph4,mean([Ph5,Ph6,Ph7]),Ph8,Ph9,Phr1,Ph10,pd,Ph23,Ph24],'o')
hold on
semilogx(w, Ph); grid
title('Caracteristica de faza')
xlabel('Frecventa (rad/s)');
ylabel('Faza (grade)');
%%
% Punctele de frecvență și fază
frecv =[w1,w2,w3,w4,w5,w6,w7,w8,w9,wr1,w10,w11,w12,w13,w14,w15,w16,w17,w18,w19,w20,w21,w21,w23,w24];
frequencies=unique(frecv)
ph = [Ph1, Ph2, Ph3, Ph4,Ph5,Ph6,Ph7,Ph8,Ph9,Phr,Ph10,Ph11,Ph12,Ph13,Ph14,Ph15,Ph16,Ph17,Ph18,Ph19,Ph20,Ph21,Ph22,Ph23,Ph24];
phases=unique(ph)
% Interpolare spline
frequencies_interp = logspace(log10(min(frequencies)), log10(max(frequencies)), 100); % 100 puncte de frecvență interpolate
phases_interp = interp1(frequencies, phases, frequencies_interp, 'spline');

% Plotare
subplot()
semilogx(frequencies, phases, 'o', 'DisplayName', 'Puncte măsurate');
hold on;
semilogx(frequencies_interp, phases_interp, 'DisplayName', 'Interpolare spline');
grid on;
xlabel('Frecvența (rad/s)');
ylabel('Faza (grade)');
title('Caracteristica de fază');
legend;
semilogx(w,Ph)
