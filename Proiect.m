load('scope46.mat')
t=scope46{:,1}
u=scope46{:,2}
y1=scope46{:,3}
y2=scope46{:,4}
figure;
plot(t,[u+6,y1+3,y2]), shg
title("Datele experimentale(intrare, iesire si iesire cu zero)")
xlabel('Timp[sec]')
ylabel('Volt[V]')
legend('u(t)','y1(t)','y2(t)')


%%
plot(t,[u,y1])
title("Achizitia Datelor(sistem 1)")
xlabel('Timp[sec]')
ylabel('Volt[V]')
legend('u(t)','y1(t)')

%% Identificare Sistem 1
figure
plot(t,y1)
Mr1=(y1(257)-y1(265))/(u(254)-u(262)) %modulul de rezonanta
roots([4 0 -4 0 1/Mr1^2]) %
%zeta1=0.384
%Tr1=0.0001618;
zeta1=0.3719;
Tr1=t(273)-t(257);
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
xlabel('Timp[sec]')
ylabel('Volt[V]')
legend('y1','y1sim')
empn1=norm(y1-y1sim)/norm(y1-mean(y1))
%%
plot(t,[u,y1])
%%
wm=pi/(t(486)-t(480))
Mm=(y1(484)-y1(489))/(u(480)-u(486))
w=pi/(t(990)-t(986))
M=(y1(989)-y1(993)/(u(986)-u(990)))
w/wm
p=20*log10(Mm/M)*10
%%
%Bode sistem 
%%plot(t,[u,y1])
%pentru panta 
%panta=(y2-y1)/(x2-x1)
%p=20*log(M24/M10)/log10(w24/w10)
w1=pi/(t(101)-t(85))%t de la y min - t de la y max
M1=(y1(86)-y1(102))/(u(85)-u(101)) 
Ph1=rad2deg(2*w1*(t(85)-t(86)))%t de la u max - t de la y max
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
xlabel('Pulsatia (rad/sec)');
ylabel('Modul (dB)');

subplot(212)
semilogx([w1,w2,w3,w4,w5,w8,w9,wr1,w10,wd,w23,w24],[Ph1, Ph2, Ph3, Ph4,mean([Ph5,Ph6,Ph7]),Ph8,Ph9,Phr1,Ph10,pd,Ph23,Ph24],'o')
hold on
semilogx(w, Ph); grid
title('Caracteristica de faza')
xlabel('Pulsatia (rad/sec)');
ylabel('Faza (grade)');
%pentru panta 
%panta=(y2-y1)/(x2-x1)
%p=20*log(M24/M10)/(w24/w10) *10
p=20*log10(M24/M10)/(w24/w10)*10
%%
plot(t,[u y2])
%% Identificare Sistem 2
%%pulsatia de frangere din bode => zero ul
%Ph, M, si Tzw
K2=mean(y2)/mean(u)  
figure
plot(t,[u,y2])
Mr2=(y2(256)-y2(264))/(u(254)-u(262))
roots([4 0 -4 0 1/Mr2^2])
zeta2=0.3521
Tr2=t(272)-t(256);
Fr2=1/Tr2
wr2=pi/(t(264)-t(256))
wn2=wr2/sqrt(1-2*zeta2^2)
Phr=wr2*(t(254)-t(256))
x=fsolve(@(T)atan((2*zeta1*wr2/wn2)/(1-(wr2/wn2)^2))+Phr-atan(T*wr2),0)
%Phr=-atan((2*zeta1*wr1/wn1)/(1-(wr1/wn1)^2))+atan(T*wr2)
T1=sqrt(Mr2^2*4*zeta2^2*(1-zeta2^2)-1)/wr2 %9.6170e-06
T2=8.9711e-06 %din modul
H2=tf([K2*x*wn2^2 K2*wn2^2],[1 2*zeta2*wn2 wn2^2])
[num,den]=tfdata(H2,'v');
[A,B,C,D]=tf2ss(num,den);
[y2sim,~,~]=lsim(ss(A',C',B',D'),u,t,[y2(1),(y2(2)-y2(1))/(t(5)-t(4))]);
figure; 
plot(t,[y2,y2sim])
legend('y2','y2sim')
empn2=norm(y2-y2sim)/norm(y2-mean(y2))
%% Bode sistem 2
figure

 plot(t,[u,y1, final.Position ...
     ,'o'])
%%

w1=pi/(t(100)-t(85))
M1=(y2(86)-y2(101))/(u(85)-u(100))
Ph1=rad2deg(w1*(t(85)-t(86)))
w2=pi/(t(115)-t(100))
M2=(y2(115)-y2(101))/(u(115)-u(100))
Ph2=rad2deg(w2*(t(100)-t(101)))
w3=pi/(t(127)-t(115))
M3=(y2(115)-y2(128))/(u(115)-u(127))
Ph3=rad2deg(w3*(t(127)-t(128)))
w4=pi/(t(140)-t(127))
M4=(y2(141)-y2(128))/(u(140)-u(127))
Ph4=rad2deg(w4*(t(127)-t(128)))
w5=pi/(t(151)-t(140))
M5=(y2(141)-y2(152))/(u(140)-u(151))
Ph5=rad2deg(w5*(t(140)-t(141)))
w6=pi/(t(162)-t(151))
M6=(y2(163)-y2(152))/(u(162)-u(151))
Ph6=rad2deg(w6*(t(151)-t(152)))
w7=pi/(t(173)-t(162))
M7=(y2(163)-y2(174))/(u(162)-u(173))
Ph7=rad2deg(w7*(t(162)-t(163)))
w8=pi/(t(183)-t(173))
M8=(y2(185)-y2(174))/(u(183)-u(173))
Ph8=rad2deg(w8*(t(173)-t(174)))

Phr2=rad2deg(wr2*(t(254)-t(256)))
w10=pi/(t(584)-t(579))
M10=(y2(581)-y2(587))/(u(579)-u(584))
Ph10=rad2deg(w10*(t(579)-t(581)))
w9=pi/(t(675)-t(670))
M9=(y2(673)-y2(678))/(u(670)-u(675))
Ph9=rad2deg(w9*(t(670)-t(673)))
w11=pi/(t(680)-t(675))
M11=(y2(683)-y2(678))/(u(680)-u(675))
Ph11=rad2deg(w11*(t(675)-t(678)))

w12=pi/(t(789)-t(784))
M12=(y2(787)-y2(792))/(u(784)-u(789))
Ph12=rad2deg(w12*(t(784)-t(787)))
w13=pi/(t(793)-t(789))
M13=(y2(796)-y2(792))/(u(793)-u(789))
Ph13=rad2deg(w13*(t(789)-t(792)))
w14=pi/(t(798)-t(793))
M14=(y2(796)-y2(801))/(u(793)-u(798))
Ph14=rad2deg(w14*(t(793)-t(796)))
w15=pi/(t(803)-t(798))
M15=(y2(805)-y2(801))/(u(803)-u(798))
Ph15=rad2deg(w15*(t(798)-t(801)))
w16=pi/(t(807)-t(803))
M16=(y2(805)-y2(810))/(u(803)-u(807))
Ph16=rad2deg(w16*(t(803)-t(805)))
w17=pi/(t(812)-t(807))
M17=(y2(814)-y2(810))/(u(812)-u(807))
Ph17=rad2deg(w17*(t(807)-t(810)))

w18=pi/(t(990)-t(986))
M18=(y2(989)-y2(993))/(u(986)-u(990))
Ph18=rad2deg(w18*(t(986)-t(989)))

md=mean([M9,M11,M12,M14,M15,M17])
mp=mean([Ph9,Ph11,Ph12,Ph14,Ph15,Ph17])
w=logspace(4,6);
[num,den]=tfdata(H2,'v');
[M,Ph]=bode(num,den,w);
subplot(211)
semilogx([w1,w2,w3,wr2,w4,w5,w6,w7,w8,wr2,w10,w9,w13],20*log10([M1,M2,M3,Mr2,M4,M5,M6,M7,M8,Mr2,M10,md,mean([M13,M16,M18])]),'o');
hold on
semilogx(w,20*log10(M)); grid
title('Caracteristica de modul')
xlabel('Frecventa (rad/s)');
ylabel('Modul (dB)');

subplot(212)
semilogx([w1,w3,w4,w5,w8,wr2,w10,w9,w13],[mean([Ph1,Ph2]),Ph3,Ph4,mean([Ph5,Ph6,Ph7]),Ph8,Phr2,Ph10,mp,mean([Ph13,Ph16,Ph18])],'o')
hold on
semilogx(w,Ph); grid
title('Caracteristica de faza')
xlabel('Frecventa (rad/s)');
ylabel('Faza (grade)');
p=20*log10(M18/M10)/(w18/w10) *10
%%
y1=y1(2:end);
u=u(2:end);
t=t(2:end)
%%
%export to pdf
dt=t(2)-t(1);
datay1=iddata(y1,u,dt)

modarxy1=arx(datay1,[2 2 0]) %3 3 1  2 3 0
figure
resid(datay1,modarxy1,10) %6
figure
compare(datay1,modarxy1)
[num,den]=tfdata(modarxy1,"v")
%in continu la d2c nu putem folosi zoh trebuie adaugata acea intarziere
Hz=tf(num,den,dt)
[A,B,C,D]=tf2ss(num,den)
%Hs=d2c(Hz,'foh')
%nums,dens]=tfdata(Hs,'v')
x1=pinv(B')*(y1(1)-D*u(1))
x2=A'*x1+C'*u(1)
ycarx1z=dlsim(A',C',B',D,u,[x1(1,:),x2(2,:)]);

%ycarx1s=lsim(As',Cs',Bs',Ds,u,t,[y1(1),(y1(3)-y1(2))/(t(5)-t(4))])
figurehelp
plot(t,[y1,ycarx1z])
empnarx1=norm(y1-ycarx1z)/norm(y1-mean(y1))
title('Compararea y1 cu simularea modelului arx in discret')
xlabel('Timp[sec]')
ylabel('Volt[V]')
legend('y1','ycarx1z')
%figure
%plot(t,[y1,ycarx1s])
%empnarx1s=norm(y1-ycarx1s)/norm(y1-mean(y1))
%title('Compararea y1 cu simularea modelului arx in continuu')
%xlabel('Timp[sec]')
%ylabel('Volt[V]')
%legend('y1','ycarx1s')
%%
datay1=iddata(y1,u,dt)
modarmaxy1=armax(datay1,[2 2 2 1])
figure
resid(datay1,modarmaxy1,15)
figure
compare(datay1,modarmaxy1)

[num,den]=tfdata(modarmaxy1,"v")
Hz=tf(num,den,dt)
%Hs=d2c(Hz,'foh')
%[nums,dens]=tfdata(Hs,'v')
[A,B,C,D]=tf2ss(num,den)
%[As,Bs,Cs,Ds]=tf2ss(nums,dens)
x1=pinv(B')*(y1(1)-D*u(1))
x2=A'*x1+C'*u(1)
ycarmax1z=dlsim(A',C',B',D,u,[x1(1,:),x2(2,:)]);
%ycarmax1z=dlsim(A',C',B',D,u,[y1(2),-0.8])
%ycarmax1s=lsim(As',Cs',Bs',Ds,u,t,[y1(1),(y1(1)-y1(2))/(t(5)-t(4))])
figure
plot(t,[y1,ycarmax1z])
empnarmax1=norm(y1-ycarmax1z)/norm(y1-mean(y1))
title('Compararea y1 cu simularea modelului armax in discret')
xlabel('Timp[sec]')
ylabel('Volt[V]')
legend('y1','ycarmax1z')
%figure
%plot(t,[y1,ycarmax1s])
%empnarx1s=norm(y1-ycarmax1s)/norm(y1-mean(y1))
%title('Compararea y1 cu simularea modelului armax in continuu')
%xlabel('Timp[sec]')
%ylabel('Volt[V]')
%legend('y1','ycarmax1s')
%%
dt=t(2)-t(1);
datay2=iddata(y2,u,dt)
modarxy2=arx(datay2,[3 2 0])
H2=tf(modarxy2.B,modarxy2.A)
figure
resid(datay2,modarxy2,10)
figure
compare(datay2,modarxy2)
[num,den]=tfdata(modarxy2, ...
    "v")
Hz=tf(num,den,dt)
Hs=d2c(Hz,'foh')
[nums,dens]=tfdata(Hs,'v')
[A,B,C,D]=tf2ss(num,den)
x1=pinv(B')*(y2(1)-D*u(1))
x2=A'*x1+C'*u(1)
x3=A'*x2+C'*u(2)
%[As,Bs,Cs,Ds]=tf2ss(nums,dens)
% d1=(y2(2)-y2(1))/dt;
% d2=(d1-y2(2))/dt
% d3=(d2-y2(3))/dt
ycarx2z=dlsim(A',C',B',D,u,[x1(1,:),x3(2,:),x3(3,:)])
%ycarx2s=lsim(As',Cs',Bs',Ds,u,t,[y2(1),d1,y2(2),d2])
figure
plot(t,[y2,ycarx2z])
empnarx2=norm(y2-ycarx2z)/norm(y2-mean(y2))
title('Compararea y2 cu simularea modelului arx in discret')
xlabel('Timp[sec]')
ylabel('Volt[V]')
legend('y1','ycarx2z')
%figure
%plot(t,[y2,ycarx2s])
%empnarx2=norm(y2-ycarx2s)/norm(y2-mean(y2))
%title('Compararea y2 cu simularea modelului armax in continuu')
%xlabel('Timp[sec]')   
%ylabel('Volt[V]')
%legend('y1','ycarmax2s')

%%
% Semnal de intrare si iesire

% Calculul FFT
U = fft(u);
Y = fft(y1);

% Estimarea functiei de transfer
H = Y ./ U;

% Plotare Nyquist
figure;
plot(real(H), imag(H));
hold on;
plot(real(H), -imag(H)); % Simetria Nyquist
xlabel('Re(H(jw))');
ylabel('Im(H(jw))');
title('Diagrama Nyquist');
axis equal;
grid on;

%%
modarmaxy2=armax(datay2,[2 3 2 0])
figure
resid(datay2,modarmaxy2,10)
figure
compare(datay2,modarmaxy2)
[num,den]=tfdata(modarmaxy2,"v")
Hz=tf(num,den,dt)
%Hs=d2c(Hz)
%[nums,dens]=tfdata(Hs,'v')
[A,B,C,D]=tf2ss(num,den)
x1=pinv(B')*(y2(1)-D*u(1))
x2=A'*x1+C'*u(1)
ycarmax2z=dlsim(A',C',B',D,u,[x1(1,:),x2(2,:)])
figure
plot(t,[y2,ycarmax2z])
empnarx2=norm(y2-ycarmax2z)/norm(y2-mean(y2))
title('Compararea y2 cu simularea modelului armax in discret')
xlabel('Timp[sec]')
ylabel('Volt[V]')
legend('y1','ycarmax1z')

%%
datay1=iddata(y1,u,dt)
modoey1=oe(datay1,[3 3 1])
figure
resid(datay1,modoey1,12)
figure
compare(datay1,modoey1)
%%
datay1=iddata(y1,u,dt)
modivy1=iv4(datay1,[3 3 1])
figure
resid(datay1,modivy1,12)
figure
compare(datay1,modivy1)
%%
modoey2=oe(datay2,[3 3 1])
figure
resid(datay2,modoey2,12)
figure
compare(datay2,modoey2)
%%
modivy2=iv4(datay2,[2 2 0])
figure
resid(datay2,modivy2,6)
figure
compare(datay2,modivy2)
%%
dt=t(2)-t(1);
dy1=iddata(y1(18:end),u(18:end),dt)
mody1=n4sid(dy1,1:7)
figure
resid(dy1,mody1,5)
figure
compare(dy1,mody1)
%%
dy2=iddata(y2(52:end),u(52:end),dt)
mody2=n4sid(dy2,1:7)
figure
resid(dy2,mody2)
figure
compare(dy2,mody2)
%%
dy12=iddata([y1,y2],u,dt)
mody12=n4sid(dy12,1:7)
%figure
%resid(dy12,mody12)
%figure
%compare(dy12,mody12)
mody12p=pem(dy12,mody12)
%resid(dy12,mody12p,5)
%figure
%compare(dy12,mody12p)
Hzy1=ss(mody12p.A, mody12p.B,mody12p.C(1,:),mody12p.D(1,:),dt)
mody12ps=d2c(mody12p)
%yc1=lsim(mody12ps.A, mody12ps.B,mody12ps.C(1,:),mody12ps.D(1,:),u,t)


yc1=dlsim(mody12p.A, mody12p.B,mody12p.C(1,:),mody12p.D(1,:),u)
figure
plot(t,[y1,yc1])
empn41=norm(y1-yc1)/norm(y1-mean(y1))
yc2=dlsim(mody12p.A, mody12p.B,mody12p.C(2,:),mody12p.D(2,:),u)
figure
plot(t,[y2,yc2])
empn42=norm(y2-yc2)/norm(y2-mean(y2))