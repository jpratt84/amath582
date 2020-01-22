% AMATH 582 HW 1
clear all; close all; clc;
load Testdata

%% Determine Central Freq. of Marble, De-noise and Filter Signal
L = 15; % spatial domain
n = 64; % Fourier modes
x2 = linspace(-L,L,n+1); x = x2(1:n); y=x; z=x;
k = (2*pi/(2*L))*[0:(n/2-1) -n/2:-1]; ks = fftshift(k); %scale wavenumbers; FFT assumes 2pi periodicity

[X,Y,Z] = meshgrid(x,y,z); % discretize spatial domain
[Kx,Ky,Kz] = meshgrid(ks,ks,ks); % discretize frequency domain
Uave = zeros(n,n,n); % 3-D matrix of zeros for storing average signal values

for j=1:20
    Un(:,:,:) = reshape(Undata(j,:),n,n,n); % row, column, slice
    Utn = fftshift(fftn(Un)); %transform data from spatial to freq. domain
    Uave = Uave + Utn; %average each of the 20 measurement sets to minimize white noise
end

%Plot normalized Uave data as isosurface in freq. domain
figure(1),
close all, isosurface(Kx,Ky,Kz,abs(Uave)/max(abs(Uave(:))),0.6)
axis([-6 6 -6 6 -6 6]), grid on;
xlabel('Kx'), ylabel('Ky'), zlabel('Kz')
title('Isosurface of Averaged Signal in Freq. Domain');
[M,I] = max(Uave(:)) %max value of Uave, index of max value of Uave
% M = 5.4258e+03 + 3.5367e+02i
% I = 133724
[row, col, slice] = ind2sub(size(Uave),I) %convert linear index to row, col, slice
% row = 28, col = 42, slice = 33

kx0 = ks(col); %central freq. in Kx
ky0 = ks(row); %central freq. in Ky
kz0 = ks(slice); %central freq. in Kz

% Construct 3-D Gaussian filter around central frequency listed above
tau = 0.35; % bandwidth of filter
filter = exp(-tau*((Kx-kx0).^2+(Ky-ky0).^2 + (Kz-kz0).^2));

% Isosurface of filtered, normalized Uave data in freq. domain
figure(2),
isosurface(Kx,Ky,Kz,filter.*abs(Uave)/max(abs(Uave(:))),0.6)
axis([-6 6 -6 6 -6 6]), grid on
xlabel('Kx'), ylabel('Ky'), zlabel('Kz')
title('Isosurface of Filtered Avg. Signal in Freq. Domain');

%% Compute Trajectory of Marble after Filtering

% Create vectors to store spatial position data after filtering
rowf = zeros(1,20); 
colf = zeros(1,20);
slicef = zeros(1,20);

figure(3),
for i=1:20
    Un(:,:,:) = reshape(Undata(i,:),n,n,n);
    Utn = fftshift(fftn(Un)); % apply 3-D FFT to noisy spatial data, transform to freq. domain
    Utnf = filter.*Utn; % apply 3-D Gaussian filter around central freq. to noisy data
    Unf = ifftn(Utnf); % transform filtered data back to spatial domain
    [mx,idx] = max(Unf(:)); %obtain max value of filtered spatial data, and index of value
    [row,col,slice] = ind2sub(size(Unf),idx); %convert index to row, col, slice
    rowf(i) = row; %store row index for each of 20 measurements
    colf(i) = col;
    slicef(i) = slice;
    isosurface(X,Y,Z,abs(Unf),0.4) % plots position of marble for each of the 20 measurements as an isosurface
    axis([-15 15 -15 15 -15 15]), grid on, drawnow
    pause(0.1)
    xlabel('X'), ylabel('Y'), zlabel('Z')
    title('Marble Trajectory over Time')
end

figure(4),
plot3(x(colf),y(rowf),z(slicef),'-o') %plot position of marble based on spatial position coordinates obtained above
grid on;
axis([-15 15 -15 15 -15 15]);
xlabel('x'),ylabel('y'),zlabel('z');
title('Marble Trajectory')

% Compute final position of marble
x20 = x(colf(20))
y20 = y(rowf(20))
z20 = z(slicef(20))