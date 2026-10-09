%  Copyright (C) 2018 Texas Instruments Incorporated - http://www.ti.com/
%
%
%   Redistribution and use in source and binary forms, with or without
%   modification, are permitted provided that the following conditions
%   are met:
%
%     Redistributions of source code must retain the above copyright
%     notice, this list of conditions and the following disclaimer.
%
%     Redistributions in binary form must reproduce the above copyright
%     notice, this list of conditions and the following disclaimer in the
%     documentation and/or other materials provided with the
%     distribution.
%
%     Neither the name of Texas Instruments Incorporated nor the names of
%     its contributors may be used to endorse or promote products derived
%     from this software without specific prior written permission.
%
%   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
%   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
%   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
%   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
%   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
%   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
%   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
%   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
%   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
%   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
%   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

% plot_range_azimuth_2D.m
%
% Function to plot range and azimuth heat map

%input
%   range_resolution: range resolution to calculate axis to plot
%   radar_data_pre_3dfft: input 3D matrix, rangeFFT x DopplerFFT x virtualArray
%   TDM_MIMO_numTX: number of TXs used for processing
%   numRxAnt: : number of RXs used for processing
%   antenna_azimuthonly: azimuth array ID
%   LOG: 1:plot non-linear scale, ^0.4 by default
%   STATIC_ONLY: 1 = plot heatmap for zero-Doppler; 0 = plot heatmap for nonzero-Doppler
%   PLOT_ON: 1 = plot on; 0 = plot off
%   minRangeBinKeep: start range index to keep
%   rightRangeBinDiscard: number of right most range bins to discard

%output
%   mag_data_static: zero Doppler range/azimuth heatmap
%   mag_data_dynamic: non-zero Doppler range/azimuth heatmap
%   y_axis: y axis used for visualization
%   x_axis: x axis used for visualization


% The transfer between FFT HZ unit to Meter unit
function  [mag_data_static mag_data_dynamic radar_data_angle_range_whole y_axis x_axis indSel] = plot_range_azimuth_2D(range_resolution, radar_data_pre_3dfft,TDM_MIMO_numTX,numRxAnt,...
    antenna_azimuthonly, LOG, STATIC_ONLY, PLOT_ON, minRangeBinKeep,  rightRangeBinDiscard)
%{
fprintf('\n----------------------1\n');
disp(radar_data_pre_3dfft(1, 1, 10));
fprintf('\n----------------------2\n');
disp(radar_data_pre_3dfft(5, 5, 20))
fprintf('\n----------------------3\n');
%}

dopplerFFTSize = size(radar_data_pre_3dfft,2);
rangeFFTSize = size(radar_data_pre_3dfft,1);
angleFFTSize = 256;
% ratio used to decide engergy threshold used to pick non-zero Doppler bins
ratio = 0.5; 
DopplerCorrection = 0;

% Correcting the data caused by doppler effect
if DopplerCorrection == 1 % This one is not used !!!!!!!!!!
    % add Doppler correction before generating the heatmap
    radar_data_pre_3dfft_DopCor= [];
    for dopplerInd = 1: dopplerFFTSize
        deltaPhi = 2*pi*(dopplerInd-1-dopplerFFTSize/2)/( TDM_MIMO_numTX*dopplerFFTSize);
        sig_bin_org =squeeze(radar_data_pre_3dfft(:,dopplerInd,:));
        for i_TX = 1:TDM_MIMO_numTX
            RX_ID = (i_TX-1)*numRxAnt+1 : i_TX*numRxAnt;
            corVec = repmat(exp(-1j*(i_TX-1)*deltaPhi), rangeFFTSize, numRxAnt);
            radar_data_pre_3dfft_DopCor(:,dopplerInd, RX_ID)= sig_bin_org(:,RX_ID ).* corVec;
        end
    end
    radar_data_pre_3dfft = radar_data_pre_3dfft_DopCor;
end
%fprintf('+ %d +:', antenna_azimuthonly);
%{
fprintf('--------------------1111--Check_Angle_AZIM0---------------\n');
disp(radar_data_pre_3dfft(1, 1:2, 1:2));
fprintf('\n SHAPE + %d +:', size(radar_data_pre_3dfft));
disp(radar_data_pre_3dfft(1, 20:21, 20:21));
fprintf('\n--------------------2222--Check_Angle_AZIM0---------------\n');
%}
radar_data_pre_3dfft = radar_data_pre_3dfft(:,:,antenna_azimuthonly);
radar_data_angle_range = fft(radar_data_pre_3dfft, angleFFTSize, 3);
%{
fprintf('--------------------11111--Check_Angle_Matrix_FFT2---------------\n');
disp(radar_data_angle_range(2, 2, 10));
disp(radar_data_angle_range(4, 4, 20));
fprintf('\n--------------------22222--Check_Angle_Matrix_FFT2---------------\n');
%}
n_angle_fft_size = size(radar_data_angle_range,3);
n_range_fft_size = size(radar_data_angle_range,1);

%decide non-zerp doppler bins to be used for dynamic range-azimuth heatmap
% Size: 1, Range_Size, Doppler_Size, Angular_Size
DopplerPower = sum(mean((abs(radar_data_pre_3dfft(:,:,:))),3),1); 
% Size: 1, Doppler_Size
% Only keep the second dimension
DopplerPower_noDC = DopplerPower([1:dopplerFFTSize/2-1 dopplerFFTSize/2+3:end]);
% The relation between angle and fft is frequency=d*cos(\theta)/wavelength
% considering the IT instrument, fre = 0.5 * cos(\theta)
[peakVal peakInd] = max(DopplerPower_noDC);
threshold = peakVal*ratio;
indSel = find(DopplerPower_noDC >threshold);
% ------------- Doppler -----------
%fprintf(' + %d +:', indSel);
for ii = 1:length(indSel)
    if indSel(ii) > dopplerFFTSize/2-1
        indSel(ii) = indSel(ii) + 3; 
        % To make it is correct and usable in velocity computation.
    end
end

%radar_data_angle_range: NumChirp, NumSample, RxTx
radar_data_angle_range_dynamic = squeeze(sum(abs(radar_data_angle_range(:,indSel,:)),2)); % Only keep Dynamic Part -> containing dynamic things
radar_data_angle_range_whole = abs(radar_data_angle_range);

radar_data_angle_range_Static = squeeze(sum(abs(radar_data_angle_range(:,dopplerFFTSize/2+1,:)),2)); % Only Keep Static Part

%fprintf(' + %d +:', size(radar_data_angle_range_Static));
indices_1D = (minRangeBinKeep:n_range_fft_size-rightRangeBinDiscard);
% fprintf('------------------------2');
% fprintf('%d-',indices_1D);
% Changing the range from 15 to 10m
%indices_1D = (minRangeBinKeep:2/3*(n_range_fft_size-rightRangeBinDiscard-minRangeBinKeep)+minRangeBinKeep);
% fprintf('------------------------2');
% fprintf('%d-',indices_1D);

%max_range = (n_range_fft_size-1)*range_resolution;
%max_range = max_range/2;
d = 1;

%generate range/angleFFT for zeroDoppler and non-zero Doppler respectively
%fprintf(' + %d +:', size(radar_data_angle_range_dynamic));
%fprintf('\n----------------------1-------------------------\n');
radar_data_angle_range_dynamic = fftshift(radar_data_angle_range_dynamic,2);
%fprintf(' + %d +:', size(radar_data_angle_range_dynamic));
%fprintf('\n----------------------2-------------------------\n');
radar_data_angle_range_whole = fftshift(radar_data_angle_range_whole,3);
radar_data_angle_range_Static = fftshift(radar_data_angle_range_Static,2); 
% only keep the antenna dimen


sine_theta = -2*((-n_angle_fft_size/2:n_angle_fft_size/2)/n_angle_fft_size)/d; 

% sine_theta = f_s * lambda/d = n/N*2
% -1 ~ 1: Indicating from -pi/2 ~ pi/2
cos_theta = sqrt(1-sine_theta.^2); % -1 ~ 1

[R_mat, sine_theta_mat] = meshgrid(indices_1D*range_resolution,sine_theta); 
% This one is used to transfer frequency to meter
% Key item: Range_resolution
[~, cos_theta_mat] = meshgrid(indices_1D,cos_theta); 
% R_mat: size(size(sine_theta), sine(indices_1d)) -- x-axis is repeat, y
% aixs is the range from (indices_1D*range_resolution)
% cos_theta_mat: size(size(sine_theta, sine(indices_1d))) -- x-axis is the
% range from sine_theta, y axis is repeated, 

%{
fprintf('--------------1\n');
fprintf('%d--', size(indices_1D));
fprintf('--------------2\n');
fprintf('%d--', size(sine_theta));
fprintf('--------------3\n');
fprintf('%d--', size(R_mat));
fprintf('--------------4\n');
fprintf('%d--', size(sine_theta_mat));
fprintf('--------------5\n');
%}
%{
fprintf('\n 0-------------------------\n');
fprintf(' + %d +:', size(radar_data_angle_range_dynamic));
%}
x_axis = R_mat.*cos_theta_mat; % The transfer for angle
y_axis = R_mat.*sine_theta_mat;
mag_data_dynamic = squeeze(abs(radar_data_angle_range_dynamic(indices_1D+1,[1:end 1])));
radar_data_angle_range_whole = abs(radar_data_angle_range_whole(indices_1D+1,:,[1:end 1]));
mag_data_static = squeeze(abs(radar_data_angle_range_Static(indices_1D+1,[1:end 1])));

%fprintf('- %d -:', size(mag_data_static));
%fprintf('\n-------------------------\n');

%{
fprintf('\n 1-------------------------\n');
fprintf(' + %d +:', size(mag_data_dynamic));
fprintf('\n 2-------------------------\n');
fprintf(' + %d +:', size(radar_data_angle_range_whole));
fprintf('\n 3-------------------------\n');
fprintf(' + %d +:', size(mag_data_static));
%}
mag_data_dynamic = mag_data_dynamic'; % caused  by meshgrid
radar_data_angle_range_whole = permute(radar_data_angle_range_whole, [3, 1, 2]);
mag_data_static = mag_data_static';
%fprintf('-------------------------1\n');
%fprintf(' + %d +:', size(mag_data_dynamic));
%{
fprintf('\n 11-------------------------\n');
fprintf(' + %d +:', size(mag_data_dynamic));
fprintf('\n 22-------------------------\n');
fprintf(' + %d +:', size(radar_data_angle_range_whole));
fprintf('\n 33-------------------------\n');
fprintf(' + %d +:', size(mag_data_static));
%}
mag_data_dynamic = flipud(mag_data_dynamic);
radar_data_angle_range_whole = flipud(radar_data_angle_range_whole);
mag_data_static = flipud(mag_data_static);


sum_result = sum(mag_data_static, 1);
[max_value, max_index] = max(sum_result);
disp((max_index+minRangeBinKeep)*range_resolution);

%{
fprintf('\n 111-------------------------\n');
fprintf(' + %d +:', size(mag_data_dynamic));
fprintf('\n 222-------------------------\n');
fprintf(' + %d +:', size(radar_data_angle_range_whole));
fprintf('\n 333-------------------------\n');
fprintf(' + %d +:', size(mag_data_static));
%}
if PLOT_ON
    log_plot = LOG;
    if STATIC_ONLY == 1
        %fprintf('---------------------Only Keep the Static Part');
        if log_plot
            %surf(y_axis, x_axis, (mag_data_static(y_indice)).^0.4,'EdgeColor','none');
            % Usually this case.
            %fprintf(' max + %d +: \n', max(mag_data_static));
            %%fprintf(' min + %d +: \n', min(mag_data_static));
            %fprintf(' mean + %d +: \n', mean(mag_data_static));
            surf(y_axis, x_axis, (mag_data_static).^0.4,'EdgeColor','none');
            %fprintf('-------------Ending-----------------');
        else
            surf(y_axis, x_axis, abs(mag_data_static),'EdgeColor','none');
        end
    else
        if log_plot
            surf(y_axis, x_axis, (mag_data_dynamic).^0.4,'EdgeColor','none'); % Better Understanding is needed.
        else
            surf(y_axis, x_axis, abs(mag_data_dynamic),'EdgeColor','none'); % Better Understanding is needed.
        end
    end
    view(2);
    xlabel('meters')
    ylabel('meters')
end
end