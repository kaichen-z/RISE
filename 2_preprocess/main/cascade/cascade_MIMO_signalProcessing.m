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
%
%

% cascade_MIMO_signalProcessing.m
%
% Top level main test chain to process the raw ADC data. The processing
% chain including adc data calibration module, range FFT module, DopplerFFT
% module, CFAR module, DOA module. Each module is first initialized before
% actually used in the chain.


%{
Question:
1. understanding deep about the range-angule.
2. understanding better the object detection process.
%}

clearvars
close all

PLOT_ON = 1; % 1: turn plot on; 0: turn plot off
LOG_ON = 1; % 1: log10 scale; 0: linear scale
% numFrames_toRun = 10; %number of frame to run, can be less than the frame saved in the raw data
SAVEOUTPUT_ON = 0;
PARAM_FILE_GEN_ON = 1;
DISPLAY_RANGE_AZIMUTH_DYNAMIC_HEATMAP = 0 ; % Will make things slower
dataPlatform = 'TDA2';

%% get the input path and testList
pro_path = getenv('CASCADE_SIGNAL_PROCESSING_CHAIN_MIMO');
input_path = strcat(pro_path,'\main\cascade\input\');
testList = strcat(input_path,'testList.txt');
%path for input folder
fidList = fopen(testList,'r');
testID = 1;
%{
filename_prefix = 'C:\\kaichen\\ghost\\ti\\mmwave_studio_03_00_00_14\\mmWaveStudio\\PostProc\\Cascade_Capture_22xx_test4\\preprocess\\';
if ~exist(filename_prefix, 'dir')
    mkdir(filename_prefix);
    fprintf('Directory "%s" created.\n', filename_prefix);
else
    fprintf('Directory "%s" already exists.\n', filename_prefix);
end
%}
while ~feof(fidList)
    
    %% get each test vectors within the test list
    % test data file name
    dataFolder_test = fgetl(fidList);    
    %calibration file name
    [parentDir, ~, ~] = fileparts(dataFolder_test);
    filename_prefix = fullfile(parentDir, 'preprocess');
    % Check if the new directory exists
    if ~exist(filename_prefix, 'dir')
        % Create the new directory
        mkdir(filename_prefix);
        disp(['Directory created: ', filename_prefix]);
    else
        disp(['Directory already exists: ', filename_prefix]);
    end

    dataFolder_calib = fgetl(fidList);
    %module_param_file defines parameters to init each signal processing
    %module
    module_param_file = fgetl(fidList);
     %parameter file name for the test
    pathGenParaFile = [input_path, 'test', num2str(testID), '_param.m'];
    %important to clear the same.m file, since Matlab does not clear cache
    %automatically
    clear(pathGenParaFile);
    %generate parameter file for the test to run
    if PARAM_FILE_GEN_ON == 1     
        parameter_file_gen_json(dataFolder_test, dataFolder_calib, module_param_file, pathGenParaFile, dataPlatform);
    end
    %load calibration parameters
    load(dataFolder_calib)
    
    % simTopObj is used for top level parameter parsing and data loading and saving
    simTopObj           = simTopCascade('pfile', pathGenParaFile);
    calibrationObj      = calibrationCascade('pfile', pathGenParaFile, 'calibrationfilePath', dataFolder_calib);
    rangeFFTObj         = rangeProcCascade('pfile', pathGenParaFile);
    DopplerFFTObj       = DopplerProcClutterRemove('pfile', pathGenParaFile);
    detectionObj        = CFAR_CASO('pfile', pathGenParaFile);
    DOAObj              = DOACascade('pfile', pathGenParaFile);
    
    % get system level variables
    platform            = simTopObj.platform;
    numValidFrames      = simTopObj.totNumFrames;
    cnt = 1;
    frameCountGlobal = 0;
    
    % Get Unique File Idxs in the "dataFolder_test"   
    [fileIdx_unique] = getUniqueFileIdx(dataFolder_test);
    
    for i_file = 1:(length(fileIdx_unique))
       %fprintf(' + %d +:', i_file);
       %fprintf('----------------', length(fileIdx_unique));
       % Get File Names for the Master, Slave1, Slave2, Slave3   
       [fileNameStruct]= getBinFileNames_withIdx(dataFolder_test, fileIdx_unique{i_file});        
       %fprintf(' + %d +:', fileNameStruct);
       %pass the Data File to the calibration Object
       calibrationObj.binfilePath = fileNameStruct;
        
       detection_results = [];  
        
       % Get Valid Number of Frames 
       [numValidFrames dataFileSize] = getValidNumFrames(fullfile(dataFolder_test, fileNameStruct.masterIdxFile));
       %intentionally skip the first frame due to TDA2 
       
       for frameIdx = 2:1:numValidFrames;%numFrames_toRun
            tic; % Time consuming    
            %read and calibrate raw ADC data            
            calibrationObj.frameIdx = frameIdx;
            frameCountGlobal = frameCountGlobal+1;
            adcData = datapath(calibrationObj); 
            % numSamplePerChirp, numChirpsPerFrame, RX, TX
            % RX Channel re-ordering
            adcData = adcData(:,:,calibrationObj.RxForMIMOProcess,:);            
            %only take TX and RXs required for MIMO data analysis
            % adcData = adcData
            if mod(frameIdx, 10)==1
                fprintf('Processing %3d frame...\n', frameIdx);
            end
            %perform 2D FFT
            rangeFFTOut = [];
            DopplerFFTOut = [];             
            %{
            fprintf('--------------------11--Check_Input_Matrix---------------\n');
            disp(adcData(11, 33, 11, 11));
            disp(adcData(11, 11, 11, 11));
            fprintf('\n--------------------22--Check_Input_Matrix---------------\n');
            %}
            for i_tx = 1: size(adcData,4)
                % range FFT        
                rangeFFTOut(:,:,:,i_tx) = datapath(rangeFFTObj, adcData(:,:,:,i_tx));
                %{
                fprintf('---%d', size(rangeFFTOut(:,1,1,1)));
                fprintf('---%d', size(0:length(rangeFFTOut(:,1,1,1))));
                plot(0:length(rangeFFTOut(:,1,1,1))-1,abs(rangeFFTOut(:,1,1,1)),"LineWidth",3)
                title("Complex Magnitude of fft Spectrum")
                xlabel("f (Hz)")
                ylabel("|fft(X)|")
                %}
                %DopplerFFTOut(:,:,:,i_tx) = datapath(DopplerFFTObj, rangeFFTOut(:,:,:,i_tx));
                % Along the second dimension ----------
            end       
            for i_tx = 1: size(adcData,4)
                DopplerFFTOut(:,:,:,i_tx) = datapath(DopplerFFTObj, rangeFFTOut(:,:,:,i_tx));
            end 
            %{
            fprintf('--------------------111--Check_Doppler_Matrix_Shift0---------------\n');
            disp(DopplerFFTOut(11, 33, 11, 11));
            disp(DopplerFFTOut(11, 11, 11, 11));
            disp(DopplerFFTOut(11, 51, 11, 11));
            fprintf('\n--------------------222--Check_Doppler_Matrix_Shift0---------------\n');
            %}
            % CFAR done along only TX and RX used in MIMO array
            DopplerFFTOut = reshape(DopplerFFTOut,size(DopplerFFTOut,1), size(DopplerFFTOut,2), size(DopplerFFTOut,3)*size(DopplerFFTOut,4));
            %{
            fprintf('--------------------111--Check_Doppler_Matrix_Shift0---------------\n');
            disp(DopplerFFTOut(2, 2, 10));
            disp(DopplerFFTOut(4, 4, 50));
            fprintf('\n--------------------222--Check_Doppler_Matrix_Shift0---------------\n');
            %}
            % dimension (numSample, numChirps, RX * TX)
            %detection  
            sig_integrate = 10*log10(sum((abs(DopplerFFTOut)).^2,3) + 1); % result for plotting          
            detection_results = datapath(detectionObj, DopplerFFTOut); % needs deeper understanding
            %fprintf('-------------2--------------');            
            detection_results_all{cnt} =  detection_results;   
            detect_all_points = [];
            for iobj = 1:length(detection_results)
                detect_all_points (iobj,1)=detection_results(iobj).rangeInd+1;
                detect_all_points (iobj,2)=detection_results(iobj).dopplerInd_org+1;
                detect_all_points (iobj,4)=detection_results(iobj).estSNR;
            end         
                        
            if PLOT_ON
                figure(1);
                set(gcf,'units','normalized','outerposition',[0 0 1 1])        
                % ------------------------------------------- FIGURE1
                % ------------------------------------------- Only plot
                % signal based on the detection results.
                subplot(2,2,1)               
                plot((1:size(sig_integrate,1))*detectionObj.rangeBinSize, sig_integrate(:,size(sig_integrate,2)/2+1),'g','LineWidth',4);hold on; grid on
                % The up green line
                for ii=1:size(sig_integrate,2) 
                    % Along numChirps 
                    % Plotting along different receiver
                    plot((1:size(sig_integrate,1))*detectionObj.rangeBinSize, sig_integrate(:,ii));hold on; grid on
                    if ~isempty(detection_results)
                        ind = find(detect_all_points(:,2)==ii);
                        if (~isempty(ind))
                            rangeInd = detect_all_points(ind,1);
                            % Plotting objects dot along line
                            plot(rangeInd*detectionObj.rangeBinSize, sig_integrate(rangeInd,ii),'o','LineWidth',2,...
                                'MarkerEdgeColor','k',...
                                'MarkerFaceColor',[.49 1 .63],...
                                'MarkerSize',6);
                        end
                    end
                end
                %title(['FrameID: ' num2str(cnt)]);
                xlabel('Range(m)');
                ylabel('Receive Power (dB)')
                title(['Range Profile(zero Doppler - thick green line): frameID ' num2str(frameIdx)]);
                hold off;

                subplot(2,2,2);
                %subplot_tight(2,2,2,0.1)
                imagesc((sig_integrate))
                c = colorbar;
                c.Label.String = 'Relative Power(dB)';
                title(' Range/Velocity Plot');
                pause(0.01)
            end
            
            angles_all_points = [];
            xyz = [];
            %if 0
            if ~isempty(detection_results)
                % DOA, the results include detection results + angle estimation results.
                % access data with angleEst{frame}(objectIdx).fieldName
                angleEst = datapath(DOAObj, detection_results);
                if length(angleEst) > 0
                    for iobj = 1:length(angleEst)
                        angles_all_points (iobj,1:2)=angleEst(iobj).angles(1:2);
                        angles_all_points (iobj,3)=angleEst(iobj).estSNR;
                        angles_all_points (iobj,4)=angleEst(iobj).rangeInd;
                        angles_all_points (iobj,5)=angleEst(iobj).doppler_corr;
                        angles_all_points (iobj,6)=angleEst(iobj).range;
                        %switch left and right, the azimuth angle is flipped
                        xyz(iobj,1) = angles_all_points (iobj,6)*sind(angles_all_points (iobj,1)*-1)*cosd(angles_all_points (iobj,2));
                        xyz(iobj,2) = angles_all_points (iobj,6)*cosd(angles_all_points (iobj,1)*-1)*cosd(angles_all_points (iobj,2));
                        %switch upside and down, the elevation angle is flipped
                        xyz(iobj,3) = angles_all_points (iobj,6)*sind(angles_all_points (iobj,2)*-1);
                        xyz(iobj,4) = angleEst(iobj).doppler_corr;
                        xyz(iobj,9) = angleEst(iobj).dopplerInd_org;
                        xyz(iobj,5) = angleEst(iobj).range;
                        xyz(iobj,6) = angleEst(iobj).estSNR;
                        xyz(iobj,7) = angleEst(iobj).doppler_corr_overlap;
                        xyz(iobj,8) = angleEst(iobj).doppler_corr_FFT;            
                    end
                    angles_all_all{cnt} = angles_all_points;
                    xyz_all{cnt}  = xyz;
                    maxRangeShow = detectionObj.rangeBinSize*rangeFFTObj.rangeFFTSize;
                    %tic
                    if PLOT_ON
                        moveID = find(abs(xyz(:,4))>=0); % wired, but stick to the original code.
                        subplot(2,2,4);                        
                        if cnt==1
                            scatter3(xyz(moveID,1),xyz(moveID,2),xyz(moveID,3),45,(xyz(moveID,4)),'filled');
                            % scatter3(X, Y, Z, S, C)
                            % S is the size of point
                            % C is the color of point
                        else
                            yz = [xyz_all{cnt}; xyz_all{cnt-1}];
                            scatter3(xyz(moveID,1),xyz(moveID,2),xyz(moveID,3),45,(xyz(moveID,4)),'filled');
                        end
                        FourDim = [xyz(moveID,1),xyz(moveID,2),xyz(moveID,3),xyz(moveID,4)];
                        % ----------SAVE(Point Cloud)--------
                        filename = sprintf('%s\\point_cloud_%d.mat', filename_prefix, cnt);
                        save(filename, 'FourDim');

                        c = colorbar;
                        c.Label.String = 'velocity (m/s)';                        
                        grid on;
                        xlim([-20 20])
                        ylim([1 maxRangeShow])
                        %zlim([-4 4])
                        zlim([-5 5])
                        xlabel('X (m)')
                        ylabel('y (m)')
                        zlabel('Z (m)')                        
                        view([-9 15])                        
                        title(' 3D point cloud');
                        
                        %plot range and azimuth heatmap
                        subplot(2,2,3)
                        STATIC_ONLY = 1;
                        minRangeBinKeep =  5;
                        rightRangeBinDiscard =  20;
                        % The process of angule is conducted here
                        % fprintf('Antenna_Azimuthonly----------%d\n', detectionObj.antenna_azimuthonly);
                        %fprintf('Antenna_Azimuthonly_Length----------%d\n', length(detectionObj.antenna_azimuthonly));
                        % Only making use of the TX*RX combination with
                        % detected object
                        %{
                        fprintf('--------------------111--Check_Doppler_Matrix_Shift---------------\n');
                        disp(DopplerFFTOut(1:2, 1:2, 1));
                        fprintf('\n SHAPE + %d +:', size(DopplerFFTOut));
                        disp(DopplerFFTOut(2, 2, 7:9));
                        fprintf('\n--------------------222--Check_Doppler_Matrix_Shift---------------\n');
                        %}
                        [mag_data_static(:,:,frameCountGlobal) mag_data_dynamic(:,:,frameCountGlobal) mag_data_whole(:,:,:,frameCountGlobal) y_axis x_axis indSel] =... 
                            plot_range_azimuth_2D(detectionObj.rangeBinSize, DopplerFFTOut,...
                            length(calibrationObj.IdTxForMIMOProcess),length(calibrationObj.RxForMIMOProcess), ...
                            detectionObj.antenna_azimuthonly, LOG_ON, STATIC_ONLY, PLOT_ON, minRangeBinKeep, rightRangeBinDiscard);
                        
                        vel_index = (1:detectionObj.dopplerFFTSize);
                        velocity = (vel_index-detectionObj.dopplerFFTSize/2)*detectionObj.velocityBinSize;
                        y_axis_list(:,:,frameCountGlobal) = y_axis;
                        x_axis_list(:,:,frameCountGlobal) = x_axis;
                        velocity_list(:,frameCountGlobal) = velocity;
                        %{
                        fprintf(' + %d +:', size(mag_data_static(:,:,frameCountGlobal)));
                        fprintf('\n 1------------------------------------------------------\n');
                        fprintf(' + %d +:', size(mag_data_dynamic(:,:,frameCountGlobal)));
                        fprintf('\n 2------------------------------------------------------\n');
                        fprintf(' + %d +:', size(mag_data_dynamic_whole(:,:,:,frameCountGlobal)));
                        fprintf('\n 3------------------------------------------------------\n');
                        fprintf(' + %d +:', size(y_axis));
                        fprintf('\n 4------------------------------------------------------\n');
                        fprintf(' + %d +:', size(x_axis));
                        fprintf('\n 5------------------------------------------------------\n');
                        fprintf(' + %d +:', size(velocity));
                        fprintf('\n 6------------------------------------------------------\n');
                        %}
                        title('range/azimuth heat map static objects')
                        % mag_data_static is used for range-angular map
                        %fprintf('----Stop-----%d\n',frameIdx);
                        x = linspace(0, 2*pi, 100);
                        %{
                        surf(y_axis, x_axis, (mag_data_dynamic(:,:,frameCountGlobal)).^0.4,'EdgeColor','none'); 
                        view(2);
                        xlabel('meters');
                        ylabel('meters');
                        saveas(gcf, 'C:\kaichen\ghost\ti\mmwave_studio_03_00_00_14\mmWaveStudio\PostProc\Cascade_Capture_22xx_set1\dynamic.png');

                        surf(y_axis, x_axis, (mag_data_static(:,:,frameCountGlobal)).^0.4,'EdgeColor','none'); 
                        view(2);
                        xlabel('meters');
                        ylabel('meters');
                        saveas(gcf, 'C:\kaichen\ghost\ti\mmwave_studio_03_00_00_14\mmWaveStudio\PostProc\Cascade_Capture_22xx_set1\static.png');
                        fprintf('--------------finish');
                        %}

                        %fprintf('-----------------------------------------SAVED-----------------------------');
                        if (DISPLAY_RANGE_AZIMUTH_DYNAMIC_HEATMAP)                   
                            figure(2)
                            subplot(121);
                            surf(y_axis, x_axis, (mag_data_static(:,:,frameCountGlobal)).^0.4,'EdgeColor','none');
                            view(2);
                            xlabel('meters');    ylabel('meters')
                            title({'Static Range-Azimuth Heatmap',strcat('Current Frame Number = ', num2str(frameCountGlobal))})
                            subplot(122);
                            surf(y_axis, x_axis, (mag_data_dynamic(:,:,frameCountGlobal)).^0.4,'EdgeColor','none');
                            view(2);    
                            xlabel('meters');    ylabel('meters')
                            title('Dynamic HeatMap')
                        end
                        pause(0.1) 

                    end     
                end
            end          
            %{
            %----------------SAVE HYPERPARMETER--------------------
            minRangeBinKeep =  5;
            rightRangeBinDiscard =  20;
            dict = struct('rangeWindowCoeffVec', rangeFFTObj.rangeWindowCoeffVec,  ...
                    'rangeFFTSize', rangeFFTObj.rangeFFTSize, ...
                    'rangeBinSize', detectionObj.rangeBinSize, ...
                    'dopplerWindowCoeffVec', DopplerFFTObj.dopplerWindowCoeffVec, ...
                    'dopplerFFTSize', DopplerFFTObj.dopplerFFTSize, ...
                    'velocityBinSize', detectionObj.velocityBinSize, ...
                    'antenna_azimuthonly',detectionObj.antenna_azimuthonly, ...
                    'antenna_azimuth_elev', DOAObj.D, ...
                    'minRangeBinKeep', minRangeBinKeep,...
                    'rightRangeBinDiscard', rightRangeBinDiscard);
            jsonStr = jsonencode(dict);
            filename = sprintf('%s\\data_%d.json', filename_prefix, cnt);
            fid = fopen(filename, 'w');
            fwrite(fid, jsonStr, 'char');
            fclose(fid);
            %----------------SAVE 3D VOLUME--------------------
            filename = sprintf('%s\\original_mat_%d.mat', filename_prefix, cnt);
            save(filename, 'adcData');
            %}

            cnt = cnt + 1;    
            toc; % Time consuming    
       end
       fprintf('-------------------11');
    end
    %----------------SAVE MATRIX--------------------
    %{
    filename = sprintf('%s\\matrix_data_static.mat', filename_prefix);
    save(filename, 'mag_data_static');
    filename = sprintf('%s\\matrix_data_dynamic.mat', filename_prefix);
    save(filename, 'mag_data_dynamic');
    filename = sprintf('%s\\matrix_data_whole.mat', filename_prefix);
    save(filename, 'mag_data_whole');
    filename = sprintf('%s\\y_axis.mat', filename_prefix);
    save(filename, 'y_axis_list');
    filename = sprintf('%s\\x_axis.mat', filename_prefix);
    save(filename, 'x_axis_list');
    filename = sprintf('%s\\vel_axis.mat', filename_prefix);
    save(filename, 'velocity_list');
    %}

    ind = strfind(dataFolder_test, '\');
    testName = dataFolder_test(ind(end-1)+1:(ind(end)-1));
    if SAVEOUTPUT_ON == 1
        save(['.\main\cascade\output\newOutput_',testName,'.mat'],'angles_all_all', 'detection_results_all','xyz_all');
    end
    testID = testID + 1;
end
