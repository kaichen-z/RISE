
%clearvars
%close all

PLOT_ON = 0; % 1: turn plot on; 0: turn plot off
LOG_ON = 1; % 1: log10 scale; 0: linear scale
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

while ~feof(fidList)
    %% get each test vectors within the test list
    % test data file name
    dataFolder_test = fgetl(fidList);        
    % Extract the directory path from the file path
    % -------------------- Create Document --------------------
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

    %calibration file name
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

    % get system level variables
    platform            = simTopObj.platform;
    numValidFrames      = simTopObj.totNumFrames;
    cnt = 1;
    frameCountGlobal = 0;
    
    % Get Unique File Idxs in the "dataFolder_test"   
    [fileIdx_unique] = getUniqueFileIdx(dataFolder_test);
    
    for i_file = 1:(length(fileIdx_unique))
       [fileNameStruct]= getBinFileNames_withIdx(dataFolder_test, fileIdx_unique{i_file});        
       calibrationObj.binfilePath = fileNameStruct;
        
       detection_results = [];  
        
       [numValidFrames dataFileSize] = getValidNumFrames(fullfile(dataFolder_test, fileNameStruct.masterIdxFile));
       for frameIdx = 2:1:numValidFrames;
            tic;          
            calibrationObj.frameIdx = frameIdx;
            frameCountGlobal = frameCountGlobal+1;
            adcData = datapath(calibrationObj); 
            adcData = adcData(:,:,calibrationObj.RxForMIMOProcess,:);            
            
            run(pathGenParaFile);
            % --------------------------Range----------------------
            minRangeBinKeep =  5;
            rightRangeBinDiscard =  20;

            rangeWinLen                 = length(rangeProcCascade_rangeWindowCoeff);
            rangeWindowCoeffVec         = ones(rangeProcCascade_numAdcSamplePerChirp, 1);
            rangeWindowCoeffVec(1:rangeWinLen) = rangeProcCascade_rangeWindowCoeff;
            rangeWindowCoeffVec(rangeProcCascade_numAdcSamplePerChirp-rangeWinLen+1:rangeProcCascade_numAdcSamplePerChirp) = rangeWindowCoeffVec(rangeWinLen:-1:1);
            
            % --------------------------Doppler----------------------
            dopplerWinLen               = length(DopplerProcClutterRemove_dopplerWindowCoeff);
            dopplerWindowCoeffVec       = ones(DopplerProcClutterRemove_numChirpsPerVirAnt, 1);
            dopplerWindowCoeffVec(1:dopplerWinLen) = DopplerProcClutterRemove_dopplerWindowCoeff;
            dopplerWindowCoeffVec(DopplerProcClutterRemove_numChirpsPerVirAnt-dopplerWinLen+1:DopplerProcClutterRemove_numChirpsPerVirAnt) = dopplerWindowCoeffVec(dopplerWinLen:-1:1);

            dict = struct('rangeWindowCoeffVec', rangeWindowCoeffVec,  ...
                    'rangeFFTSize', rangeProcCascade_rangeFFTSize, ...
                    'rangeBinSize', rangeBinSize, ...
                    'dopplerWindowCoeffVec', dopplerWindowCoeffVec, ...
                    'dopplerFFTSize', DopplerProcClutterRemove_dopplerFFTSize, ...
                    'velocityBinSize', velocityBinSize, ...
                    'antenna_azimuthonly', antenna_azimuthonly, ...
                    'antenna_azimuth_elev', DOACascade_D, ...
                    'minRangeBinKeep', minRangeBinKeep,...
                    'rightRangeBinDiscard', rightRangeBinDiscard);
            jsonStr = jsonencode(dict);
            filename = sprintf('%s\\original_hyp_%d_%d.json', filename_prefix, i_file, cnt);
            fid = fopen(filename, 'w');
            fwrite(fid, jsonStr, 'char');
            fclose(fid);
            %----------------SAVE 3D VOLUME--------------------
            filename = sprintf('%s\\original_mat_%d_%d.mat', filename_prefix, i_file, cnt);
            save(filename, 'adcData');


            cnt = cnt + 1;    
            toc; % Time consuming    
       end
       fprintf('-------------------11');
    end
    ind = strfind(dataFolder_test, '\');
    testName = dataFolder_test(ind(end-1)+1:(ind(end)-1));
    if SAVEOUTPUT_ON == 1
        save(['.\main\cascade\output\newOutput_',testName,'.mat'],'angles_all_all', 'detection_results_all','xyz_all');
    end
    testID = testID + 1;
end
