% Define the base directory
% C:\kaichen\ghost\ti\mmwave_studio_03_00_00_14\mmWaveStudio\PostProc\Radar_Data_Scene1\background_eight1_empty_room\
% C:\kaichen\ghost\ti\mmwave_studio_03_00_00_14\mmWaveStudio\MatlabExamples\4chip_cascade_MIMO_example\main\cascade\input\calibrateResults_high.mat
% C:\kaichen\ghost\ti\mmwave_studio_03_00_00_14\mmWaveStudio\MatlabExamples\4chip_cascade_MIMO_example\main\cascade\paramGen\module_param.m
% D:\Raw_Radar_Data_Scene1_7_Important\Raw_Data_2
baseDir = 'C:/kaichen/ghost/ti/mmwave_studio_03_00_00_14/mmWaveStudio/PostProc/Radar_Data_Scene7';
baseDir = 'D:/Raw_Radar_Data_Scene1_7_Important/Raw_Data_7'; 
baseDir = 'E:/Raw_Radar_Data_Scene1_7_Important/Raw_Data_12'; 
% List all folders in the base directory
folderList_iter = dir(baseDir);
folderList_iter = folderList_iter([folderList_iter.isdir]);  % Only keep directories
folderList_iter = folderList_iter(~ismember({folderList_iter.name}, {'.', '..'}));  % Exclude . and ..
% Loop through each folder
for i = 1:length(folderList_iter)
    folderName_iter = folderList_iter(i).name;
    folderPath = fullfile(baseDir, folderName_iter);
    fprintf('Running For -----: %s\n', folderName_iter);
    if ~endsWith(folderPath, '\')
        folderPath = [folderPath, '\'];
    end
    % Modify the first line of input.txt
    inputFilePath = fullfile('C:/kaichen/ghost/ti/mmwave_studio_03_00_00_14/mmWaveStudio/MatlabExamples/4chip_cascade_MIMO_example/main/cascade/input/testList.txt');
    if isfile(inputFilePath)
        fid = fopen(inputFilePath, 'r');
        lines = textscan(fid, '%s', 'Delimiter', '\n'); % 按行读取
        fclose(fid);
        lines = lines{1}; % 获取行内容的单元数组
        lines{1} = folderPath; % 修改第一行内容
        fid = fopen(inputFilePath, 'w');
        fprintf(fid, '%s\n', lines{:}); % 保留所有其他行，只修改第一行
        fclose(fid);
        fprintf('Modified input.txt for folder: %s\n', folderName_iter);
    else
        fprintf('input.txt not found in folder: %s\n', folderName_iter);
        continue;
    end
    % Run pre.m
    try
        %run('C:\kaichen\ghost\ti\mmwave_studio_03_00_00_14\mmWaveStudio\MatlabExamples\4chip_cascade_MIMO_example\main\cascade\Pre_cascade_MIMO_signalProcessing.m');
        Pre_cascade_MIMO_signalProcessing;
        fprintf('Executed for folder %s\n', folderName_iter);
    catch ME
        fprintf('Problem in : %s\n Error: %s\n', folderName_iter, ME.message);
    end

end