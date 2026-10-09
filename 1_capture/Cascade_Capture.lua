--[[
    A. FRAMING & CAPTURE
    1. Triggering Slave (3, 2, 1) sequentially in a hardware triggered mode.
    2. Triggering Master in a software triggered mode.

    B. TRANSFERRING FILES
    1. The data is stored in file(s) with max cap placed at 2 GB.
    2. The files can be retrieved from the SSD (/mnt/ssd folder) using WinSCP.

Note: Update lines 18 to 49 as needed before using this script.
--

Note: "capture_time"  is a timeout for this script alone to exit - it does not control the 
actual duration of capture. The actual capture duration depends on the configured frame time 
and number of frames.
--]]
RSTD.Sleep(5000)

capture_time                     =   25000                             -- ms
inter_loop_time                  =   100                             -- ms
num_loops                        =   1

--[[
Note: Change the following three parameters as desired:
1. n_files_allocation: is the number of files to preallocate on the SSD.
   This improves capture reliability by not having frame drops while switching files.
   The tradeoff is that each file is a fixed 2047 MB even if a smaller number of frames are captured.
   Pre-allocate as many files as needed based on (size_per_frame * number_of_frames) to be captured.
2. data_packaging: select whether to use 16-bit ADC data as is, or drop 4 lsbits and save 4*12-bit numbers in a packed form
   This allows a higher frame rate to be achieved, at the expense of some post-processing to unpack the data later.
   (Matlab should still be able to unpack the data using the '*ubit12' argument to fread instead of 'uint16')
   The default is no-packing, for simplicity
3. capture_directory: is the filename under which captures are stored on the SSD
   and is also the directory to which files will be transferred back to the host
   The captures are copied to the PostProc folder within mmWave Studio

   Note: If this script is called multiple times without changing the directory name, then all 
         captured files will be in the same directory with filename suffixes incremented automatically. 
         It may be hard to know which captured files correspond to which run of the script.
   Note: It is strongly recommended to change this directory name between captures.
--]]

n_files_allocation              =   0
data_packaging                  =   0                       -- 0: 16-bit, 1: 12-bit
capture_directory               =   "scene12_set3_r3"      --  "Cascade_Capture_22xx"
num_frames_to_capture           =   0                       -- 0: default case; Any positive value - number of frames to capture 

framing_type                    =   1                       -- 0: infinite, 1: finite
stop_frame_mode                 =   0                       -- 0: Frame boundary, 2: Sub-frame boundary, 
                                                            -- 3: Burst boundary, 4: HW/Sub-frame triggered

----------------------------------DATA CAPTURE-------------------------------------------
-- Function to start/stop frame



function Framing_Control(Device_ID, En1_Dis0)
    local status = 0         
        if (En1_Dis0 == 1) then 
            status = ar1.StartFrame_mult(dev_list[Device_ID]) --Start Trigger Frame
            if (status == 0) then
                WriteToLog("Device "..Device_ID.." : Start Frame Successful\n", "green")
            else
                WriteToLog("Device "..Device_ID.." : Start Frame Failed\n", "red")
                return -5
            end
        else
            status = ar1.StopFrame_mult(dev_list[Device_ID], stop_frame_mode) --Stop Trigger Frame
            if (status == 0) then
                WriteToLog("Device "..Device_ID.." : Stop Frame Successful\n", "green")
            else
                WriteToLog("Device "..Device_ID.." : Stop Frame Failed\n", "red")
                return -5
            end
        end
    
    return status
end


while (num_loops > 0)
do

WriteToLog("Loops Remaining : "..num_loops.."\n", "purple")

-- TDA ARM
WriteToLog("Starting TDA ARM...\n", "blue")
status = ar1.TDACaptureCard_StartRecord_mult(1, n_files_allocation, data_packaging, capture_directory, num_frames_to_capture)
if (status == 0) then
    WriteToLog("TDA ARM Successful\n", "green")
else
    WriteToLog("TDA ARM Failed\n", "red")
    return -5
end    

RSTD.Sleep(1000)

-- Triggering the data capture
WriteToLog("Starting Frame Trigger sequence...\n", "blue")

if (RadarDevice[4]==1)then
    Framing_Control(4,1)
end

if (RadarDevice[3]==1)then
    Framing_Control(3,1)
end

if (RadarDevice[2]==1)then
    Framing_Control(2,1)
end

Framing_Control(1,1)

WriteToLog("Capturing AWR device data to the TDA SSD...\n", "blue")
RSTD.Sleep(capture_time)

if (framing_type == 0) then
    
    -- Stop capturing
    WriteToLog("Starting Frame Stop sequence...\n", "blue")
    if (RadarDevice[4]==1)then
        Framing_Control(4,0)
    end

    if (RadarDevice[3]==1)then
        Framing_Control(3,0)
    end

    if (RadarDevice[2]==1)then
        Framing_Control(2,0)
    end
    
    Framing_Control(1,0)
end

WriteToLog("Capture sequence completed...\n", "blue")
    
num_loops = num_loops - 1
RSTD.Sleep(inter_loop_time)

end



--[[
-- Enable the below if required
WriteToLog("Starting Transfer files using WinSCP..\n", "blue")
status = ar1.TransferFilesUsingWinSCP_mult(1)
if(status == 0) then
    WriteToLog("Transferred files! COMPLETE!\n", "green")
else
    WriteToLog("Transferring files FAILED!\n", "red")
    return -5
end
--]]


--[[
local folders = {"test1", "test2"} -- List of folders to process

for _, folder in ipairs(folders) do
    capture_directory = folder -- Set the current folder
    WriteToLog("Starting Transfer files from folder: " .. folder .. " using WinSCP..\n", "blue")
    
    local status = ar1.TransferFilesUsingWinSCP_mult(1)
    if status == 0 then
        WriteToLog("Transferred files from " .. folder .. "! COMPLETE!\n", "green")
    else
        WriteToLog("Transferring files from " .. folder .. " FAILED!\n", "red")
        return -5
    end
end
--]]