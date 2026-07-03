%%-------------------------------------------------------------------------
clear all;
close all;
clc ;
%%
res = [0.64 0.64 0.64];
B0   = 7 ;
gyro = 42.7747892 ;                             % MHz / T
%N  = size(msk);
%Ne = size(magn,4);
scale_factor =  B0 * gyro;
%% load data Sim1Snr1

Sim1Snr1_path='../../../QSM_data//data_for_experiments/Simulation/Sim1Snr1/';

%magn=niftiread(strcat('../../QSM_data/data_for_experiments/Simulation/Sim1Snr1/','Magnitude.nii.gz'));
magn=niftiread(strcat(Sim1Snr1_path,'Magnitude.nii.gz'));
msk = niftiread(strcat(Sim1Snr1_path,'MaskBrainExtracted.nii.gz'));
freq = niftiread(strcat(Sim1Snr1_path,'Frequency.nii.gz'));
phs = niftiread(strcat(Sim1Snr1_path,'Phase.nii.gz'));
cos = niftiread(strcat(Sim1Snr1_path,'Chi_GT.nii.gz'));

load('../savedModels/LPCNN_QSM_MODELS/full_sampling_models/Jul_17_08_08_pm_model_K_3_model_WideResNet_data_source_1/predictions_55/Sim1Snr1_QSM.mat');
load('../savedModels/LPCNN_QSM_MODELS/full_sampling_models/Jul_17_08_11_pm_model_K_3_model_WideResNet_data_source_2/predictions_77/Sim1Snr1_QSM.mat');
load('../savedModels/LPCNN_QSM_MODELS/full_sampling_models/Jul_20_11_14_am_model_K_3_model_WideResNet_data_source_3/predictions_79/Sim1Snr1_QSM.mat');
load('../savedModels/LPCNN_QSM_MODELS/full_sampling_models/Jul_20_11_29_am_model_K_3_model_WideResNet_data_source_4/predictions_89/Sim1Snr1_QSM.mat');

experimental_phs=freq/((gyro * B0));

figure('Position', [1 1 1200 600],'Visible', 'on');        
subplot(2,2,1);
colormap('gray');
imagesc(experimental_phs(:,:,102,1));
xlabel('COSMOS');
colorbar;


subplot(2,2,2);
colormap('gray');
imagesc(msk(:,:,102));
xlabel('freq/((gyro * B0)');
colorbar;


subplot(2,2,3);
colormap('gray');
imagesc(cos(:,:,102));
xlabel('COSMOS');
colorbar;

subplot(2,2,4);
colormap('gray');
imagesc(Sim1Snr1_QSM(:,:,102));
xlabel('recon QSM');
colorbar;

ssim_scoresl = round(compute_ssim(Sim1Snr1_QSM.*msk,cos.*msk), 4);      
rmse_scoresl = round(compute_rmse(Sim1Snr1_QSM.*msk,cos.*msk), 4);      
psnr_scoresl = round(compute_psnr(Sim1Snr1_QSM.*msk,cos.*msk), 4);      
hfen_scoresl = round(compute_hfen(Sim1Snr1_QSM.*msk,cos.*msk), 4);
xsim_scoresl = round(compute_xsim(Sim1Snr1_QSM.*msk,cos.*msk), 4);      

fprintf("\n %f %f %f %f %f",ssim_scoresl,xsim_scoresl,psnr_scoresl,rmse_scoresl,hfen_scoresl)


%% load data Sim1Snr2

Sim1Snr1_path='../../../QSM_data/data_for_experiments/Simulation/Sim1Snr2/';

%magn=niftiread(strcat('../../QSM_data/data_for_experiments/Simulation/Sim1Snr1/','Magnitude.nii.gz'));
magn=niftiread(strcat(Sim1Snr1_path,'Magnitude.nii.gz'));
msk = niftiread(strcat(Sim1Snr1_path,'MaskBrainExtracted.nii.gz'));
freq = niftiread(strcat(Sim1Snr1_path,'Frequency.nii.gz'));
phs = niftiread(strcat(Sim1Snr1_path,'Phase.nii.gz'));
cos = niftiread(strcat(Sim1Snr1_path,'Chi_GT.nii.gz'));

load('../savedModels/LPCNN_QSM_MODELS/full_sampling_models/Jul_17_08_08_pm_model_K_3_model_WideResNet_data_source_1/predictions_55/Sim1Snr2_QSM.mat');
load('../savedModels/LPCNN_QSM_MODELS/full_sampling_models/Jul_17_08_11_pm_model_K_3_model_WideResNet_data_source_2/predictions_77/Sim1Snr2_QSM.mat');
load('../savedModels/LPCNN_QSM_MODELS/full_sampling_models/Jul_20_11_14_am_model_K_3_model_WideResNet_data_source_3/predictions_79/Sim1Snr2_QSM.mat');
load('../savedModels/LPCNN_QSM_MODELS/full_sampling_models/Jul_20_11_29_am_model_K_3_model_WideResNet_data_source_4/predictions_89/Sim1Snr2_QSM.mat');

experimental_phs=freq/((gyro * B0));

figure('Position', [1 1 1200 600],'Visible', 'on');        
subplot(2,2,1);
colormap('gray');
imagesc(experimental_phs(:,:,102,1));
xlabel('COSMOS');
colorbar;


subplot(2,2,2);
colormap('gray');
imagesc(msk(:,:,102));
xlabel('freq/((gyro * B0)');
colorbar;


subplot(2,2,3);
colormap('gray');
imagesc(cos(:,:,102));
xlabel('COSMOS');
colorbar;

subplot(2,2,4);
colormap('gray');
imagesc(Sim1Snr1_QSM(:,:,102));
xlabel('recon QSM');
colorbar;

ssim_scoresl = round(compute_ssim(Sim1Snr2_QSM.*msk,cos.*msk), 4);      
rmse_scoresl = round(compute_rmse(Sim1Snr2_QSM.*msk,cos.*msk), 4);      
psnr_scoresl = round(compute_psnr(Sim1Snr2_QSM.*msk,cos.*msk), 4);      
hfen_scoresl = round(compute_hfen(Sim1Snr2_QSM.*msk,cos.*msk), 4);
xsim_scoresl = round(compute_xsim(Sim1Snr2_QSM.*msk,cos.*msk), 4);      

fprintf("\n %f %f %f %f %f",ssim_scoresl,xsim_scoresl,psnr_scoresl,rmse_scoresl,hfen_scoresl)
%%
%% load data Sim2Snr1

Sim1Snr1_path='../../../QSM_data/data_for_experiments/Simulation/Sim2Snr1/';

magn=niftiread(strcat(Sim1Snr1_path,'Magnitude.nii.gz'));
msk = niftiread(strcat(Sim1Snr1_path,'MaskBrainExtracted.nii.gz'));
freq = niftiread(strcat(Sim1Snr1_path,'Frequency.nii.gz'));
phs = niftiread(strcat(Sim1Snr1_path,'Phase.nii.gz'));
cos = niftiread(strcat(Sim1Snr1_path,'Chi_GT.nii.gz'));

load('../savedModels/LPCNN_QSM_MODELS/full_sampling_models/Jul_17_08_08_pm_model_K_3_model_WideResNet_data_source_1/predictions_55/Sim2Snr1_QSM.mat');
load('../savedModels/LPCNN_QSM_MODELS/full_sampling_models/Jul_17_08_11_pm_model_K_3_model_WideResNet_data_source_2/predictions_77/Sim2Snr1_QSM.mat');
load('../savedModels/LPCNN_QSM_MODELS/full_sampling_models/Jul_20_11_14_am_model_K_3_model_WideResNet_data_source_3/predictions_79/Sim2Snr1_QSM.mat');
load('../savedModels/LPCNN_QSM_MODELS/full_sampling_models/Jul_20_11_29_am_model_K_3_model_WideResNet_data_source_4/predictions_89/Sim2Snr1_QSM.mat');

experimental_phs=freq/((gyro * B0));

figure('Position', [1 1 1200 600],'Visible', 'on');        
subplot(2,2,1);
colormap('gray');
imagesc(experimental_phs(:,:,102,1));
xlabel('COSMOS');
colorbar;


subplot(2,2,2);
colormap('gray');
imagesc(msk(:,:,102));
xlabel('freq/((gyro * B0)');
colorbar;


subplot(2,2,3);
colormap('gray');
imagesc(cos(:,:,102));
xlabel('COSMOS');
colorbar;

subplot(2,2,4);
colormap('gray');
imagesc(Sim1Snr1_QSM(:,:,102));
xlabel('recon QSM');
colorbar;

ssim_scoresl = round(compute_ssim(Sim2Snr1_QSM.*msk,cos.*msk), 4);      
rmse_scoresl = round(compute_rmse(Sim2Snr1_QSM.*msk,cos.*msk), 4);      
psnr_scoresl = round(compute_psnr(Sim2Snr1_QSM.*msk,cos.*msk), 4);      
hfen_scoresl = round(compute_hfen(Sim2Snr1_QSM.*msk,cos.*msk), 4);
xsim_scoresl = round(compute_xsim(Sim2Snr1_QSM.*msk,cos.*msk), 4);      

fprintf("\n %f %f %f %f %f",ssim_scoresl,xsim_scoresl,psnr_scoresl,rmse_scoresl,hfen_scoresl)
%%
%% load data Sim2Snr2

Sim1Snr1_path='../../../QSM_data/data_for_experiments/Simulation/Sim2Snr2/';

magn=niftiread(strcat(Sim1Snr1_path,'Magnitude.nii.gz'));
msk = niftiread(strcat(Sim1Snr1_path,'MaskBrainExtracted.nii.gz'));
freq = niftiread(strcat(Sim1Snr1_path,'Frequency.nii.gz'));
phs = niftiread(strcat(Sim1Snr1_path,'Phase.nii.gz'));
cos = niftiread(strcat(Sim1Snr1_path,'Chi_GT.nii.gz'));

load('../savedModels/LPCNN_QSM_MODELS/full_sampling_models/Jul_17_08_08_pm_model_K_3_model_WideResNet_data_source_1/predictions_55/Sim2Snr2_QSM.mat');
load('../savedModels/LPCNN_QSM_MODELS/full_sampling_models/Jul_17_08_11_pm_model_K_3_model_WideResNet_data_source_2/predictions_77/Sim2Snr2_QSM.mat');
load('../savedModels/LPCNN_QSM_MODELS/full_sampling_models/Jul_20_11_14_am_model_K_3_model_WideResNet_data_source_3/predictions_79/Sim2Snr2_QSM.mat');
load('../savedModels/LPCNN_QSM_MODELS/full_sampling_models/Jul_20_11_29_am_model_K_3_model_WideResNet_data_source_4/predictions_89/Sim2Snr2_QSM.mat');

experimental_phs=freq/((gyro * B0));

figure('Position', [1 1 1200 600],'Visible', 'on');        
subplot(2,2,1);
colormap('gray');
imagesc(experimental_phs(:,:,102,1));
xlabel('COSMOS');
colorbar;


subplot(2,2,2);
colormap('gray');
imagesc(msk(:,:,102));
xlabel('freq/((gyro * B0)');
colorbar;


subplot(2,2,3);
colormap('gray');
imagesc(cos(:,:,102));
xlabel('COSMOS');
colorbar;

subplot(2,2,4);
colormap('gray');
imagesc(Sim1Snr1_QSM(:,:,102));
xlabel('recon QSM');
colorbar;

ssim_scoresl = round(compute_ssim(Sim2Snr2_QSM.*msk,cos.*msk), 4);      
rmse_scoresl = round(compute_rmse(Sim2Snr2_QSM.*msk,cos.*msk), 4);      
psnr_scoresl = round(compute_psnr(Sim2Snr2_QSM.*msk,cos.*msk), 4);      
hfen_scoresl = round(compute_hfen(Sim2Snr2_QSM.*msk,cos.*msk), 4);
xsim_scoresl = round(compute_xsim(Sim2Snr2_QSM.*msk,cos.*msk), 4);      

fprintf("\n %f %f %f %f %f",ssim_scoresl,xsim_scoresl,psnr_scoresl,rmse_scoresl,hfen_scoresl)
