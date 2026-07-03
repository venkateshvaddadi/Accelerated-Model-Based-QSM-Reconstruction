%% This file will generate metrics for QSMnet on all test cases.

%%clc; 
%%
clear all; 
close all; 
addpath('./'); 
format short;
%%
num_cases = 6;  
orients=5;

ssim_scores = zeros(6,5);
rmse_scores = zeros(6,5);
psnr_scores = zeros(6,5);
hfen_scores = zeros(6,5);

savefigure = 1;
%%
experiements_folder="../savedModels/Spinet_complex_QSM_MODELS_dw_QSMnet_loss_l1_lambda_p_trainging/";
experiment="Jun_21_04_40_pm_model_K_1_B_2_lr_0.0001_N_16800/";
epoch=48

full_path_for_results=strcat(experiements_folder,experiment,'predictions_',num2str(epoch),'/modl-net');
disp(full_path_for_results)

%%
test_data_path='../../MoDL_QSM_PyTorch/data/Data/Testing_Data/'
%%
for i=1:6
    for j=1:5
        idx = strcat('-',num2str(i),num2str(j),'.mat');
        %%disp(strcat(full_path_for_results,idx))
        
        load(strcat(test_data_path,'cos/cos',idx));
        load(strcat(full_path_for_results,idx));
        load(strcat(test_data_path,'msk/msk' ,idx));
        modl = squeeze(modl);
        modl=modl .* single(msk);
        
        alpha = 0.9;
        
        %%modl = (modl + (modl - imgaussfilt3(modl, 0.5, 'FilterSize', 5)) * alpha) .* single(msk); 
        
         if savefigure
            disp_fig(modl, cos,i,j,experiements_folder,experiment,epoch);
         end
        
        ssim_scores(i,j) = round(compute_ssim(modl,cos), 4);      
        rmse_scores(i,j) = round(compute_rmse(modl,cos), 4);      
        psnr_scores(i,j) = round(compute_psnr(modl,cos), 4);      
        hfen_scores(i,j) = round(compute_hfen(modl,cos), 4);     
        %%disp([round(i),round(j),ssim_scores(i,j),rmse_scores(i,j),psnr_scores(i,j),hfen_scores(i,j)])
        fprintf('%d %d %.4f %.4f %.4f %.4f \n',i,j,ssim_scores(i,j),rmse_scores(i,j),psnr_scores(i,j),hfen_scores(i,j) )
        
    end
     clear cos susc
end

results = [     ssim_scores          mean(ssim_scores,2)            rmse_scores        mean(rmse_scores,2)  ;
            mean(ssim_scores,1)      mean(ssim_scores(:))        mean(rmse_scores,1)   mean(rmse_scores(:)) ;
               psnr_scores           mean(psnr_scores,2)            hfen_scores        mean(hfen_scores,2)  ;
            mean(psnr_scores,1)      mean(psnr_scores(:))        mean(hfen_scores,1)   mean(hfen_scores(:))];
%%
%%results
%%
disp('Final results:')
%disp([mean(ssim_scores(:)),mean(psnr_scores(:)),mean(rmse_scores(:)),mean(hfen_scores(:))])

fprintf('%.4f %.4f %.4f %.4f\n',mean(ssim_scores(:)),mean(psnr_scores(:)),mean(rmse_scores(:)),mean(hfen_scores(:)))
%%
output_file_path=strcat(experiements_folder,experiment,'output_csv/',num2str(epoch),'.csv');
writematrix(results, output_file_path) 
%%




disp(ssim_scores)






