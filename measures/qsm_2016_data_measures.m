

clear all; 
close all; 
clc;
%%
epoch=22



experiments_folder="../savedModels/LPCNN_QSM_MODELS/experiments_on_given_data//"
experiment_name="Oct_28_07_15_pm_model_K_3__data_source_4_model_WideResNet_MSE_given_data///"
full_path_for_results=strcat(experiments_folder,"/",experiment_name,'/','predictions_',num2str(epoch),'/','modl-net');

savefigure = 1;

ls(strcat(experiments_folder,"/",experiment_name,'/','predictions_',num2str(epoch),'/'))
%%


data_source='qsm_2016_challange';
data_source_no=1

if(strcmp(data_source,'generated_data'))
    raw_data_path='../../QSM_data/data_for_experiments/generated_data/raw_data/';
    test_case =[7,32,9,10];
end

if(strcmp(data_source,'given_data'))
    raw_data_path='../../QSM_data/data_for_experiments/raw_data_names_modified/';
    if(data_source_no==1)
        test_case=[7,8,9,10,11,12];
    end
    if(data_source_no==2)
        test_case=[10,11,12,1,2,3]
    end
    if(data_source_no==3)
        test_case=[1,2,3,4,5,6]
    end
    if(data_source_no==4)
        test_case=[4,5,6,7,8,9]
    end

    
end   

if(strcmp(data_source,'lpcnn_data'))
    raw_data_path='../../../QSM_data/data_for_experiments/LPCNN_data/'
    test_case=[1,2,3]

end

if(strcmp(data_source,'qsm_2016_challange'))
    raw_data_path='../../../QSM_data/data_for_experiments/qsm_2016_recon_challenge/updated_data/'
    test_case=[1]
end


%%

no_patients=size(test_case,2)
ssim_scores = zeros(no_patients,4);
rmse_scores = zeros(no_patients,4);
psnr_scores = zeros(no_patients,4);
hfen_scores = zeros(no_patients,4);
xsim_scores = zeros(no_patients,1);


%%

for i=1:size(test_case,2)
    patient_id=test_case(i);
    file_path=strcat(raw_data_path,'patient_',num2str(patient_id),'/')
    ls(file_path)
    for j=1:4
        %fprintf("(%d %d)\n",i,j)
        
        % loading cosmos,msk results..
        load(strcat(file_path,'cos',num2str(j),'.mat'));
        load(strcat(file_path,'msk',num2str(j),'.mat'));

        % loading generated results..
        results_file_path=strcat(full_path_for_results,'-',num2str(patient_id),'-',num2str(j),'.mat');
        load(results_file_path);
        modl = squeeze(modl);

        modl=modl .* single(msk);
        cos=cos.*single(msk);
        
         if savefigure
            disp_fig(modl, cos,patient_id,j,experiments_folder,experiment_name,epoch);
         end

        ssim_scores(i,j) = round(compute_ssim(modl,cos), 4);      
        rmse_scores(i,j) = round(compute_rmse(modl,cos), 4);      
        psnr_scores(i,j) = round(compute_psnr(modl,cos), 4);      
        hfen_scores(i,j) = round(compute_hfen(modl,cos), 4);     
        xsim_scores(i,j)=round(compute_xsim(modl,cos),4);

        fprintf('%d %d %.4f %.4f %.4f %.4f %.4f \n',patient_id,j,ssim_scores(i,j),xsim_scores(i,j),psnr_scores(i,j),rmse_scores(i,j),hfen_scores(i,j) )


    end
end

%%

 %writing the results as a csv file.

output_file_name=strcat(num2str(epoch),'_',num2str(mean(ssim_scores(:))),'_',num2str(mean(psnr_scores(:))),'_',num2str(mean(rmse_scores(:))),'_',num2str(mean(hfen_scores(:))),'.csv');
output_file_path=strcat(experiments_folder,experiment_name,'output_csv/',output_file_name);
%%disp(output_file_path)


output=([
    ssim_scores,mean(ssim_scores,2); ...
    mean(ssim_scores,1),mean(ssim_scores(:));
    0,0,0,0,0;
    0,0,0,0,0;
    psnr_scores,mean(psnr_scores,2);mean(psnr_scores,1),mean(psnr_scores(:));
    0,0,0,0,0;
    0,0,0,0,0;
    rmse_scores,mean(rmse_scores,2);mean(rmse_scores,1),mean(rmse_scores(:));
    0,0,0,0,0;
    0,0,0,0,0;
    hfen_scores,mean(hfen_scores,2);mean(hfen_scores,1),mean(hfen_scores(:));
    0,0,0,0,0;
    0,0,0,0,0;
    0,mean(ssim_scores(:)),mean(psnr_scores(:)),mean(rmse_scores(:)),mean(hfen_scores(:))
    ]);

writematrix(round(output,4), output_file_path) 
disp('csv generated...')

%%
% display final results

disp('Final results:')
fprintf('%.4f %.4f %.4f %.4f\n',mean(ssim_scores(:)),mean(psnr_scores(:)),mean(rmse_scores(:)),mean(hfen_scores(:)))






