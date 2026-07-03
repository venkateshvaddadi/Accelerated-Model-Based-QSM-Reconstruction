
clc;
clear all; 
close all; 
format short;

%%


epoch=24
for epoch=0:100

    experiements_folder="../savedModels/LPCNN_QSM_MODELS//full_sampling_models/With_k_1_init_for_rebuttal//"
    experiment_name="Jan_11_08_16_pm_model_K_3_model_WideResNet__patient_1/"
    full_path_for_results=strcat(experiements_folder,experiment_name,'predictions_',num2str(epoch),'/modl-net');
    

    %%
    data_source='given_single_patient_data';
    Training_patient_no=1
    data_source_no=1
    
    if(strcmp(data_source,'generated_data'))
        raw_data_path='../../../QSM_data/data_for_experiments/generated_data/raw_data/';
        data_path='../../../QSM_data/data_for_experiments/generated_data/data_source_1/';
        test_case =[7,32,9,10];
    end
    
    
    if(strcmp(data_source,'generated_noisy_data'))
        raw_data_path='../../../QSM_data/data_for_experiments/generated_data/raw_data/'
        test_case =[7,32,9,10]
    end
    
    if(strcmp(data_source,'generated_undersampled_data'))
        raw_data_path='../../../QSM_data/data_for_experiments/generated_data/raw_data/';
        test_case =[7,32,9,10]
    end
    
    
    if(strcmp(data_source,'given_data'))
        raw_data_path='../../../QSM_data/data_for_experiments/given_data/raw_data_names_modified/'
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
    if(strcmp(data_source,'given_single_patient_data'))
        raw_data_path='../../../QSM_data/data_for_experiments/given_data/raw_data_names_modified/'
        patients_list =[7,8,9,10,11,12]
        if(Training_patient_no==1)
            data_path='../../../QSM_data/data_for_experiments/given_data/single_patient/patient_1/'
            test_case =[2,3,4,5,7,8,9,10,11,12]
        end
        if(Training_patient_no==2)
            data_path='../../../QSM_data/data_for_experiments/given_data/single_patient/patient_2/'
            test_case =[1,3,4,5,7,8,9,10,11,12]
        end
        if(Training_patient_no==3)
            data_path='../../../QSM_data/data_for_experiments/given_data/single_patient/patient_3/'
            test_case =[1,2,4,5,7,8,9,10,11,12]
        end
        if(Training_patient_no==4)
            data_path='../../../QSM_data/data_for_experiments/given_data/single_patient/patient_4/'
            test_case =[1,2,3,5,7,8,9,10,11,12]
        end
    end
    ls(raw_data_path)
    
    %%
    no_patients=size(test_case,2)
    ssim_scores = zeros(no_patients,5);
    rmse_scores = zeros(no_patients,5);
    psnr_scores = zeros(no_patients,5);
    hfen_scores = zeros(no_patients,5);
    
    %%
    
    savefigure = 1;
    
    %%
    for i=1:size(test_case,2)
        patient_id=test_case(i);
        file_path=strcat(raw_data_path,'patient_',num2str(patient_id),'/');
        for j=1:5
            %fprintf("(%d %d)\n",i,j)
            
            % loading cosmos,msk results..
            load(strcat(file_path,'cos',num2str(j),'.mat'));
            load(strcat(file_path,'msk',num2str(j),'.mat'));
    
            % loading generated results..
            results_file_path=strcat(full_path_for_results,'-',num2str(patient_id),'-',num2str(j),'.mat');
            load(results_file_path);
            modl = squeeze(modl);
            modl=modl .* single(msk);
            cos=cos.*msk;
            
             if savefigure
                disp_fig(modl, cos,patient_id,j,experiements_folder,experiment_name,epoch);
             end
    
            ssim_scores(i,j) = round(compute_ssim(modl,cos), 4);      
            rmse_scores(i,j) = round(compute_rmse(modl,cos), 4);      
            psnr_scores(i,j) = round(compute_psnr(modl,cos), 4);      
            hfen_scores(i,j) = round(compute_hfen(modl,cos), 4);     
            fprintf('%d %d %.4f %.4f %.4f %.4f \n',patient_id,j,ssim_scores(i,j),psnr_scores(i,j),rmse_scores(i,j),hfen_scores(i,j) )
    
    
        end
    end
    
    %%
    
    results = [     ssim_scores          mean(ssim_scores,2)            rmse_scores        mean(rmse_scores,2)  ;
                mean(ssim_scores,1)      mean(ssim_scores(:))        mean(rmse_scores,1)   mean(rmse_scores(:)) ;
                   psnr_scores           mean(psnr_scores,2)            hfen_scores        mean(hfen_scores,2)  ;
                mean(psnr_scores,1)      mean(psnr_scores(:))        mean(hfen_scores,1)   mean(hfen_scores(:))];
    
    
     %writing the results as a csv file.
    
    output_file_name=strcat(num2str(epoch),'_',num2str(mean(ssim_scores(:))),'_',num2str(mean(psnr_scores(:))),'_',num2str(mean(rmse_scores(:))),'_',num2str(mean(hfen_scores(:))),'.csv');
    mkdir(strcat(experiements_folder,experiment_name,'output_csv/'))
    output_file_path=strcat(experiements_folder,experiment_name,'output_csv/',output_file_name);
    disp(output_file_path)
    
    
    output=([
        ssim_scores,mean(ssim_scores,2);mean(ssim_scores,1),mean(ssim_scores(:));
        0,0,0,0,0,0;
        0,0,0,0,0,0;
        psnr_scores,mean(psnr_scores,2);mean(psnr_scores,1),mean(psnr_scores(:));
        0,0,0,0,0,0;
        0,0,0,0,0,0;
        rmse_scores,mean(rmse_scores,2);mean(rmse_scores,1),mean(rmse_scores(:));
        0,0,0,0,0,0;
        0,0,0,0,0,0;
        hfen_scores,mean(hfen_scores,2);mean(hfen_scores,1),mean(hfen_scores(:));
        0,0,0,0,0,0;
        0,0,0,0,0,0;
        0,0,mean(ssim_scores(:)),mean(psnr_scores(:)),mean(rmse_scores(:)),mean(hfen_scores(:));
        0,0,std(ssim_scores(:)),std(psnr_scores(:)),std(rmse_scores(:)),std(hfen_scores(:));
    
        ]);
    
    writematrix(round(output,4), output_file_path) 
    
    %%
    % display final results
    
    disp('Final results:')
    fprintf('%.4f %.4f %.4f %.4f\n',mean(ssim_scores(:)),mean(psnr_scores(:)),mean(rmse_scores(:)),mean(hfen_scores(:)))
    
    
    fprintf('%.4f %.4f %.4f %.4f\n',std(ssim_scores(:)),std(psnr_scores(:)),std(rmse_scores(:)),std(hfen_scores(:)))
    
    
end
