clc;
clear all; 
close all; 
format short;


%% --- Configuration ---
% Update these to match your DE-PROX experiment folder

experiments_folder = "../savedModels/MOMENTUM_QSM_EXPERIMENTS//";
experiment_name="MomentumQSM_K_9"

experiments_folder = "../savedModels/ReducedRankExtrapolationQSM/";
experiment_name="RRE_QSM_K_4_order_4/"
experiment_name="RRE_QSM_K_9_order_3/"
experiment_name="RRE_QSM_K_12_order_3/"
experiment_name="RRE_QSM_K_6_order_3/"



experiments_folder = "../savedModels/MinimalPolynomialExtrapolationQSM/";
experiment_name="MPEQSM6_order_3/"




% experiments_folder = "../savedModels/MinimalPolynomialExtrapolationQSM//";
% experiment_name="MPEQSM6_order_3"
% 
% experiments_folder = "../savedModels/MOMENTUM_QSM_EXPERIMENTS//";
% experiment_name="MomentumQSM_K_9"
% 
% experiments_folder = "../savedModels/ReducedRankExtrapolationQSM//";
% experiment_name="RRE_QSM_K_6_order_3"
% 
% experiments_folder = "../savedModels/DEQ_QSM_EXPERIMENTS//";
% experiment_name="DEQ_QSM__Apr_10_20_18_01_K_140"
% 
% experiment_name="DEQ_QSM__Apr_11_08_15_34_200"

% Path where your Python test script saved the .mat files
% Matches the 'test_results' folder created in the Python test script
full_path_for_results = fullfile(experiments_folder, experiment_name);

%% --- Data Paths ---
data_source = 'given_data';
data_source_no = 1; 

if(strcmp(data_source, 'given_data'))
    raw_data_path = '../../../QSM_data/data_for_experiments/given_data/raw_data_names_modified/';
    if(data_source_no == 1)
        test_case = [1,2,3,4,5,6,7,8,9,10, 11, 12 ];
        %test_case = [7,8,9,10, 11, 12 ];

    end
end

%% --- Initialize Metrics ---
no_patients = size(test_case, 2);
ssim_scores = zeros(no_patients, 5);
rmse_scores = zeros(no_patients, 5);
psnr_scores = zeros(no_patients, 5);
hfen_scores = zeros(no_patients, 5);
xsim_scores = zeros(no_patients, 5);

savefigure = 0;

%% --- Evaluation Loop ---
for i = 1:size(test_case, 2)
    patient_id = test_case(i);
    file_path = fullfile(raw_data_path, ['patient_', num2str(patient_id)], '/');
    
    for j = 1:5
        % 1. Load Ground Truth and Mask
        load(fullfile(file_path, ['cos', num2str(j), '.mat'])); % Loads 'cos'
        load(fullfile(file_path, ['msk', num2str(j), '.mat'])); % Loads 'msk'

        % 2. Load DE-PROX Reconstruction
        % Note: Filename format matches 'de_prox_recon_{i}_{j}.mat' from Python script
        results_file_path = fullfile(full_path_for_results, sprintf('patient_%d_orientation_%d.mat', patient_id, j));
        
        if exist(results_file_path, 'file')
            data_struct = load(results_file_path);
            modl = squeeze(data_struct.chi_recon); % DE-PROX result
            
            % Apply mask to both for fair comparison [cite: 58]
            modl = modl .* single(msk);
            cos_gt = cos .* msk;
            
            if savefigure
                % You may need to update disp_fig to handle DE-PROX naming
                disp_fig(modl, cos_gt, patient_id, j, experiments_folder, experiment_name, epoch);
            end

            % 3. Compute Metrics
            ssim_scores(i,j) = round(compute_ssim(modl, cos_gt), 4);      
            rmse_scores(i,j) = round(compute_rmse(modl, cos_gt), 4);      
            psnr_scores(i,j) = round(compute_psnr(modl, cos_gt), 4);      
            hfen_scores(i,j) = round(compute_hfen(modl, cos_gt), 4);     
            xsim_scores(i,j) = round(compute_xsim(modl, cos_gt), 4);

            fprintf('P:%d O:%d | SSIM: %.4f | PSNR: %.4f | RMSE: %.4f |HFEN: %.4f \n\n', ...
                patient_id, j, ssim_scores(i,j), psnr_scores(i,j), rmse_scores(i,j), hfen_scores(i,j));
        else
            warning('File not found: %s', results_file_path);
        end
    end
end

%% --- Output CSV Management ---
output_dir = fullfile(experiments_folder, experiment_name, 'output_csv/');
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

output_file_name = sprintf('MOMENTUM_QSM_SSIM_%.4f__xSIM_%.4f_PSNR_%.4f_NRMSE_%.4f_HFEN_%.4f.csv', ...
     mean(ssim_scores(:)),mean(xsim_scores(:)),mean(psnr_scores(:)),mean(rmse_scores(:)), mean(hfen_scores(:)));
output_file_path = fullfile(output_dir, output_file_name);

% Construct output matrix (Matches your template logic)
output_final = [
    ssim_scores, mean(ssim_scores, 2); mean(ssim_scores, 1), mean(ssim_scores(:));
    zeros(2, 6);
    psnr_scores, mean(psnr_scores, 2); mean(psnr_scores, 1), mean(psnr_scores(:));
    zeros(2, 6);
    rmse_scores, mean(rmse_scores, 2); mean(rmse_scores, 1), mean(rmse_scores(:));
    zeros(2, 6);
    hfen_scores, mean(hfen_scores, 2); mean(hfen_scores, 1), mean(hfen_scores(:))
];


writematrix(round(output_final, 4), output_file_path);
disp(['Results saved to: ', output_file_path]);

%%
fprintf('  SSIM  = %.4f\n', mean(ssim_scores(:)));
fprintf('  PSNR  = %.4f dB\n', mean(psnr_scores(:)));
fprintf('  RMSE  = %.4f\n', mean(rmse_scores(:)));
fprintf('HFEN  = %.4f\n', mean(hfen_scores(:)));

%%
disp('Final results:')
fprintf('%.4f %.4f %.4f %.4f %.4f\n',mean(ssim_scores(:)),mean(xsim_scores(:)),mean(psnr_scores(:)),mean(rmse_scores(:)),mean(hfen_scores(:)))
fprintf('%.4f %.4f %.4f %.4f %.4f\n',std(ssim_scores(:)),std(xsim_scores(:)),std(psnr_scores(:)),std(rmse_scores(:)),std(hfen_scores(:)))
