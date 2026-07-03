%% naveenp@iisc.ac.in MIG, CDS, IISc Bangalore

%% This code will just write the output images to './'

function [] = disp_fig(net,cos,i,j,experiements_folder,experiment,epoch)

        a=round(size(cos,3)/2);
        b=round(size(cos,2)/2);
        c=round(size(cos,1)/2);  
        
        im1 = imrotate((squeeze(cos(:,:,a))),-90);
        im2 = imrotate((squeeze(cos(:,b,:))), 90);
        im3 = imrotate((squeeze(cos(c,:,:))), 90);        
        im4 = imrotate((squeeze(net(:,:,a))),-90);
        im5 = imrotate((squeeze(net(:,b,:))), 90);
        im6 = imrotate((squeeze(net(c,:,:))), 90);
 
        figure('Position', [1 1 1200 600],'Visible', 'off');        
        subplot(2,3,1);
        colormap('gray');
        imagesc(im1,[-0.1, 0.1]);
        xlabel('COSMOS');
        colorbar;
        subplot(2,3,2);
        colormap('gray');
        imagesc(im2,[-0.1, 0.1]);
        xlabel('COSMOS');
        colorbar;
        subplot(2,3,3);
        colormap('gray');
        imagesc(im3,[-0.1, 0.1]);
        xlabel('COSMOS');
        colorbar;
        subplot(2,3,4);
        colormap('gray');
        imagesc(im4,[-0.1, 0.1]);
        psnr_im4 = compute_psnr(im4,im1);
        xlabel(strcat('LPCNN(',num2str(psnr_im4),')'));
        colorbar;
        subplot(2,3,5);
        colormap('gray');
        imagesc(im5,[-0.1, 0.1]);
        psnr_im5 = compute_psnr(im5,im2);
        xlabel(strcat('LPCNN(',num2str(psnr_im5),')'));
        colorbar;
        subplot(2,3,6);
        colormap('gray');
        imagesc(im6,[-0.1, 0.1]);
        psnr_im6 = compute_psnr(im6,im3);
        xlabel(strcat('LPCNN(',num2str(psnr_im6),')'));
        colorbar;   
        filename = strcat('chi-MoDL-test',num2str(i),'-',num2str(j),'-','image.png');
        full_path_for_results=strcat(experiements_folder,experiment,"predictions_",num2str(epoch),"/",filename);
        imwrite(getframe(gcf).cdata,full_path_for_results);
        close all;
        
               
end