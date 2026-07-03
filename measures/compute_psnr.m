function [ PSNR ] = compute_psnr( chi_recon, chi_true )


img1 = chi_recon;

img2 = chi_true;

min_img = min(min(img1(:)), min(img2(:)));

img1(img1~=0) = img1(img1~=0) - min_img;
img2(img2~=0) = img2(img2~=0) - min_img;

max_img = max(max(img1(:)), max(img2(:)));

img1 = 255 * img1 ./ max_img;
img2 = 255 * img2 ./ max_img;


PSNR = psnr(img1, img2, 255);

end

