# Neural-Garment-Dynamic-Super-resolution

## Introduction
![](teaser.png) <br />
This repository contains the implemetation of our Siggraph Asia 2024 paper, [Neural Garment Dynamic Super-resolution](https://github.com/MengZephyr/Neural-Garment-Dynamic-Super-resolution/blob/main/papers/GDSR_SIGA_2024.pdf). Given low-resolution (LR) garment simulation as input(a,c,e,g), we present a learning-based method of garment dynamic super resolution (GDSR) to construct high-resolution garment dynamic geometry(b, d, d, h). We learn super resolution (SR) features from the coarse garment dynamics and the garment-body interactions, to produce high-resolution fine-grained and plausible wrinkle details to enhance the low-resolution geometry of various garment types, and enable the generalization ability of our method across unseen body motions(b, d, d, h), unseen body shape(d, f), and unseen garment types(f, h).

We provide Google drive links for downloading the training data (five outfits and some motion sequences), the network checkpoint:
>[Five outfits](https://drive.google.com/drive/folders/1vNkcLLMDHyUzN40RE6x8LbNXIjQ3LspF?usp=sharing) <br />
>[body motions](https://drive.google.com/drive/folders/1tXZCJiVOuLLa2fuOlwrCKhA5_v3vY0eP?usp=sharing) <br />
>[Checkpoint](https://drive.google.com/file/d/1lrYa4SK0uH1IdjrvzjBHyD-a-tc60BCe/view?usp=sharing) <br />

## Have fun!
If you want to take a shot of our method, you can follow below step by step:

> Throw an outfit and a motion sequence into Marvelous Designer.
 
> Choose the material settings. We use Silk_Charmuse in our paper.

> Run simulations with the partical settings of PD=10mm (High Resolution) and PD=30mm (Low Resolution), respecitvely, and export the OBJ sequences named with "PD10" and "PD30" ("weld" and "thin" chosen) in the file folder: './Data/[Garment]/[motion]/PD10/' and './Data/[Garment]/[motion]/PD30/'.

> Export the static garments at the canonical pose for the high-resolution and low-resolution respectively, named as "PD10_C.obj" and "PD30_C.obj" ("weld" and "thin" is chosen) in the folder: './Data/[Garment]/Canonical/weld/'.

> Export the flattened garments, named as "PD10_Flatten.obj" and "PD30_Flatten.obj" ( "unweld" and "thin" chosen) into the folder: './Data/[Garment]/Canonical/weld/'.

> Use GDSR/uv_abstract.py to depart the geometry and the uv information, and to generate "PD10_geo.ply", "PD10_uv.ply", "PD30_geo.ply", and "PD30_uv.ply" in the folder: './Data/[Garment]/Canonical/weld/'.
 
> Compile the C++ project, UV_Sampling_proj, which is dependent on the 3rd library of opencv_4.6.0 and embree-3.5.2.x64.vc14.windows.

> Only Uncommend "Sampling_between_Different_PDResolution_Across_UV()" in the main function, and run the project to generate "10_from_30_Sampling.txt" and "30_from_10_Sampling.txt" into the folder: './Data/[Garment]/Canonical/weld/test/'.

> Only Uncommend "Geo_UV_Map()" in the main function, and run the project to generate "PD10_g_to_u.txt", "PD10_u_to_g.txt", "PD30_g_to_u.txt" and "PD30_u_to_g.txt" in the folder: './Data/[Garment]/Canonical/weld/'.

> Only Uncommend "GeoImage_Rasterization()" in the main function, and run the project to generate "PD10_1024_pixelGeoSample.txt" into the folder: './Data/[Garment]/Canonical/weld/test/'.

> Import the body motion ('.fbx') into blend and export the obj sequence named as 'b' into the folder: './Data/body/[motion]/'.

>  
 
