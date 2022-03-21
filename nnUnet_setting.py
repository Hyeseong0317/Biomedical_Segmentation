nnU-Net

------------.barsch 설정----------
export nnUNet_raw_data_base="/mnt/intern/nnUNet/nnUNet_raw_data_base"
export nnUNet_preprocessed="/mnt/intern/nnUNet/nnUNet_preprocessed"
export RESULTS_FOLDER="/mnt/intern/nnUNet/nnUNet_trained_models"

-------------nnUNet 연습--------------
nnUNet_convert_decathlon_task -i '/mnt/intern/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task04_Hippocampus'

nnUNet_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -t TASK_NAME_OR_ID -m CONFIGURATION --save_npz

nnUNet_predict -i '/mnt/intern/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task004_Hippocampus/imagesTs' -o '/mnt/intern/nnUNet/output_folder' -t 'Task004_Hippocampus'

----------nnUNet GPU 설정 후 커스텀 데이터 연습---------------
CUDA_VISIBLE_DEVICES=5 nnUNet_train 2d nnUNetTrainerV2 0 

AORTA_001_0000.nii.gz

nvidia-smi -->Terminal에 입력하여 GPU 사용량 확인

-----------dataset sanity check--------------
nnUNet_plan_and_preprocess -t 500 --verify_dataset_integrity

----------nnUNet GPU 설정 후 모델 Training------------
CUDA_VISIBLE_DEVICES=5 nnUNet_train 3d_fullres nnUNetTrainerV2 Task500_Aorta 0 --npz
CUDA_VISIBLE_DEVICES=5 nnUNet_train 2d nnUNetTrainerV2 Task500_Aorta 0 --npz

nnUNet_predict -i  '/mnt/intern/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task500_Aorta/imagesTs' -o  '/mnt/intern/nnUNet/Aorta_output' -t 'Task500_Aorta' -m 3d_fullres

nnUNet_predict -i  '/mnt/intern/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task500_Aorta/imagesTs' -o  '/mnt/intern/nnUNet/Aorta_output' -t 'Task500_Aorta' -m 2d

------------DSC 평균-------------


