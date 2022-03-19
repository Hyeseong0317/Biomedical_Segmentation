nnU-Net

------------.bashrc 설정----------
export nnUNet_raw_data_base="/mnt/intern/nnUNet/nnUNet_raw_data_base"
export nnUNet_preprocessed="/mnt/intern/nnUNet/nnUNet_preprocessed"
export RESULTS_FOLDER="/mnt/intern/nnUNet/nnUNet_trained_models"

-------------nnUNet 연습--------------
nnUNet_convert_decathlon_task -i '/mnt/intern/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task04_Hippocampus'

nnUNet_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -t TASK_NAME_OR_ID -m CONFIGURATION --save_npz

nnUNet_predict -i '/mnt/intern/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task004_Hippocampus/imagesTs' -o '/mnt/intern/nnUNet/output_folder' -t 'Task004_Hippocampus'

----------nnUNet GPU 5번으로 설정 후 커스텀 데이터 연습---------------
CUDA_VISIBLE_DEVICES=5 nnUNet_train 2d nnUNetTrainerV2 0 
