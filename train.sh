/home/grads/mantanguo2/anaconda3/bin/python /public/mantanguo/github/train.py  --batch_size  1 --num_source 3  --validation_data_path /public/mantanguo/Dataset/ALFR/train_DTU_RGB_79x49_flow_79x49x3x2_6dof_79x49x6_sc_79x49x3.h5  --training_data_path /public/mantanguo/Dataset/ALFR/train_DTU_RGB_79x49_flow_79x49x3x2_6dof_79x49x6_sc_79x49x3.h5 
# /home/grads/mantanguo2/anaconda3/bin/python /public/mantanguo/github/test.py  --num_source 3 --model_name dtu_s3.pth --test_data_path /public/mantanguo/Dataset/ALFR/test_DTU_RGB_18x49_flow_18x49x3x2_6dof_18x49x6_sc_18x49x3.h5