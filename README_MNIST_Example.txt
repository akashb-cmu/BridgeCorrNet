Instructions to run MNIST experiment:

Download mnist data from http://yann.lecun.com/exdb/mnist/

1) python train_bridge_corrnet.py <PATH TO DOWNLOADED MNIST DATA> <PATH TO STORE GENERATED MNIST LEFT/RIGHT/PIVOT VIEWS> <PATH TO STORE TRAINED MODEL>

eg. python train_bridge_corrnet.py ../mnist_images/ /home/akashb/Desktop/Acads/Sem2/Projects/WMT/Corr_net_author/CorrNet/mnistExample/generated_views/ ../Model/saved_model/

2) python project_bridge_corrnet.py <PATH TO STORE GENERATED MNIST LEFT/RIGHT/PIVOT VIEWS> <PATH TO STORE TRAINED MODEL> <PATH TO STORE PROJECTED VIEWS>

eg. python project_bridge_corrnet.py /home/akashb/Desktop/Acads/Sem2/Projects/WMT/Corr_net_author/CorrNet/mnistExample/generated_views/ /home/akashb/Desktop/Acads/Sem2/Projects/WMT/Corr_net_author/CorrNet/Model/saved_model/ ./TGT_DIR/projected_views/

3) python evaluate_bridge_corrnet.py <PATH TO STORED PROJECTED VIEWS>

eg. python evaluate_bridge_corrnet.py ./TGT_DIR/projected_views/



Using the mnist data found at http://yann.lecun.com/exdb/mnist/ , you should obtain the following result:

Left_only and Right_only
28.7177037552
view1 to view2
0.6891
view2 to view1
0.7304
Left_only and pivot_only
37.3754298182
view1 to view2
0.7975
view2 to view1
0.7865
Right_only and pivot_only
41.8844806119
view1 to view2
0.8327
view2 to view1
0.7642
Left_pivot and right_pivot
43.600299611
view1 to view2
0.8589
view2 to view1
0.8601
Left_pivot and right_only
36.6328180087
view1 to view2
0.7581
view2 to view1
0.7943
Right_pivot and left_only
32.108290323
view1 to view2
0.7501
view2 to view1
0.7471
