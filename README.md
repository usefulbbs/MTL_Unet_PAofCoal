# Multitask_Unet_ProximateAnalysis
 Proximate analysis of coal indicates the moisture, ash, volatile content, and calorific value, which has been widely utilized as the basis for determining the rank of coal which is in connection with coal price and utilization. However, determining these characteristics are time consuming and require various laboratory equipment. An alternative way for proximate analysis is spectral analysis in combination with various machine learning methods. However, most previous works analyze individual characteristics and suffer from poor prediction performance. In this study, we propose a novel strategy for proximate analysis based on near-infrared spectroscopy and a multi-task attention Unet. The main contribution can be summarized as follows:

1. We design a multi-output regression model based on correlation analysis of four coal quality indicators, which not only improves the generalization ability but also alleviates overfitting.

2. We propose to combine U-shaped networks with convolutional block attention modules and multi-scale feature fusion technology. The shared feature extraction subnetwork can focus more on specific spectrum band and enrich the information of the output feature maps, facilitating the improvement of the regression performance.

3. Based on extensive experiments on the live industrial data, the proposed method achieves the best performance in terms of root mean square error (RMSE), mean absolute error (MAE), and correlation coefficient (R). 

The codes utilized in this project includes,

(1) thre_sigma_IC.m : An iterative outlier detection method based on the Pauta criterion and Euclidean distance (I-PC-ED)
(2) SPXY.py: Sample set Portioning based on a joint X-Y distance (SPXY) algorithm
(3) MTL_4task_train.py: The code for training the multitask Unet
