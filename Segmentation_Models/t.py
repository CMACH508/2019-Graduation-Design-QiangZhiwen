from atmp import *
import numpy, elasticdeform,cv2

x_train,y_train=generate_data_format("data/train/image/","data/train/label/")
print(x_train.shape)
print(y_train.shape)


path1='data/train/label/'
path2='data/train/elas/label/'

alpha=95 
sigma=8


# for index in range(90,100):
#     for j in range (5,10):

#         path1='data/train/label/'
#         path2='data/train/elas/label/'

#         alpha=index
#         sigma=j

#         for i in range(30):
#             X=cv2.imread(path1+str(i)+".png")
#             X_deformed = elastic_transform(X, alpha,sigma,5)
#             cv2.imwrite( path2+ str(i)+'_'+str(alpha)+'_'+str(sigma)+'_5.png', X_deformed)

#         print("half done")

#         path1='data/train/image/'
#         path2='data/train/elas/image/'
#         for i in range(30):
#             X=cv2.imread(path1+str(i)+".png")
#             X_deformed = elastic_transform(X,alpha,sigma,5)
#             cv2.imwrite( path2+ str(i)+'_'+str(alpha)+'_'+str(sigma)+'_5.png', X_deformed)
#         print("done")


