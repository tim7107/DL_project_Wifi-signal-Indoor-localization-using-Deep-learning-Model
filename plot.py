import numpy as np
import matplotlib.pyplot as plt
#wopretrain_test=np.load('ResNet18_pretrain_testingdata.npy')
#wopretrain_train=np.load('ResNet18_pretrain_trainingdata.npy')

#pretrain_train=np.load('ResNet50_pretrain_trainingdata.npy')
#pretrain_test=np.load('ResNet50_pretrain_testingdata.npy')

wopretrain_test=np.load('UJIindoorLoc_testingdata_loss.npy',allow_pickle=True)
pretrain_test=np.load('UJIindoorLoc_testingdata_acc.npy',allow_pickle=True)

plt.title(' UJIindoorLoc Dataset ' ,fontsize=14)
plt.xlabel('epoch')
plt.ylabel('loss')

plt.plot(wopretrain_test,label='Testing loss')
plt.legend(loc='best')


"""
plt.plot(pretrain_test,label='Testing accuracy')
plt.legend(loc='best')
"""
plt.show()
