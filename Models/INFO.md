# INFO

- model.pth.tar : taken from repo https://github.com/arnoweng/CheXNet
- model 51000 : First trial with errors in the loss function
- model 39000 : Second trial with a lot of overfitting
- model 86400 : No dropout, vertical & horizontal flip for data augmentation
- model 235000 : 0.2 dropout, horizontal flip only
- model 178800 : 0.2 dropout, horizontal flip only, learning rate decay (x0.1 every 10 epochs)
- model 113800 : No dropout, horizontal flip only, learning rate decay (x0.1 every 10 epochs)
- model 31800 : No dropout, horizontal flip only, learning rate decay (x0.1 every 10 epochs) batch size 32
- model 90600 : 0.01 dropout, horizontal flip only, learning rate decay (x0.5 every 10 epochs)
- model 37200 : No dropout, horizontal flip only, learning rate decay (x0.1 every 10 epochs) batch size 16, input size 448, max_pooling layer



# New trainings

- model 13000 : No dropout, no data augmentation learning_rate = 0.0001 sched_step_size = 10 sched_gamma = 0.1 batch_size = 16
- model 26200 : No dropout, horizontal flip learning_rate = 0.0001 sched_step_size = 10 sched_gamma = 0.1 batch_size = 16

-model 52600  No dropout, horizontal flip RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1)) learning_rate = 0.0001 sched_step_size = 10 sched_gamma = 0.1 batch_size = 16
- model 145600 : No dropout, horizontal flip transforms.RandomAffine(180, translate=(0.5, 0.5), scale=(0.7, 1.2)) learning_rate = 0.0001 sched_step_size = 10 sched_gamma = 0.1 batch_size = 16