# INFO

- model 51000 : First trial with errors in the loss function
- model 39000 : Second trial with a lot of overfitting
- model 86400 : No dropout, vertical & horizontal flip for data augmentation
- model 235000 : 0.2 dropout, horizontal flip only
- model 178800 : 0.2 dropout, horizontal flip only, learning rate decay (x0.1 every 10 epochs)
- model 113800 : No dropout, horizontal flip only, learning rate decay (x0.1 every 10 epochs)
- model 31800 : No dropout, horizontal flip only, learning rate decay (x0.1 every 10 epochs) batch size 32

- model 1 : 0.01 dropout, horizontal flip only, learning rate decay (x0.5 every 10 epochs)
- model 3 : No dropout, horizontal flip only, learning rate decay (x0.1 every 10 epochs) batch size 16, input size 448, max_pooling layer