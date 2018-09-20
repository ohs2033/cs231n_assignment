import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  delta = 1
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + delta # note delta = 1
      if margin > 0:
        loss += margin

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  loss_values = []

  for i in range(num_train):
    gradients = np.zeros(num_classes)
    scores = X[i].dot(W)
    correct_class_scores = scores[y[i]]
    hinge_losses = np.maximum(0, scores - correct_class_scores + delta)
    hinge_losses[y[i]] = 0
    total_loss = np.sum(hinge_losses)

    
    loss_mask = np.array(hinge_losses > 0, dtype=int)

    
    # calculate other class gradient
    gradients = np.multiply(loss_mask.reshape(10,1), X[i])

    # if gradients[y[i]] != 0:
      # raise ValueError('invalid y value ', y[i])
    y_gradient = - np.sum(loss_mask) * X[i]
    # print(y_gradient.shape)
    gradients[y[i], :] =  y_gradient
    gradients = np.transpose(gradients)
    dW += gradients


  dW /= num_train


  return loss, dW



def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero


  num_class = W.shape[1]
  num_train = X.shape[0]

  scores = np.dot(X,W)
  print(scores.shape) # (num_train, 10)
  print(dW.shape)

  #[sy1, sy2, sy3, sy4]
  y_scores = np.choose(y, scores.T)

  print('y scores vector ', y_scores.shape)

  scores -= y_scores.reshape(500,1)
  scores += 1
  scores = np.maximum(0, scores)
  scores[range(num_train), y] = 0

  loss = np.sum(scores)

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)



  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  mask = np.array(scores != 0 , dtype=int)
  gradients = np.multiply(X, scores.reshape(num_train, num_class))

  print(gradients.shape)
  # scores[[range(num_train)]] - scores[]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
