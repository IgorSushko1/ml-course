class KNearestNeighbor(object):

  def __init__(self):
    pass

  def train(self, X, y):

    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):

    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                """"
               ####################################################################
                TODO:                                                            
                Compute the l2 distance between the ith test point and the jth   
                training point, and store the result in dists[i, j]. You should  
                not use a loop over dimension, nor use np.linalg.norm().         
               ####################################################################
                *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****"""
                dists[i, j] = np.sqrt(np.sum(np.square(X[i] - self.X_train[j])))
                """*****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****"""
        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            """######################################################################
            TODO:                                                              
            Compute the l2 distance between the ith test point and all training
            points, and store the result in dists[i, :].                       
            Do not use np.linalg.norm().                                       
            ######################################################################"""

            """*****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****"""
            dists[i, :] = np.sqrt(np.sum(np.square(X[i] - self.X_train), axis=1))
            """*****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****"""
        return dists

  def compute_distances_no_loops(self, X):

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        """########################################################################
        TODO:                                                                
        Compute the l2 distance between all test points and all training     
        points without using any explicit loops, and store the result in     
        dists.                                                               
                                                                             
        You should implement this function using only basic array operations;
        in particular you should not use functions from scipy,               
        nor use np.linalg.norm().                                            
                                                                             
        HINT: Try to formulate the l2 distance using matrix multiplication   
              and two broadcast sums.                                        
        ########################################################################
        ***START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)***** """

        dists = np.sqrt(np.sum(X**2, axis=1)[:, np.newaxis] + np.sum(self.X_train**2, axis=1) - 2 * np.dot(X, self.X_train.T))
        """ *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)***** """
        return dists

  def predict_labels(self, dists, k=1):

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            """A list of length k storing the labels of the k nearest neighbors to
            the ith test point.
           ########################################################################
            TODO:                                                                
            Use the distance matrix to find the k nearest neighbors of the ith   
            testing point, and use self.y_train to find the labels of these      
            neighbors. Store these labels in closest_y.                          
            Hint: Look up the function numpy.argsort.                            
           ########################################################################
            *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****"""
            closest_y = self.y_train[np.argsort(dists[i])[:k]]
            """*****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
           ########################################################################
            TODO:                                                                
            Now that you have found the labels of the k nearest neighbors, you   
            need to find the most common label in the list closest_y of labels.  
            Store this label in y_pred[i]. Break ties by choosing the smaller    
            label.                                                               
           ########################################################################
            *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****"""
            y_pred[i] = np.argmax(np.bincount(closest_y))

            """*****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****"""

    return y_pred

from k_nearest_neighbor import KNearestNeighbor
classifier = KNearestNeighbor()
classifier.fit(X_train, y_train)