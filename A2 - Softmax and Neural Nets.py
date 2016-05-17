import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

image_size = 28
num_labels = 10

j = 1
def ref(dset, labels):
  dset = dset.reshape((-1, image_size**2)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dset, labels

def accuracy(predictions, labels):
  return (np.sum(np.argmax(predictions, 1) != np.argmax(labels, 1)))

def singlelayernerualnet(hu = 1000, learning_rate = 0.00001):
  with np.load("notMNIST.npz") as data:
	  images , labels = data["images"], data["labels"]

  hidden1_units = hu
  images = np.transpose(images)
  train_dataset = images[:15000]
  train_labels = labels[:15000,0]
  valid_labels = labels[15000:16000,0]
  valid_dataset = images[15000:16000]
  test_dataset = images[16000:]
  test_labels = labels[16000:,0]

  train_dataset, train_labels = ref(train_dataset, train_labels)
  valid_dataset, valid_labels = ref(valid_dataset, valid_labels)
  test_dataset, test_labels = ref(test_dataset, test_labels)


  batch_size = 100

  graph = tf.Graph()
  with graph.as_default():

    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_verify = tf.constant(train_dataset)
    tf_verifylabels = tf.constant(train_labels)
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_valid_labels = tf.constant(valid_labels)
    tf_test_dataset = tf.constant(test_dataset)
    
    # Variables.
    weightsh1 = tf.Variable(tf.truncated_normal([image_size * image_size, hidden1_units]))
    biasesh1 = tf.Variable(tf.ones([hidden1_units]))
    weights = tf.Variable(tf.truncated_normal([hidden1_units, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels])) 
    
    # Training computation.
    hidden1 = tf.nn.relu(tf.matmul(tf_train_dataset, weightsh1) + biasesh1)
    hiddenverify = tf.nn.relu(tf.matmul(tf_verify, weightsh1) + biasesh1)
    hidden1v = tf.nn.relu(tf.matmul(tf_valid_dataset, weightsh1) + biasesh1)
    hidden1t = tf.nn.relu(tf.matmul(tf_test_dataset, weightsh1) + biasesh1)
    logits = tf.matmul(hidden1, weights) + biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    train_ce = tf.nn.softmax_cross_entropy_with_logits(tf.matmul(hiddenverify, weights) + biases, tf_verifylabels)
    valid_ce = tf.nn.softmax_cross_entropy_with_logits(tf.matmul(hidden1v, weights) + biases, tf_valid_labels)
    
    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.matmul(hidden1v, weights) + biases)
    tf_verify_prediction = tf.nn.softmax(tf.matmul(hiddenverify, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(hidden1t, weights) + biases)

  num_steps = 1050*150+1

  epoch = []
  train_errors = []
  valid_errors = []
  train_loss = []
  valid_loss = []

  with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print("Initialized")
    for step in range(num_steps):
      # Pick an offset within the training data, which has been randomized.
      # Note: we could use better randomization across epochs.
      offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
      # Generate a minibatch.
      batch_data = train_dataset[offset:(offset + batch_size), :]
      batch_labels = train_labels[offset:(offset + batch_size), :]
      # Prepare a dictionary telling the session where to feed the minibatch.
      # The key of the dictionary is the placeholder node of the graph to be fed,
      # and the value is the numpy array to feed to it.
      feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
      _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
      if (step % 150 == 0):
        print step
        train_loss.append(-np.mean(train_ce.eval()))
        valid_loss.append(-np.mean(valid_ce.eval()))
        epoch.append(step/150)
        train_errors.append(accuracy(tf_verify_prediction.eval(), train_labels))
        valid_errors.append(accuracy(valid_prediction.eval(), valid_labels))

    test_errors = accuracy(test_prediction.eval(), test_labels)
    print("Test Errors: %.1f" % test_errors)

    plt.plot(epoch, train_errors, 'r', label = 'Training Errors')
    plt.title('Training Errors vs Epochs')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Training Errors")
    plt.legend(loc = 'upper right')
    plt.show()

    plt.plot(epoch, valid_errors, 'b', label = 'Validation Errors')
    plt.title('Validation Errors vs Epochs')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Validation Errors")
    plt.legend(loc = 'upper right')
    plt.show()

    plt.plot(epoch, train_loss, 'g', label = 'Training Log-Likelihood')
    plt.plot(epoch, valid_loss, 'b', label = 'Validation Log-Likelihood')
    plt.title('Log-Likelihood vs Epochs')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Log-Likelihood")
    plt.legend(loc = 'lower right')
    plt.show()

    return valid_errors[-1], test_errors




def twolayerneuralnet(hu = 500, learning_rate = 0.00001):
  with np.load("notMNIST.npz") as data:
	  images , labels = data["images"], data["labels"]

  hidden1_units = hu
  images = np.transpose(images)
  train_dataset = images[:15000]
  train_labels = labels[:15000,0]
  valid_labels = labels[15000:16000,0]
  valid_dataset = images[15000:16000]
  test_dataset = images[16000:]
  test_labels = labels[16000:,0]

  train_dataset, train_labels = ref(train_dataset, train_labels)
  valid_dataset, valid_labels = ref(valid_dataset, valid_labels)
  test_dataset, test_labels = ref(test_dataset, test_labels)

  batch_size = 100

  graph = tf.Graph()
  with graph.as_default():

    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_verify = tf.constant(train_dataset)
    tf_verifylabels = tf.constant(train_labels)
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_valid_labels = tf.constant(valid_labels)
    tf_test_dataset = tf.constant(test_dataset)
    
    # Variables.
    weightsh1 = tf.Variable(tf.truncated_normal([image_size * image_size, hidden1_units]))
    biasesh1 = tf.Variable(tf.ones([hidden1_units]))
    weightsh2 = tf.Variable(tf.truncated_normal([hidden1_units, hidden1_units]))
    biasesh2 = tf.Variable(tf.ones([hidden1_units]))
    weights = tf.Variable(tf.truncated_normal([hidden1_units, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels])) 
    
    # Training computation.
    hidden1 = tf.nn.relu(tf.matmul(tf_train_dataset, weightsh1) + biasesh1)
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weightsh2) + biasesh2)

    hidden1verify = tf.nn.relu(tf.matmul(tf_verify, weightsh1) + biasesh1)
    hidden2verify = tf.nn.relu(tf.matmul(hidden1verify, weightsh2) + biasesh2)
    
    hidden1v = tf.nn.relu(tf.matmul(tf_valid_dataset, weightsh1) + biasesh1)
    hidden2v = tf.nn.relu(tf.matmul(hidden1v, weightsh2) + biasesh2)

    hidden1t = tf.nn.relu(tf.matmul(tf_test_dataset, weightsh1) + biasesh1)    
    hidden2t = tf.nn.relu(tf.matmul(hidden1t, weightsh2) + biasesh2)

    logits = tf.matmul(hidden2, weights) + biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    train_ce = tf.nn.softmax_cross_entropy_with_logits(tf.matmul(hidden2verify, weights) + biases, tf_verifylabels)
    valid_ce = tf.nn.softmax_cross_entropy_with_logits(tf.matmul(hidden2v, weights) + biases, tf_valid_labels)
    
    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.matmul(hidden2v, weights) + biases)
    tf_verify_prediction = tf.nn.softmax(tf.matmul(hidden2verify, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(hidden2t, weights) + biases)

  num_steps = 1050*150 + 1

  epoch = []
  train_errors = []
  valid_errors = []
  train_loss = []
  valid_loss = []

  with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print("Initialized")
    for step in range(num_steps):
      # Pick an offset within the training data, which has been randomized.
      # Note: we could use better randomization across epochs.
      offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
      # Generate a minibatch.
      batch_data = train_dataset[offset:(offset + batch_size), :]
      batch_labels = train_labels[offset:(offset + batch_size), :]
      # Prepare a dictionary telling the session where to feed the minibatch.
      # The key of the dictionary is the placeholder node of the graph to be fed,
      # and the value is the numpy array to feed to it.
      feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
      _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
      if (step % 150 == 0):
        print step
        train_loss.append(-np.mean(train_ce.eval()))
        valid_loss.append(-np.mean(valid_ce.eval()))
        epoch.append(step/150)
        train_errors.append(accuracy(tf_verify_prediction.eval(), train_labels))
        valid_errors.append(accuracy(valid_prediction.eval(), valid_labels))

    test_errors = accuracy(test_prediction.eval(), test_labels)
    print("Test Errors: %.1f" % test_errors)

    plt.plot(epoch, train_errors, 'r', label = 'Training Error')
    plt.title('Training Errors vs Epochs')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Training Errors")
    plt.legend(loc = 'upper right')
    plt.show()

    plt.plot(epoch, valid_errors, 'b', label = 'Validation Error')
    plt.title('Errors vs Epochs')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Validation Errors")
    plt.legend(loc = 'upper right')
    plt.show()

    plt.plot(epoch, train_loss, 'g', label = 'Training Log-Likelihood')
    plt.plot(epoch, valid_loss, 'b', label = 'Validation Log-Likelihood')
    plt.title('Log-Likelihood vs Epochs')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Log-Likelihood")
    plt.legend(loc = 'lower right')
    plt.show()

    return valid_errors[-1], test_errors







def singlelayerdropout(hu = 1000, learning_rate = 0.0001):
  with np.load("notMNIST.npz") as data:
	  images , labels = data["images"], data["labels"]

  hidden1_units = hu
  images = np.transpose(images)
  train_dataset = images[:15000]
  train_labels = labels[:15000,0]
  valid_labels = labels[15000:16000,0]
  valid_dataset = images[15000:16000]
  test_dataset = images[16000:]
  test_labels = labels[16000:,0]

  train_dataset, train_labels = ref(train_dataset, train_labels)
  valid_dataset, valid_labels = ref(valid_dataset, valid_labels)
  test_dataset, test_labels = ref(test_dataset, test_labels)


  batch_size = 100

  graph = tf.Graph()
  with graph.as_default():

    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_verify = tf.constant(train_dataset)
    tf_verifylabels = tf.constant(train_labels)
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_valid_labels = tf.constant(valid_labels)
    tf_test_dataset = tf.constant(test_dataset)
    
    # Variables.
    weightsh1 = tf.Variable(tf.truncated_normal([image_size * image_size, hidden1_units]))
    biasesh1 = tf.Variable(tf.ones([hidden1_units]))
    weights = tf.Variable(tf.truncated_normal([hidden1_units, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels])) 
   
    # Training computation.
    hidden1 = tf.nn.relu(tf.matmul(tf_train_dataset, weightsh1) + biasesh1)
    hiddenverify = tf.nn.relu(tf.matmul(tf_verify, weightsh1) + biasesh1)
    hidden1v = tf.nn.relu(tf.matmul(tf_valid_dataset, weightsh1) + biasesh1)
    hidden1t = tf.nn.relu(tf.matmul(tf_test_dataset, weightsh1) + biasesh1)

	# Dropout
    keep_prob = tf.placeholder(tf.float32)
    dropouth1 = tf.nn.dropout(hidden1, keep_prob)

    logits = tf.matmul(dropouth1, weights) + biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    train_ce = tf.nn.softmax_cross_entropy_with_logits(tf.matmul(hiddenverify, weights) + biases, tf_verifylabels)
    valid_ce = tf.nn.softmax_cross_entropy_with_logits(tf.matmul(hidden1v, weights) + biases, tf_valid_labels)
    
    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.matmul(hidden1v, weights) + biases)
    tf_verify_prediction = tf.nn.softmax(tf.matmul(hiddenverify, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(hidden1t, weights) + biases)

  num_steps = 150*1050 + 1

  epoch = []
  train_errors = []
  valid_errors = []
  train_loss = []
  valid_loss = []

  with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print("Initialized")
    for step in range(num_steps):
      # Pick an offset within the training data, which has been randomized.
      # Note: we could use better randomization across epochs.
      offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
      # Generate a minibatch.
      batch_data = train_dataset[offset:(offset + batch_size), :]
      batch_labels = train_labels[offset:(offset + batch_size), :]
      # Prepare a dictionary telling the session where to feed the minibatch.
      # The key of the dictionary is the placeholder node of the graph to be fed,
      # and the value is the numpy array to feed to it.
      feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob: 0.5}
      _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
      if (step % 150 == 0):
        print step
        train_loss.append(-np.mean(train_ce.eval()))
        valid_loss.append(-np.mean(valid_ce.eval()))
        epoch.append(step/150)
        train_errors.append(accuracy(tf_verify_prediction.eval(), train_labels))
        valid_errors.append(accuracy(valid_prediction.eval(), valid_labels))
    test_errors = accuracy(test_prediction.eval(), test_labels)
    print("Test Errors: %.1f" % test_errors)

    plt.plot(epoch, train_errors, 'r', label = 'Training Errors')
    plt.title('Training Errors vs Epochs')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Training Errors")
    plt.legend(loc = 'upper right')
    plt.show()

    plt.plot(epoch, valid_errors, 'b', label = 'Validation Error')
    plt.title('Validation Errors vs Epochs')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Validation Errors")
    plt.legend(loc = 'upper right')
    plt.show()

    plt.plot(epoch, train_loss, 'g', 'Training Log-Likelihood')
    plt.plot(epoch, valid_loss, 'b', 'Validation Log-Likelihood')
    plt.title('Log-Likelihood vs Epochs')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Log-Likelihood")
    plt.legend(loc = 'lower right')
    plt.show()

    return train_errors[-1], valid_errors[-1]


def task1(learning_rate = 0.0001):
  with np.load("notMNIST.npz") as data:
	  images , labels = data["images"], data["labels"]

  images = np.transpose(images)
  train_dataset = images[:15000]
  train_labels = labels[:15000,0]
  valid_labels = labels[15000:16000,0]
  valid_dataset = images[15000:16000]
  test_dataset = images[16000:]
  test_labels = labels[16000:,0]
  test_labels.shape

  train_dataset, train_labels = ref(train_dataset, train_labels)
  valid_dataset, valid_labels = ref(valid_dataset, valid_labels)
  test_dataset, test_labels = ref(test_dataset, test_labels)


  batch_size = 100

  graph = tf.Graph()
  with graph.as_default():

    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_verify = tf.constant(train_dataset)
    tf_verify.get_shape()
    tf_verifylabels = tf.constant(train_labels)
    tf_verifylabels.get_shape()
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_valid_labels = tf.constant(valid_labels)
    tf_test_dataset = tf.constant(test_dataset)
    
    # Variables.
    weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))
    
    # Training computation.
    logits = tf.matmul(tf_train_dataset, weights) + biases
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)
    loss = tf.reduce_mean(cross_entropy)
    ce = tf.nn.softmax_cross_entropy_with_logits(tf.matmul(tf_verify, weights) + biases, tf_verifylabels)
    cev = tf.nn.softmax_cross_entropy_with_logits(tf.matmul(tf_valid_dataset, weights) + biases, tf_valid_labels)
    
    # Optimizer.
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.85).minimize(loss)
    
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
    tf_verify_prediction = tf.nn.softmax(tf.matmul(tf_verify, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

  num_steps = 1050*150 + 1

  ep = []
  tr = []
  vl = []
  los = []
  llval = []

  with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print("Initialized")
    for step in range(num_steps):
      # Pick an offset within the training data, which has been randomized.
      # Note: we could use better randomization across epochs.
      offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
      # Generate a minibatch.
      batch_data = train_dataset[offset:(offset + batch_size), :]
      batch_labels = train_labels[offset:(offset + batch_size), :]
      # Prepare a dictionary telling the session where to feed the minibatch.
      # The key of the dictionary is the placeholder node of the graph to be fed,
      # and the value is the numpy array to feed to it.
      feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
      _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
      if (step % 150 == 0):
        c = ce.eval()
        cv = cev.eval()
        los.append(-np.mean(c))
        llval.append(-np.mean(cv))
        ep.append(step/150)
        tr.append(accuracy(tf_verify_prediction.eval(), train_labels))
        vl.append(accuracy(valid_prediction.eval(), valid_labels))

    test_errors = accuracy(test_prediction.eval(), test_labels)
    print("Test Erros: %.1f" % test_errors)

    plt.plot(ep, tr, 'r', label = 'Training Errors')
    plt.title('Training Errors vs Epochs')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Training Errors")
    plt.legend(loc = 'upper right')
    plt.show()

    plt.plot(ep, vl, 'b', label = 'Validation Error')
    plt.title('Validation Errors vs Epochs')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Validation Errors")
    plt.legend(loc = 'upper right')
    plt.show()

    plt.plot(ep, los, 'g', label = 'Training Log-Likelihood')
    plt.plot(ep, llval, 'b', label = 'Validation Log-Likelihood')
    plt.title('Log-Likelihood vs Epochs')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Log-Likelihood")
    plt.legend(loc = 'lower right')
    plt.show()
  return vl[-1], test_errors

def task2():
  singlelayernerualnet(1000)

def task3():
  for h in [100, 500, 1000]:
    singlelayernerualnet(h)

def task4():
  twolayerneuralnet(1000)

def task5():
  singlelayerdropout()

def task6():
  f = open('output.txt', 'r+')
  global j
  j = 1
  #Format of tuples to be added (Expt Number, 'type', learning rate, hidden units, layers, validation errors, test errors)
  for i in range(1):
    t = int(time.time())
    np.random.seed(t)
    dropout = np.random.randint(0, 2)
    hidden_units = np.random.randint(100, 501)
    layer = np.random.randint(0,3)
    learning_rate = float(np.power(10,(np.random.uniform(-5,-4))))
    if dropout == 1 and layer == 1:
      print 'dropout with l = ', learning_rate, 'hu = ', hidden_units 
      valE, testE = singlelayerdropout(hidden_units, learning_rate)
      f.write('Run Number: %f, Type: %s, Learning Rate: %f, Hidden Units: %f, Layers: %f, Validation Errors: %f, Test Errors: %f \n' %(j, 'dropout', learning_rate, hidden_units, 1, valE, testE))
    else:
      if layer == 0:
        print '0 layer with l = ', learning_rate
        valE , testE = task1(learning_rate)
        f.write('Run Number: %f, Type: %s, Learning Rate: %f, Hidden Units: %f, Layers: %f, Validation Errors: %f, Test Errors: %f \n \n' %(j,'O Hidden Layer',learning_rate,hidden_units,layer,valE,testE))
      elif layer == 1:
        print '1 layer with l = ', learning_rate, 'hu = ', hidden_units
        valE, testE = singlelayernerualnet(hidden_units, learning_rate)
        f.write('Run Number: %f, Type: %s, Learning Rate: %f, Hidden Units: %f, Layers: %f, Validation Errors: %f, Test Errors: %f \n' %(j,'1 Hidden Layer', learning_rate, hidden_units, layer, valE, testE))
      else:
        print '2 layer with l = ', learning_rate, 'hu = ', hidden_units
        valE, testE = twolayerneuralnet(hidden_units, learning_rate)
        f.write('Run Number: %f, Type: %s, Learning Rate: %f, Hidden Units: %f, Layers: %f, Validation Errors: %f, Test Errors: %f \n' %(j,'2 Hidden Layers', learning_rate, hidden_units, layer, valE, testE))
    j += 1

  f.close()


#singlelayernerualnet(500, 0.000001)
'''
t3500 Test Errors: 430.0
t4 Test Errors: 388.0
t5 Test Errors: 299.0
t5 Test Errors: 283.0

('O Hidden Layer', 0.0012353361286592006, 419, 0, 166, 469)
('dropout', 0.00027173048060239987, 437, 1, 5224, 439)
('O Hidden Layer', 0.00368091817125338, 411, 0, 188, 462)
('dropout', 0.00015707376642320518, 294, 1, 7539, 558)
('1 Hidden Layer', 0.00032534238781468765, 386, 1, 144, 413)
'''
twolayerneuralnet(500, 0.00001)
