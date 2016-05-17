import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def KMeansClusters(k = 3, dim=2):
	data2D = (np.load("data2D.npy")).astype("float32")
	centroids2D = (np.random.normal(0, 1, (k, dim))).astype("float32")

	plt.plot(data2D[:,0], data2D[:,1], 'wo')
	plt.plot(centroids2D[:,0], centroids2D[:,1], 'b^')
	plt.show()

	graph = tf.Graph()
	with graph.as_default():

		# define constants, variables, placeholders
		data_tf = tf.constant(data2D)
		centroids_tf = tf.Variable(centroids2D)
		expanded_data = tf.expand_dims(data_tf, 1)
		expanded_centroids = tf.expand_dims(centroids_tf, 0)
		distance = tf.reduce_sum(tf.square(tf.sub(expanded_data, expanded_centroids)), 2)
		if k == 1:
			mins = distance
		if k > 1:
			mins = tf.minimum(distance[:,0], distance[:,1]) # gets min of 1 and 2
		if k > 2:
			mins = tf.minimum(mins, distance[:,2])
		if k > 3:
			mins = tf.minimum(mins, distance[:,3])
		if k > 4:
			mins = tf.minimum(mins, distance[:,4])
		
		assignments = tf.argmin(distance, 1)		

		# define a loss function
		loss = tf.reduce_sum(mins)

		# define an optimizer
		optimizer = tf.train.AdamOptimizer(0.01, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(loss)

	# lists for anything to pull out of the session
	loss_list = []
	c = None
	a = None
	d = None

	with tf.Session(graph=graph) as session:
		tf.initialize_all_variables().run()

		for step in range(1500):
			_, l, c, a, d = session.run([optimizer, loss, centroids_tf, assignments, distance])
			if step % 100 == 0:
				print "Step #", step
			loss_list.append(l)

	print "LOSS: ", loss_list[-1]

	plt.plot(range(len(loss_list)), loss_list, 'r', label = 'Loss')
	plt.title('Loss vs Iteration')
	plt.xlabel("Number of Iterations")
	plt.ylabel("Loss")
	plt.legend(loc = 'upper right')
	plt.show() 

	group1 = []
	group2 = []
	group3 = []
	group4 = []
	group5 = []

	for i in range(len(data2D)):
		if a[i] == 0:
			group1.append(data2D[i])
		elif a[i] == 1:
			group2.append(data2D[i])
		elif a[i] == 2:
			group3.append(data2D[i])
		elif a[i] == 3:
			group4.append(data2D[i])
		elif a[i] == 4:
			group5.append(data2D[i])

	group1 = np.array(group1)
	group2 = np.array(group2)
	group3 = np.array(group3)
	group4 = np.array(group4)
	group5 = np.array(group5)

	#get variance for first cluster
	if k == 2:
		var1 = np.var(group1,axis=0)
		var2 = np.var(group2,axis=0)		
		avg = (var1 + var2)/2
	if k == 3:
		var1 = np.var(group1,axis=0)		
		var2 = np.var(group2,axis=0)
		var3 = np.var(group3,axis=0)
		avg = (var1 + var2 + var3)/3
	if k == 4:
		var1 = np.var(group1,axis=0)		
		var2 = np.var(group2,axis=0)
		var3 = np.var(group3,axis=0)
		var4 = np.var(group4,axis=0)
		avg = (var1 + var2 + var3 + var4)/4
	if k > 4:
		var1 = np.var(group1,axis=0)		
		var2 = np.var(group2,axis=0)
		var3 = np.var(group3,axis=0)
		var4 = np.var(group4,axis=0)		
		var5 = np.var(group5,axis=0)
		avg = (var1 + var2 + var3 + var4 +var5)/5

	#return list(avg)
	if k == 1:
		plt.plot(data2D[:,0],data2D[:,1], 'ro')
	else:
		plt.plot(group1[:,0], group1[:,1], 'ro')
		if group2.size > 0:
			plt.plot(group2[:,0], group2[:,1], 'bo')
		if group3.size > 0:
			plt.plot(group3[:,0], group3[:,1], 'go')
		if group4.size > 0:
			plt.plot(group4[:,0], group4[:,1], 'co')
		if group5.size > 0:
			plt.plot(group5[:,0], group5[:,1], 'ko')
	plt.plot(c[:,0], c[:,1], 'm^')
	plt.show()

def reduce_logsumexp(input_tensor, reduction_indices=1, keep_dims=False):
  """Computes the sum of elements across dimensions of a tensor in log domain.
     
     It uses a similar API to tf.reduce_sum.

  Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    reduction_indices: The dimensions to reduce. 
    keep_dims: If true, retains reduced dimensions with length 1.
  Returns:
    The reduced tensor.
  """
  max_input_tensor1 = tf.reduce_max(input_tensor, 
                                    reduction_indices, keep_dims=keep_dims)
  max_input_tensor2 = max_input_tensor1
  if not keep_dims:
    max_input_tensor2 = tf.expand_dims(max_input_tensor2, 
                                       reduction_indices) 
  return tf.log(tf.reduce_sum(tf.exp(input_tensor - max_input_tensor2), 
                                reduction_indices, keep_dims=keep_dims)) + max_input_tensor1

def logsoftmax(input_tensor):
  """Computes normal softmax nonlinearity in log domain.

     It can be used to normalize log probability.
     The softmax is always computed along the second dimension of the input Tensor.     
 
  Args:
    input_tensor: Unnormalized log probability.
  Returns:
    normalized log probability.
  """
  return input_tensor - reduce_logsumexp(input_tensor, 0, keep_dims=True)


def MoG(k = 3, dim=2, f = "data2D.npy"):
	data2D = (np.load(f)).astype("float32")
	centroids2D = (np.random.normal(0, 1, (k, dim))).astype("float32")
	graph = tf.Graph()

	plt.plot(data2D[:,0], data2D[:,1], 'ro')
	plt.plot(centroids2D[:,0], centroids2D[:,1], 'b^')
	plt.show()

	with graph.as_default():
		# define a loss function
		data_tf = tf.constant(data2D)
		expanded_data = tf.expand_dims(data_tf, 1)

		centroid_means = tf.Variable(centroids2D)
		expanded_means = tf.expand_dims(centroid_means, 0)
		distance = tf.reduce_sum(tf.square(tf.sub(expanded_data, expanded_means)),2)

		centroid_phi = tf.Variable(tf.random_normal(shape = [centroids2D.shape[0]]))
		centroid_var = tf.exp(centroid_phi)
		expanded_var = tf.expand_dims(centroid_var, 0)
		centroid_psi = tf.Variable(tf.random_normal(shape = [centroids2D.shape[0]]))		
	
		coeff = -0.5 * tf.log(centroid_var * 2 * np.pi)
		gaussian = -0.5 * (tf.div(distance,expanded_var))
		likelihood = tf.add(coeff, gaussian)
		prior = logsoftmax(tf.log(tf.abs(centroid_psi)))
		posterior = tf.add(likelihood, prior)
		
		assignments = tf.argmax(posterior, 1)

		loss = -1.0 * tf.reduce_sum(reduce_logsumexp(posterior,1))
		optimizer = tf.train.AdamOptimizer(0.001, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(loss)

	loss_list = []
	c = None
	v = None
	a = None
	p = None
	with tf.Session(graph=graph) as session:
		tf.initialize_all_variables().run()

		for step in range(10000):
			_, l, c, v, a, p = session.run([optimizer, loss, centroid_means, centroid_var, assignments, prior])
			loss_list.append(l)
			if step % 500 == 0:
				print "Step #", step
			
	group1 = []
	group2 = []
	group3 = []
	group4 = []
	group5 = []

	for i in range(len(data2D)):
		if a[i] == 0:
			group1.append(data2D[i])
		elif a[i] == 1:
			group2.append(data2D[i])
		elif a[i] == 2:
			group3.append(data2D[i])
		elif a[i] == 3:
			group4.append(data2D[i])
		elif a[i] == 4:
			group5.append(data2D[i])

	group1 = np.array(group1)
	group2 = np.array(group2)
	group3 = np.array(group3)
	group4 = np.array(group4)
	group5 = np.array(group5)

	print 'g1 ', group1.size
	print 'g2 ', group2.size
	print 'g3 ', group3.size
	print 'g4 ', group4.size
	print 'g5 ', group5.size

	if k == 1:
		plt.plot(data2D[:,0],data2D[:,1], 'ro')
	else:
		if group1.size > 0:
			plt.plot(group1[:,0], group1[:,1], 'ro')
		if group2.size > 0:
			plt.plot(group2[:,0], group2[:,1], 'bo')
		if group3.size > 0:
			plt.plot(group3[:,0], group3[:,1], 'go')
		if group4.size > 0:
			plt.plot(group4[:,0], group4[:,1], 'co')
		if group5.size > 0:
			plt.plot(group5[:,0], group5[:,1], 'ko')
	plt.plot(c[:,0], c[:,1], 'b^')
	plt.show()

def validLoss(validData, centroid_means,centroid_psi,centroid_var,centroids2D,k):
	graph = tf.Graph()
	print validData
	with graph.as_default():
		# define a loss function
		data_tf = tf.constant(validData)
		expanded_data = tf.expand_dims(data_tf, 1)

		expanded_means = tf.expand_dims(centroid_means, 0)
		distance = tf.reduce_sum(tf.square(tf.sub(expanded_data, expanded_means)),2)

		expanded_var = tf.expand_dims(centroid_var, 0)		
	
		coeff = -0.5 * tf.log(centroid_var * 2 * np.pi)
		gaussian = -0.5 * (tf.div(distance,expanded_var))
		likelihood = tf.add(coeff, gaussian)
		prior = logsoftmax(tf.log(tf.abs(centroid_psi)))
		posterior = tf.add(likelihood, prior)
		
		assignments = tf.argmax(posterior, 1)

		loss = -1.0 * tf.reduce_sum(reduce_logsumexp(posterior,1))

	loss_list = 0
	a = None
	with tf.Session(graph=graph) as session:
		tf.initialize_all_variables().run()
		l, a = session.run([loss, assignments])
		loss_list = l
	print "Loss: ", loss_list

	group1 = []
	group2 = []
	group3 = []
	group4 = []
	group5 = []

	for i in range(len(validData)):
		if a[i] == 0:
			group1.append(validData[i])
		elif a[i] == 1:
			group2.append(validData[i])
		elif a[i] == 2:
			group3.append(validData[i])
		elif a[i] == 3:
			group4.append(validData[i])
		elif a[i] == 4:
			group5.append(validData[i])

	group1 = np.array(group1)
	group2 = np.array(group2)
	group3 = np.array(group3)
	group4 = np.array(group4)
	group5 = np.array(group5)

	if k == 1:
		plt.plot(validData[:,0],validData[:,1], 'ro')
	else:
		if group1.size > 0:
			plt.plot(group1[:,0], group1[:,1], 'ro')
		if group2.size > 0:
			plt.plot(group2[:,0], group2[:,1], 'bo')
		if group3.size > 0:
			plt.plot(group3[:,0], group3[:,1], 'go')
		if group4.size > 0:
			plt.plot(group4[:,0], group4[:,1], 'co')
		if group5.size > 0:
			plt.plot(group5[:,0], group5[:,1], 'ko')
	plt.plot(centroid_means[:,0], centroid_means[:,1], 'm^')
	plt.show()
				

def MoGValidation(k = 3, dim=2, f = "data2D.npy"):
	data2D = (np.load(f)).astype("float32")
	validData = data2D[2*len(data2D)/3:]
	trainData = data2D[:2*len(data2D)/3]
	centroids2D = (np.random.normal(0, 1, (k, dim))).astype("float32")
	graph = tf.Graph()

	plt.plot(trainData[:,0], trainData[:,1], 'ro')
	plt.plot(centroids2D[:,0], centroids2D[:,1], 'b^')
	plt.show()

	with graph.as_default():
		# define a loss function
		data_tf = tf.constant(trainData)
		expanded_data = tf.expand_dims(data_tf, 1)

		centroid_means = tf.Variable(centroids2D)
		expanded_means = tf.expand_dims(centroid_means, 0)
		distance = tf.reduce_sum(tf.square(tf.sub(expanded_data, expanded_means)),2)

		centroid_phi = tf.Variable(tf.random_normal(shape = [centroids2D.shape[0]]))
		centroid_var = tf.exp(centroid_phi)
		expanded_var = tf.expand_dims(centroid_var, 0)
		centroid_psi = tf.Variable(tf.random_normal(shape = [centroids2D.shape[0]]))		
	
		coeff = -0.5 * tf.log(centroid_var * 2 * np.pi)
		gaussian = -0.5 * (tf.div(distance,expanded_var))
		likelihood = tf.add(coeff, gaussian)
		prior = logsoftmax(tf.log(tf.abs(centroid_psi)))
		posterior = tf.add(likelihood, prior)
		
		assignments = tf.argmax(posterior, 1)

		loss = -1.0 * tf.reduce_sum(reduce_logsumexp(posterior,1))
		optimizer = tf.train.AdamOptimizer(0.001, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(loss)

	loss_list = []
	c = None
	v = None
	with tf.Session(graph=graph) as session:
		tf.initialize_all_variables().run()

		for step in range(10000):
			_, l, c, v, cv = session.run([optimizer, loss, centroid_means, centroid_psi, centroid_var])
			loss_list.append(l)
			if step % 500 == 0:
				print "Step #", step
	validLoss(validData, c, v, cv,centroids2D,k)
