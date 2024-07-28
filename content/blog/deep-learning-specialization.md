---
tag: Deep Learning Basics
aliases: 
publish: true
slug: deep-learning-specialization
title: Review and Summary of Deep Learning Specialization from DeepLearning
description: Notes, assignments and review of the course
date: 2024-07-09
image: /thumbnails/coursera_dl_specialization.png
---
## Introduction

My current employee gave me the chance to take the Deep Learning Specialization from DeepLearn.ai via Coursera. Even though this is quite a basic level course, I decided to give it a try and take the chance to revise my foundations.

There is quite a lot of undeserved praise online for this course. It lacked depth, there were no structured course notes (just videos), and the above all author kept embarking on lengthy "intuitive" explanations aimed at people who did not take a proper calculus course.

If you graduated from university, or if you already have some knowledge on neural networks, this course is probably not suitable for you. Nevertheless, even though I did not learn almost anything new, it was a great chance to revisit basic concepts and get a certificate that can be showcased in LinkedIn.

Below is a summary of the contents developed in the specialisation, based on my notes and takeaways.

## Course 1: Neural Networks and Deep Learning

#### Week 1: Introduction to Deep Learning

This is a very relaxed welcome unit. Neural networks are loosely presented as systems that can learn to solve tasks involving many inputs and outputs, and excel at dealing with unstructured data (images, audio, text)

Structured data, on the other hand,  is highly organized and easily searchable. It is typically stored in relational databases or spreadsheets, where the data is arranged in rows and columns with specific data types.

The teacher emphasizes that neural networks performance scale with the amount of data, and in general are very data-hungry. They are able to surprass traditional ML methods (such as SVM), but they do require substantial quantity of data samples.

#### Week 2: Basics of Neural Network programming

##### Logistic Regression

Binary classification is exemplified with the task of predicting, giving an input image, whether it is a cat picture or a doc picture. 

In this example, RGB images of 64x64 pixels are rearranged into a flat vector of 12288 elements. This is the dimension of the inputs in this setup for this task.

Given a dataset of `m` cat-dog pictures and their labels, they are arranged into matrixes as follows:

X:  A `n` x `m` matrix where each column is one of the training inputs, and `n` is the leng th of the inputs (12288 for this example)
Y: A `1` x `m`  matrix where each column is the label of one of the training inputs.

The teaches introduced a well known, simple classification method: Linear Regression. 

Given an input `x` (e.g. a dog-cat image), it is meant to predict `\hat(y)`  where `0 <= \hat(y) <= 1` by applying a linear function followed by a sigmoid function to squeeze the output of the linear function to `[0,1]`

![[Pasted image 20240709193121.png]]
The parameters of the model are `w` and `b`. Recall the shape of the sigmoid function here:

![[Pasted image 20240709193224.png]]
In order to use the training set of `m` labeled samples to tune the parameters `w` and `b` , a few concepts are introduced:

* The **loss** (error) function **L**, which tells how different a model prediction is from its labelled ground truth. Since this is a binary classification problem, the **Binary Cross-Entropy Loss is introduced**

![[Pasted image 20240709193651.png]]
* The **cost** function **J** is the sum of the loss over the entire training set for any given choice of parameters
![[Pasted image 20240709194013.png]]
* **Gradient descent** is an iterative method used to find w and b that minimize J(w,b). It consists of updating w and b on each iteration by substracting the derivative of J with respect to the parameters. By moving the paremeters in the direction opposite to the gradient, the cost function value is driven down and the model should become better at predicting the training labels. A hyperparameter **alpha**, the learning rate, governs the strength of the parameter update on each iteration.

Here, the teacher goes into A LOT of extra-boring examples trying to impress the meaning of a derivative without actually diving into the math. Having taking in-depth courses of calculus at university, this was a stump.

It's actually quite simple. `f'(a)` gives an approximation of how `f` changes if you slightly modify `a`. Say `f'(a)=2`, then `f'(a+delta) ~= f(a) + 2*delta`
 
It becomes a little more interesting when we talk about derivatives in a computational graph. Here a system takes three inputs, `a`, `b` and `c` ; makes a few intermediate computations `u` and `v` and finally produces an output `J`

![[Pasted image 20240709195439.png]]

The key point here is that, if we want to know the derivative of `J` with respect to the inputs (e.g. `b` or `c`), we need to apply the **chain rule** and go through the intermediate nodes.

![[Pasted image 20240709195647.png]]

In the context of logistic regression, we can calculate the derivative of the loss L by writing it as a computational graph:

![[Pasted image 20240709195909.png]]

This example has inputs with size 2 (thus two weights). We need the derivative of `L` wrt to the parameters `w_1`, `w_2` and `b`.

In order to do so, we first need the intermediate derivatives:

`da = D L(a,y) / D a = -y/a + (1-y)/(1-a)`
`dz = DL/Dz = DL/Da * Da/Dz = da * a*(1-a) = a-y`
`dw1 = DL/dw1 = DL/z * Dz/Dw1 = dz * x1`

After calculating the final derivative, we can update
`w1 := w1 - alpha * dw1`

And do the same for the other parameters.

To scale this up to `m` examples, we need to calculate the derivatives of the cost function `J`. Since the derivative of a sum is the sum of the derivatives, this is quite straightforward.
![[Pasted image 20240709203724.png]]

This section concludes with a snippet of code that makes an iteration of gradient descent, calculating the cumulative gradient over the entire training set. Notice that this would have to be repeated iteratively until convergence.

Also notice that n=2 meaning that x is just x1 and x2. If n is large, this code is inefficient. Vectorization would have to be used to get rid of for loops.

![[Pasted image 20240709211734.png]]

##### Vectorization
Replacing explicit `for` loops by matrix/vector built in operations can considerable speed up your algorithms both in GPU and CPU.

Examples:

* In order to multiply a matrix by a vector, use `np.dot(A,v)`
* In order to apply the same function to each element in a vector `v`, use built-in operations that are very efficient `np.exp(v)`, `np.log(v)`, `np.abs(v)`, `np.maximum(v,0)`,  `v**2`, `1/v`
* use `np.sum` , 

This is mostly exemplified by the assignment 1 tasks, which you can check out in my GitHub.

#### Week 3: Shallow Neural Networks

##### Neural networks

There's a very verbose intro here just to explain the concept of a fully connected neural network and notation.

Basically it is a sequence of layers, each all consecutive layers are fully connected by parameters `w`.

![[Pasted image 20240717163408.png]]
*A neuron computes a weighted sum using a parameter `w` of all its parameters, adds a bias parameter `b` and then applies a non-linear activation function such as sigmoid*

We are introduced to different choices of activation functions. The rule of thumb is: use ReLu, unless you're on the last output layer of a binary classification task, there you can use Sigmoid.

Non-linearities are crucial to represent complex functions, otherwise the entire network collapses into a single linear function!

![[Pasted image 20240717164234.png]]

Here's a FCN with two layers (the input layer doesn't count). The notation is defined as follows:
* `**z^\[i\]\_j**`: The output of the jth neuron at the ith layer.
* `**a^\[i\]\_j**`:  Activation function applied to the previous value
* `**w^\[i\]\_j**`:  weight to connect neurons in layer `{i-1}` to neuron `j` in layer `i`
* `**b^\[i\]\_j**`:  bias used by neuron j at layer i
These values are stacked together to form matrixes that allow for efficient vectorization:
 * `**a^\[i\]**` , `**z^\[i\]**` all activations from layer i
 * `**W^\[i\]**`: Weights that connect layer `{i-1}` to layer `{i}`
 * `**b^\[i\]**`: all biases of neurons at layer i
 
Thus we can rewrite the equations in vectorized form, where the (i) superscript means sample number `i`. We can stack all samples together to vectorize this as well.

![[Pasted image 20240717165639.png]]
##### Gradient Descent for Neural Netoworks

The author presents pseudocode for a single-hidden-layer neural network, and we can see it's quite similar to logistic regression:

![[Pasted image 20240717221009.png]]
The derivatives of the activation functions are:

* Sigmoid:
```
	g(z)  = 1 / (1 + np.exp(-z))
	g'(z) = (1 / (1 + np.exp(-z))) * (1 - (1 / (1 + np.exp(-z))))
	g'(z) = g(z) * (1 - g(z))
```
* RELU
```
    g(z)  = np.maximum(0,z)
    g'(z) = { 0  if z < 0
              1  if z >= 0  }
```

The forward propagation goes as follows:

```
Z^1 = W1A0 + b1    # A^0 is X
A^1 = g1(Z1)
Z^2 = W2A1 + b2
A^2 = Sigmoid(Z2)   # Sigmoid because the output is between 0 and 1
```

Finally, let's do backpropagation:

```
dA^2 = -Y/A^2 + (1-Y)/(1-A^2)   # Calculus
dZ^2 = DL/DZ^2 = DL/DA^2 * DA^2/DZ^2 = dA^2 * (A^2 * (1-A^2)) = A^2 - Y^2
dW^2 =  dZ^2 * DZ^2/DW^2 = (dZ^2 * A^1.T)/m
db2 = Sum(dZ2) / m
dZ1 = (W2.T * dZ2) * g'1(Z1)  # element wise product (*)
dW1 = (dZ1 * A0.T) / m   # A0 = X
db1 = Sum(dZ1) / m
```

![[Pasted image 20240717223618.png]]

Mira el video de la intuition por que:
* no entiendo por que involucra ms si la loss no es la J
* no entiendo por que transpuestas, asumo que algo de calculo de matrices
* no entiendo por que Sum en las derivadas de las bs

![[Pasted image 20240717223809.png]]
*Left: single sample - Right: All training set at once*

So basically you are calculating the direction you have to go to reduce the error for each training sample, and the you average out that direction to update the weights.
##### Random Initialization of Parameters
What happens if we initialize our NN to have all zero weights? 

It's possible to construct a proof by induction that if you initialize all the values of w to 0, each neuron in a given layer will compute the same output because they all start with the same weights and biases. Consequently, the gradients for these neurons will also be identical during backpropagation. This symmetry means that each neuron in a layer will continue to update in the same way, preventing the network from breaking symmetry and learning diverse features. Essentially, the neurons in each layer will remain identical throughout the training process, significantly limiting the model's capacity to learn.

The solution to this is to initialize the parameters randomly to small numbers, e.g.:

`W^1 = np.random.randn((2,2)) * 0.01`

Large numbers would cause activations such as sigmoid or tanh to saturate, making learning slow.

link to the assignment 3.

#### Week 4: Deep Neural Networks

A deep NN is a NN with three or more layers:
* `L`: Number of layers
* `n^[l]` : number of neurons at layer `l`  (`n[0]` number of inputs, `n[L]` number of outputs)
* `g^[l]` : activation function at layer `l`
* `z^[l] = W^[l] * a^[l-1] + b^[l]`   has shape `(n[l],1)`
* `a^[l] = g^[l](z^[l])`  : activation at the layer `l`  has shape `(n[l],1)
* `W^[l]`: weights that connect layer `l-1` to layer `l` has shape `(n[l],n[l-1])`
* `b^[l]`: biases of layer `l`, shape is `(n[l],1)`
* `x = a[0]`, `a[l] = y'`

Vectorized for the entire training set:
* `m`: number of training samples
* `Z^[i] = W^[i] X + b^[i]` has shape `(n[i],m)`
* `X` has shape `(n[0],m)` (i.e. each sample is a column, contrary to the conventions that the rest of the world uses)
* `A[0]=X`
* `A[l]` = `g^[l](Z^[l])` has shape  `(n[i],m)`
* `dZ^[l]` and `dA^[l]` will have shape `(n[i],m)`

Why use deeper networks? earlier layers of the neural network can detecting simple functions, like edges and then
composing them together in the later layers of a neural network so that it can learn more and more complex functions.

This is the typical intuition for hierarchical feature extraction in CNNs (I don't actually think it applies to dense layers though)

![[Pasted image 20240718162825.png]]

Another example with speech recognition. The first level of a NN may detect low lavel audio wave forms and basic phonemes. Later layers may compose this together.

This in some way emulates what our brain does.
##### Generic forward and backward functions (vectorized to all samples)

- Pseudo code for forward propagation for layer l:

    ```
    Input  A[l-1]
    Z[l] = W[l]A[l-1] + b[l]
    A[l] = g[l](Z[l])
    Output A[l], cache(Z[l])
    ```

- Pseudo code for back propagation for layer l:
    ```
    Input da[l], Caches
    dZ[l] = dA[l] * g'[l](Z[l])
    dW[l] = (dZ[l]A[l-1].T) / m
    db[l] = sum(dZ[l])/m                # Dont forget axis=1, keepdims=True
    dA[l-1] = w[l].T * dZ[l]            # The multiplication here are a dot product.
    Output dA[l-1], dW[l], db[l]
    ```

- Cross entropy loss function is:
    ```
    dA[L] = (-(y/a) + ((1-y)/(1-a)))
    ```

##### Parameters and Hyperparameters
* The parameters of the network are `w` and `b` and they are optimized via backpropagation
* Hyperparameters are higher level parameters of the algorithm such as:
	* The learning rate (alpha)
	* The number of iterations until convergence
	* The number of hidden layers `L` and the number of hidden units per hidden layer
	* The choice of activation functions

Hyperparameters are often optimized by trying a few combinations of them and checking which one works better (this is actually developed in depth in the next course)

I cannot recommend this video and its sequel more:  https://www.youtube.com/watch?v=Ilg3gGewQ5U.

## Course 2 - Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization

### Week 1: Practical Aspects of Deep Learning

#### Train / Validation / Test set

In machine learning, we usually iterate through the loop:

`Idea -> Code -> Experiment -> Repeat`

During the process of defining the hyperparameters of a model, it is recommended that the available data is split into three sets:

| Data partition             | Purpose                                       | Size       |
| -------------------------- | --------------------------------------------- | ---------- |
| Training set               | Optimizing model parameters                   | 60 to 98 % |
| Validation set AKA dev set | Optimizing model hyperparameters              | 20 to 1%   |
| Test set                   | Provide an unbiased estimation of performance | 20 to 1%   |
If size of the dataset is greater than 1M we may use 98/1/1 or even 99.5/0.25/0.25

* Train and validation set must come from the same distribution
* The dev set is used to try out trained models with different hyperparameters, in order to define what is the best hyperparameter configuration
* It is ok to only have validation set without testing set if you don't need an unbiased estimation of performance. People call the validation set "test set" in this setup (although it is misleading).
#### Bias and Variance

![[Pasted image 20240725143318.png]]
* A model that underfits has high bias
	* e.g. training error doesn't go down near the optimal classifier error
* A model that overfits has high variance
	* big gap between training and validation error.
* A good model strikes a good balance between bias an variance
	* training and validation error closer to the optimal classifier

A basic recipe for Machine Learning is:

1. If your algorithm has high bias
	1. Make your NN larger (more hidden units or layers) - This never hurts. Still I advise starting smaller.
	2. Try out different models
	3. Train for longer
	4. Use a different optimization algorithm
2. If your algorithm has high variance
	1. Add more data
	2. Regularization
	3. A different model

Historical note: In traditional ML we usually talk about the **bias/variance tradeoff**, but in the deep learning world we can usually try to improve either without making the other worse.

#### Regularization

Regularization techniques reduce overfitting and therefore reduce variance. 

##### Weight Decay

We can apply regularization to the **parameters** of the model by penalising large weights in the cost function. In order words, we try to keep the model from becoming "too complex". This is sometimes called **weight decay**.

The cost function looks like this when we add regularization, where lambda is the regularization hyperparameter:

`J(w,b) = (1/m) * Sum(L(y(i),y'(i))) + (lambda/2m) * Sum((||W[l]||)`

We can use either L1 (Aka Lasso) or L2 (aka Ridge) norm:
* L1 encourages sparsity in `w`, potentially making for a smaller and more efficient model. It is preferred when you think some features are irrelevant.
		`||W|| = Sum(|w[i,j]|)`
* L2 is much more popular. It encourages smaller but non-zero weights.

		`||W||^2 = Sum(|w[i,j]|^2)


Notice that the gradient of L2 is proportional to `2 * lambda* w_i` , meaning that larger weighs experience a proportionally large force pulling them towards zero. This proportionality results in a more uniform shrinkage across all weights. Large weights are reduced significantly, but small weights are reduced less. This uniform shrinkage does not typically result in exact zeros but rather smaller weights overall.

On the other hand, the gradient of L1 is a constant value (lambda or -lambda), except at zero where it's undefined. This constant force encourages weights to move towards zero by a fixed amount, regardless of their size. As a result, small weights can be reduced to exactly zero, creating sparsity.

So, if we have a new `J` with L2 norm, the new backpropagation update is:

`dw[l] = (old backprop value) + lambda/m * w[l]`

Thus the new update step is:
```
w[l] = w[l] - learning_rate * dw[l]
     = w[l] - learning_rate * ((old backprop value) + lambda/m * w[l])
     = w[l] - learning_rate * (from back propagation) - (learning_rate*lambda/m) * w[l]
```

The new term shrinks `w[l]` proportionally to `w[l]`  itself and to the lambda parameter, pulling it towards zero. This is, this penalizes large weights and effectively limits the freedom in your model.

How does weight decay helps?
* Large weights can make the model respond with high sensitivity to the input features. This sensitivity allows the model to fit the training data very closely, including noise. In other words, it makes it easy to **memorize** instead of learn. It also results in more complex decision boundaries.
* Smaller weights constrain the model, reducing its sensitivity to individual input features. This results in a smoother, more general decision boundary that captures the underlying patterns of the data rather than the noise.

Imagine fitting a polynomial to data points:

![[Pasted image 20240725185046.png]]
##### Dropout

The dropout regularization eliminates some neurons (along with all their incoming and outgoing edges) on each iteration based on a probability.

The usual implementation is called **Inverted Dropout**:

```
keep_prob = 0.8   # 0 <= keep_prob <= 1
l = 3  # this code is only for layer 3
# the generated number that are less than 0.8 will be dropped. 80% stay, 20% dropped
d3 = np.random.rand(a[l].shape[0], a[l].shape[1]) < keep_prob

a3 = np.multiply(a3,d3)   # keep only the values in d3

# increase a3 to not reduce the expected value of output
# (ensures that the expected value of a3 remains the same) - to solve the scaling problem
a3 = a3 / keep_prob    
```

To ensure that the scale of the output remains the same, the outputs of the remaining neurons are scaled up. This scaling factor compensates for the dropped neurons, keeping the expected value of the outputs constant.

During inference, dropout is not applied, and the entire network is used. No scaling is needed because the dropout is not active.

Dropout can have different `keep_prob` per layer.

Why does it work?
* Can't rely on any one feature, so have to spread out weights. Otherwise neurons can become overly reliant on specific other neurons, leading to a fragile model.
* By randomly dropping out neurons during training, dropout forces the network to learn more robust and independent features.
* Since neurons cannot rely on specific other neurons, they must learn redundant representations of features. This redundancy makes the model more robust to changes in the input data.
* It works very well on CV tasks, empirically.
##### Data augmentation

Often used in CV data, it involves flipping, rotating, distorting the images. This makes the model able to generalize better to slight changes.
##### Early stopping

In this technique we plot the training set and the dev set cost together for each iteration. At some iteration the dev set cost will stop decreasing and will start increasing. We pick the point where the validation set is lowest.

##### **Model Ensembles**

- Train multiple independent models.
- At test time average their results.

While ensembling is commonly associated with traditional machine learning methods like decision trees and support vector machines, it is also a powerful technique in deep learning.

#### Normalizing the inputs

 If you normalize your inputs this will speed up the training process a lot. Substract the mean of the training set from each input and divide by the variance of the train set.

These steps should be applied to training, dev, and testing sets (but using mean and variance of the train set).

 If we don't normalize the inputs our cost function will be deep and its shape will be inconsistent (elongated) then optimizing it will take a long time.

![[Pasted image 20240725190901.png]]

Normalized inputs allow for a larger learning rate and optimization will thus be faster.

#### Vanishing / Exploding gradients

The Vanishing / Exploding gradients occurs when your derivatives become very small or very big, especially in very deep networks. It will take a long time for gradient descent to learn anything, if able to learn at all.

Definition:
* In neural networks, the learning process involves adjusting the weights using gradient descent, which relies on the gradients (partial derivatives) of the loss function with respect to the weights.
* When gradients are very small, the updates to the weights become minuscule. This means that the weights change very slowly, and the network learns at a very slow pace, potentially getting stuck and not learning effectively.
* When gradients are very large, the updates to the weights become very large as well. This can cause the weights to grow uncontrollably, leading to numerical instability and making the network unable to converge.

Causes:
* Functions like the sigmoid or tanh compress a wide range of input values into a small range (0 to 1 for sigmoid, -1 to 1 for tanh). When many layers use these functions, the gradients get multiplied several times, causing them to shrink exponentially and approach zero.
* Improper initialization can exacerbate the vanishing gradient problem. If initial weights are too small, the gradients can vanish even faster. If weights are too big, gradients can explode.
* Deep networks. The more layers there are, the more times the gradients are multiplied, increasing the likelihood of them becoming extremely small.


Solutions:

**(1) Smart initialization of weights**. Xavier (Glorot) set the initial gradients so that they are in a reasonable range.

Xavier initialization achieves this balance by setting the initial weights according to the size of the previous layer. Specifically, it uses a distribution with a variance that is inversely proportional to the number of input neurons in the layer.

![[Pasted image 20240725191655.png]]

Pytorch uses a similar approach by default for linear and convolutional layers.

![[Pasted image 20240725191803.png]]

**Others:**
- Batch normalization
- Gradient clipping
- Residual networks
- Auxiliary heads
![[Pasted image 20240725191925.png]]
*Skip connections*

![[Pasted image 20240725191944.png]]
*Auxiliary heads*

#### Gradient checking

This seems only relevant if you're implementing your own architecture and optimization code. Not the most common situation of a ML practitioner.

It involves approximating the derivatives of the weights of a specific sample, and checking that your algorithm is calculating them correctly.

### Week 2 - Optimization algorithms

#### Mini-batch gradient descent

Instead of calculating the gradient over the entire dataset (which could be HUGE and potentially intractable), we can calculate the gradient over a small sample of the training dataset to approximate the gradient, and make more frequent updates to the weights.

* **Batch gradient descent**: Calculate the gradient over the entire dataset (what we have done so far)
* **Mini-batch gradient descent**: Run gradient descent on mini batches. Works much faster in large datasets.

![[Pasted image 20240727142119.png]]
*MBGD has ups and downs, but the updates are much more frequent.*

Definitions:
* **Iteration**: Each process of calculating the gradients and updating the weights. In MBGD, it means going through a mini batch and updating the weiths.
* **Epoch**: Going over each sample in the entire training dataset. In MBGD it'd involve many iterations.

The mini batches could be either:
* Randomly sampled with replacement on each iteration. These samples can overlap with elements from previous iterations because sampling is done with replacement (Stochastic Mini-Batch Gradient Descent). SGD often refers to having batch size just 1.
* Calculated by slitting the dataset into non-overlapping mini-batches (Deterministic Mini-Batch Gradient Descent)

| Method                      | batch size     | Pros                                                                      | cons                                                                                            |
| --------------------------- | -------------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| batch gradient descent      | m              | accurate steps                                                            | takes too link per iteration, maybe intractabe                                                  |
| mini batch gradient descent | `>=1` ; `<= m` | faster learning bc you have vectorization and more regular weight updates | may not converge, requires tuning LR properly - need to tune mini batch size hyperparameter     |
| Stochastic gradient descent | usually 1      |                                                                           | very noisy gradient requiring small learning rate ; may diverge ; loses vectorization advantage |
mini bach size:
* `m<2000` => use batch gradient descent
* `m<2000` => use a power of 2 as batch size e.g. 64, depending on your GPU memory

#### Exponentially weighted averages

This is a mathematical concept, forget about NN now.  

Exponential weighted averages (EWAs), also known as exponential moving averages (EMAs), are used to smooth out data series to highlight trends by giving more weight to recent observations while exponentially decreasing the weights of older observations.

The idea behind exponential weighted averages is to apply weights that decay exponentially. More recent data points have higher weights, and the weights decrease exponentially as the data points get older.

The exponential average `v(t)` at time `t` of series `theta` is given by:

```
    v(t) = beta * v_{t-1} + (1-beta) * theta_t 
```

`beta` is the smoothing factor, and `(1-beta)` controls the rate at which the influence of past observations decays.

![[Pasted image 20240727143945.png]]
*Example of a time series and its EWAs*

The average starts at `v(0)=0` which gives a bias and shifts the average at the beginning, making it inaccurate. We can correct this bias by using the following equation:

```
    v(t) = (beta * v(t-1) + (1-beta) * theta(t)) / (1 - beta^t)
```

As t becomes larger the `(1 - beta^t)` becomes `1`

#### Momentum

Let's now use this mathematical tool in mini batch gradient descent. We can smooth out the gradients obtained by mini-batch gradient descent using a EWA, reducing oscillations in gradient descent and making it faster by following a smoother path towards minima.

Best `beta` hyperparameter average for our use case is between 0.9 and 0.98 which will average between 10 and 50 last entries. 

The update rule then becomes:

```
vdW = 0, vdb = 0
on iteration t:

	compute dw, db on current mini-batch                
			
	vdW = beta * vdW + (1 - beta) * dW
	vdb = beta * vdb + (1 - beta) * db
	
	W = W - learning_rate * vdW
	b = b - learning_rate * vdb
```

So instead of updating via the last weight, we use the EWA, which is smoother and pushes towards keeping the tendency of previous updates.
### RMSprop

Root Mean Square Prop, is another technique to speed up gradient descent. It keeps a rolling average of the square gradients:

![[Pasted image 20240727150150.png]]

This square average is used to normalize the learning rate on each weight update:
![[Pasted image 20240727150354.png]]
In pseudocode:

```
sdW = 0, sdb = 0
on iteration t:
	# can be mini-batch or batch gradient descent
	compute dw, db on current mini-batch
	
	sdW = (beta * sdW) + (1 - beta) * dW^2  # squaring is element-wise
	sdb = (beta * sdb) + (1 - beta) * db^2  # squaring is element-wise
	W = W - learning_rate * dW / sqrt(sdW)
	b = B - learning_rate * db / sqrt(sdb)
```

The benefit is that the learning rate becomes **adaptive**, adapting it to each **parameter** update individually based on gradient history. 

Let's break this down, because it can be confusing. The goal is to normalize the gradients of **each parameter** to ensure stable and consistent updates, so we are multiplying each gradient component by a custom factor that achieves this. The factor itself is  `1/ [ sqrt( E(g^2)_t ) ^ 2 ]` . This can be puzzling. 

We keep an average of the square of the weights, which captures the magnitude of the gradients without being affected by their sign.  This captures the average magnitude of (each component of) the gradient over time, regardless of direction (positive or negative).

The parameter update is scaled by the learning rate divided by the **square root** of this running average.
* The square root is meant to cancel out the squares in the average, providing an estimate of the absolute value of each parameter.
* This normalization ( `1/ [ sqrt( E(g^2)_t ) ^ 2 ]`) makes parameters with consistently **high** gradients to be **scaled down**, while lower gradients are **scaled up**. 
* Scaling down big gradients and scaling up small gradients is good because:
	* Large gradients tend to **overshoot** the optimal value, leading to instability, especially in high dimensions.
	* **Controlled updates** make steps towards the minimum more stable, preventing bouncing.
	* **RMSprop** scales the leraning rate so that it's based on the *history* of the gradients.  As optimization goes on, gradients should become smaller because you have learnt a lot. RMSProp will then help the updates not to be excessively small, allowing continued progress towards convergence. Also if you had a single iteration with huge magnitude for a weight that historically has had small magnitude, we don't let this noise nudge the parameters excessively.
	* **Gradient explosion** is mitigated by scaling down very large gradients, maintaining stability
	* Large updates can lead to overfitting, which are tempered by RMSProp.

#### Adam optimization algorithm

Adaptative Moment Estimation (Adam) works very well in NN, and it's just the combination of momentum and RMSProp together.

```
vdW = 0, vdW = 0
sdW = 0, sdb = 0
on iteration t:
	# can be mini-batch or batch gradient descent
	compute dw, db on current mini-batch                
			
	vdW = (beta1 * vdW) + (1 - beta1) * dW     # momentum
	vdb = (beta1 * vdb) + (1 - beta1) * db     # momentum
			
	sdW = (beta2 * sdW) + (1 - beta2) * dW^2   # RMSprop
	sdb = (beta2 * sdb) + (1 - beta2) * db^2   # RMSprop
			
	vdW = vdW / (1 - beta1^t)      # fixing bias
	vdb = vdb / (1 - beta1^t)      # fixing bias
			
	sdW = sdW / (1 - beta2^t)      # fixing bias
	sdb = sdb / (1 - beta2^t)      # fixing bias
					
	W = W - learning_rate * vdW / (sqrt(sdW) + epsilon)
	b = B - learning_rate * vdb / (sqrt(sdb) + epsilon)
```


The hyperparameters of adam are:
* The learning rate
* `beta1` : The momentum parameter, usually set to 0.9
* `beta2` : The RMSprop parameter, usually set to 0.999
* `epsilon`: Usually set to `10^-8`

#### Learning rate decay

During optimization, often we make the learning rate decay with iterations rather than keep it fixed. This helps make steps (and oscillations) near the optimum smaller, fostering convergence.

The learning rate can be made to decay following many policies, such as:

`learning_rate = learning_rate_0 / (1 + decay_rate * epoch_num)`

where `decay_rate` is a hyperparameter

- In the early stages of training, larger learning rates help in exploring the parameter space quickly.
- As training progresses, smaller learning rates help in refining the parameter values, leading to more precise convergence.

#### Local optima in deep learning

In high-dimensional spaces, getting stuck in a bad local optimum is rare. Instead, you're more likely to encounter saddle points, which aren't problematic. For a point to be a local optimum, it must be optimal in every dimension, which is highly unlikely.

A plateau is a region where the gradient is near zero for an extended period. Techniques like momentum or RMSprop help overcome these regions by adapting the learning process and making progress despite the flat gradients.

### Week 3: Hyperparameter Tuning, Batch Normalization and Programming Frameworks
#### Tuning hyperparameters

Andrew Ng says this is the "importance" of each hyperparameter when it comes to tuning:
1. Learning rate.
2. Momentum beta.
3. Mini-batch size.
4. No. of hidden units.
5. No. of layers.
6. Learning rate decay.
7. Regularization lambda.
8. Activation functions.
9. Adam `beta1`, `beta2` & `epsilon`.

Usually we try random values, we cannot use a grid of values because it's intractable.

It's beneficial to search parameters in an appropriate scale, if the range of search is `[a,b]` use a logarithmic scale.

e.g. a parameters goes from 0.0000001 to 1 and we want to try out 5 hyperparameter choices, here are the resulting values:

Linear (not recommended) are equally spaced:

`[0.0000001, 0.2500001, 0.5000001, 0.7500001, 1]`

Logarithmic scale:

 `[0.0000001, 0.001, 0.01, 0.1, 1]`
 
In the logarithmic scale, the values are spaced exponentially, providing a more balanced exploration across a wide range.

#### Hyperparameter tuning in practice

There are two approaches:

- **Panda Approach:** Many inexpensive trials to broadly explore the hyperparameter space.
- **Caviar Approach:** Fewer, more expensive trials to deeply explore and optimize within a smaller region of the hyperparameter space.

#### Normalizing activations

**Batch normalization** is about normalizing the activations `A[l]` (or simetimes the values before the activation, `Z[L]`). In practice ,`Z[l]` is done much more often.

Given the pre-activations for a batch of `m` training samples:

`Z[l] = [z(1), ..., z(m)]`

We compute the mean and variance across the training samples:

```
mean = 1/m * sum(z[i])
variance = 1/m * sum((z[i] - mean)^2)
```

Then normalize the data before applying the activation, driving the inputs to a zero mean, variance one distribution.

```
Z_norm[i] = (z[i] - mean) / np.sqrt(variance + epsilon)
```

Additionally, two learnable parameters (not hyperparameters) `gamma` and `beta` transform the inputs to a different distribution:

`Z_batch_norm[i] = gamma * Z_norm[i] + beta`

The following schema summarizes the transformations:
![[Pasted image 20240727202324.png]]

WHY is batch normalization useful?
* The distribution of inputs to each layer can change as the parameters of the previous layers change. This is called internal covariate shift. Batch normalization ensures that the inputs to each layer are stable, within the same range. This **stabilizes training** and keeps **gradients stable and consistent**.
* This stability allows for larger learning rates, resulting in faster optimization.

My intuition:

*Without batch normalization, the inputs to a given layer can vary widely across iterations. For example, in one iteration, the inputs to layer L could be in the range `[0,100]`, and in another iteration, they might be in the range `[−1,0]`. This variation can occur because the parameters of the preceding layers are continuously updated during training, changing the distribution of their outputs (which are the inputs to layer L).*

*These irregular and fluctuating input ranges make the optimization process more challenging. The model has to constantly adapt to the changing distribution of inputs, which can slow down learning and make it harder to converge to an optimal solution. It's akin to chasing a moving target during optimization. The parameters of the network need to adjust not only to learn the underlying patterns in the data but also to accommodate the changing scales of inputs at each layer.*

WHY do we add the step with beta and alpha instead of just normalizing to 0-1 and leave it there? Wouldn't that ensure already a stability in the inputs to the next layer?

*Without `\gamma` and `\beta`, normalization would constrain the outputs of each layer to a fixed distribution (zero mean and unit variance). This can limit the network's capacity to represent complex functions. By allowing the network to scale and shift the normalized outputs, `\gamma` and `\beta` provide flexibility. The network can learn the optimal mean and variance for each layer's activations during training.*

*If the optimal is indeed to keep the zero mean one variance representation, it will learn `alpha=1` and `beta=0` (identity transformation)*

*Example: Neural Network with and without gamma and beta*

*Imagine a simple neural network trying to learn a function where the desired output for a certain layer's neurons should be in a specific range, say `[0, 10]`.*

*Without gamma and beta, after normalization, the output of each neuron will have mean 0 and variance 1. Thus, the outputs will be centered around 0 and most values falling within `[-1,1]`. The outputs are constrained to be in this tight range,this constraint would make it very difficult for the subsequent layers to adjust and learn the correct transformation towards `[0,10]`. A  ReLu for example will push all negative values towards 0, losing half of the output range.*

*With `gamma` and `beta`, we can learn `gamma=5` `beta=5` and the range of the values is not adjusted to `[0,10]`, aligning better with the desired output distribution.*

What size does the mean, variance, gamma and beta have?

*Say we are in layer `l`  and we have batch size `m`.*

*`Z[l] = [z(1), ..., z(m)]`*

*If layer `l` has `k` neurons, then each `z_i` has shape `(k,1)`*

*The mean and variance are calculated component-wise, so they have the same shape:  `(k,1)`.*

*The scaling and shifting parameters `gamma` and `beta` are also specific for each component, so they are each of shape `(k,1)`*

*TL; DR: Everything is done component-wise.*

At test time, batch norm layer uses a mean and variance computed during train; likely a weighted average across mini-batches.

#### Softmax Regression

So far, we've only been dealing with binary classification. This means that the last layer has been a sigmoid and the loss has been binary cross entropy loss. 

NN are, of course, much more flexible. We can use different types of layers in the last layer in order to return the appropriate data format, coupled with appropriate losses.

| **Task**                                                | **Last Layer Activation**      | **Loss Function**                | **Description**                                                                          |
| ------------------------------------------------------- | ------------------------------ | -------------------------------- | ---------------------------------------------------------------------------------------- |
| **Binary Classification**                               | Sigmoid                        | Binary Cross Entropy             | Single neuron outputting probability for two classes.                                    |
| **Multi-Class Classification (one-hot encoded labels)** | Softmax                        | Categorical Cross Entropy        | Output layer size equal to number of classes, outputs probabilities.                     |
| **Multi-Class Classification (sparse labels)**          | Softmax                        | Sparse Categorical Cross Entropy | Output layer size equal to number of classes, outputs probabilities, labels as integers. |
| **Multi-Label Classification**                          | Sigmoid                        | Binary Cross Entropy             | Multiple neurons, each outputting probability for independent classes.                   |
| **Regression (one target)**                             | None/Linear                    | Mean Squared Error (MSE)         | Single neuron outputting a continuous value.                                             |
| **Regression (multiple targets)**                       | None/Linear                    | Mean Squared Error (MSE)         | Multiple neurons, each outputting a continuous value for different targets.              |

The **Softmax** is used for multi class classification. If we have four classes for example, we may have four neurons in the last layer and encode the classes as `[0 0 0 1], [0 0 1 0], [0 1 0 0], [1 0 0 0]`.

If C is the number of classes, each of the C values in the output layer will contain a probability of the example belonging to each class. The last layer will have a Softmax activation instead of sigmoid, which does the following:

![[Pasted image 20240727230405.png]]

Example:

1. `z=[2.0,1.0,0.1]`
2. `e^z = [7.38 , 2.71, 1.10]` ; `sum(e^z) = 12.21`
3. `softmax(z) = [0.65, 0.24, 0.09]`

So we basically turned a vector of pre-activations into a vector of probabilities that add up to 1.

The categorical loss function now is:

![[Pasted image 20240727230716.png]]
Notice that `\hat(y_i)` is going to be in `[0,1]`; thus its logarithm will be between `[-inf,0]`.

For each sample, only the correct class term `y_k * log(yhat_k)` is going to add to the loss, the rest of the `y_i` will be zero. If the model is predicting a low probability to the correct class, the log is going to be a large negative number, and the `-1` multiplier will make it be a large loss. Vice versa.

![[Pasted image 20240727230959.png]]

Using the new loss, we need to readjust a few derivatives:

`dz[L] = Y_hat - Y`

The derivative of softmax is:

`Y_hat * (1 - Y_hat)`

#### Existing frameworks

Deep learning is not about implementing everything from scratch, but about reusing leading existing frameworks.

| Framework     | Programming Language     | Focus                                               | Deployment          | Key Features                                        | Use Cases/Tasks                                                                             |
| ------------- | ------------------------ | --------------------------------------------------- | ------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Caffe/Caffe2  | C++                      | Image/video processing                              | Production          | Modular architecture, speed, image/video focus      | Image classification, object detection, image segmentation                                  |
| CNTK          | C++/Python               | Speech recognition, image recognition, NLP          | Production          | Computational network toolkit, distributed training | Speech recognition, image recognition, NLP, recommendation systems                          |
| DL4j          | Java                     | JVM ecosystem, distributed computing                | Production          | JVM integration, distributed computing              | Natural language processing, time series analysis, anomaly detection                        |
| Keras         | Python                   | Rapid prototyping, high-level API                   | Production          | High-level API, easy to use, modularity             | Image recognition, text generation, neural style transfer                                   |
| Lasagne       | Python                   | Theano-based, modular                               | Research            | Theano-based, modular, flexibility                  | Research, prototyping, building custom architectures                                        |
| MXNet         | C++/Python/R/Julia/Scala | Scalability, flexibility, speed                     | Production          | Scalability, hybrid programming, speed              | Image classification, object detection, natural language processing, recommendation systems |
| PaddlePaddle  | C++/Python               | Chinese-dominant, industrial applications           | Production          | Industrial applications, speed, scalability         | Computer vision, natural language processing, recommendation systems, ad optimization       |
| TensorFlow    | Python/C++               | Large-scale machine learning, distributed computing | Production          | Flexibility, scalability, community support         | Image recognition, natural language processing, speech recognition, time series analysis    |
| Theano        | Python                   | Deep learning research, optimization                | Research            | Symbolic computation, GPU optimization              | Research, prototyping, low-level optimizations                                              |
| Torch/PyTorch | Lua/Python               | Flexibility, speed, dynamic computation graph       | Research/Production | Dynamic computation graph, speed, flexibility       | Computer vision, natural language processing, reinforcement learning                        |

#### TensorFlow

The course develops a few concepts of TensorFlow. Here is a sample snippet doing optimization, checkout the programming assignment for more coding examples:

```python
import numpy as np
import tensorflow as tf


coefficients = np.array([[1.], [-10.], [25.]])

x = tf.placeholder(tf.float32, [3, 1])
w = tf.Variable(0, dtype=tf.float32)                 # Creating a variable w
cost = x[0][0]*w**2 + x[1][0]*w + x[2][0]

train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
session.run(w)    # Runs the definition of w, if you print this it will print zero
session.run(train, feed_dict={x: coefficients})

print("W after one iteration:", session.run(w))

for i in range(1000):
	session.run(train, feed_dict={x: coefficients})

print("W after 1000 iterations:", session.run(w))
```

* The backward pass is automatically done, you only specify the forward pass.
* A placeholder is a variable that may be assigned a value.