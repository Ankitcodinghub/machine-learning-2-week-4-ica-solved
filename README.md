# machine-learning-2-week-4-ica-solved
**TO GET THIS SOLUTION VISIT:** [Machine Learning 2 Week 4-ICA Solved](https://www.ankitcodinghub.com/product/machine-learning-2-week-4-ica-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;98828&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;0&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;0&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;0\/5 - (0 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;Machine Learning 2 Week 4-ICA Solved&quot;,&quot;width&quot;:&quot;0&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 0px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            <span class="kksr-muted">Rate this product</span>
    </div>
    </div>
<div class="page" title="Page 1">
<div class="layoutArea">
<div class="column"></div>
<div class="column">
&nbsp;

Exercise Sheet 4

</div>
</div>
<div class="layoutArea">
<div class="column">
Exercise 1: Sparse Coding (20 + 20 P)

Let x1, . . . , xN ∈ Rd be a dataset of N examples. Let si ∈ Rh be the source associated to example xi, and W be a matrix of size d × h that linearly reconstructs the examples from the sources. We wish to minimize the objective:

</div>
</div>
<div class="layoutArea">
<div class="column">
1 􏰄N

J = N

i=1

</div>
<div class="column">
1 􏰄N

∥ x i − W s i ∥ 2 + λ · N ∥ s i ∥ 1 + η · ∥ W ∥ 2F

</div>
</div>
<div class="layoutArea">
<div class="column">
i=1

􏰑 􏰐􏰏 􏰒 􏰑 􏰐􏰏 􏰒 􏰑 􏰐􏰏 􏰒

</div>
</div>
<div class="layoutArea">
<div class="column">
reconstruction sparsity regularization

</div>
</div>
<div class="layoutArea">
<div class="column">
with respect to the

reconstruction term is the standard mean square error, the sparsity term consists of a standard L1 penalty on the sources, and the last regularization term prevents the sparsity term from becoming ineffective.

(a) Show that for fixed sources, the optimal matrix W is given in closed form as: W =ΣXS􏰍ΣSS +ηI􏰎−1

</div>
</div>
<div class="layoutArea">
<div class="column">
where

</div>
<div class="column">
1 􏰄N

ΣXS =N

i=1

</div>
<div class="column">
1 􏰄N

ΣSS =N sis⊤i .

i=1

</div>
</div>
<div class="layoutArea">
<div class="column">
weights W and the sources s1, . . . , sN . The objective consists of three

</div>
<div class="column">
terms: The

</div>
</div>
<div class="layoutArea">
<div class="column">
xis⊤i

</div>
<div class="column">
and

</div>
</div>
<div class="layoutArea">
<div class="column">
(b) We now consider the optimization of sources. Due to the 1-norm in the sparsity term, we cannot find a closed form solution. However, we consider a local relaxation of the optimization problem where the 1-norm of the sparsity term is linearized as

∥ s i ∥ 1 = q ⊤i s i

with qi ∈ {−1, 0, 1}d a constant vector. This relaxation makes the objective function quadratic with si.

Show that under this local relaxation, the solution of the optimization problem is given in closed form as: si = 􏰍W⊤W􏰎−1􏰍W⊤xi − λ · qi/2􏰎

Although this solution is not the true minimum of J (e.g. it is not sparse), it can serve as the end-point of some line-search method for finding good source vectors si.

Exercise 2: Auto-Encoders (20 P)

In this exercise, we would like to show an equivalence between linear autoencoders with tied weights (same parameters for the encoder and decoder) and PCA. We consider the special case of an autoencoder with a single hidden unit. In that case, the autoencoder consists of the two layers:

si = w⊤xi (encoder) xˆi = w · si (decoder)

where w ∈ Rd. We consider a dataset x1, . . . , xN assumed to be centered (i.e. 􏰃i xi = 0), and we define the objective that we would like to mininize to be the mean square error between the data and the reconstruction:

1 􏰄N

J ( w ) = N

Furthermore, to make the objective closer to PCA, we can always rewrite the weight vector as w = αu where u is a unit vector (of norm 1) and α is some positive scalar, and search instead for the optimal parameters u and α.

</div>
</div>
<div class="layoutArea">
<div class="column">
i=1

</div>
</div>
<div class="layoutArea">
<div class="column">
∥ x i − xˆ i ∥ 2

</div>
</div>
</div>
<div class="page" title="Page 2">
<div class="layoutArea">
<div class="column">
(a) Show that the optimization problem can be equally rewritten as

arg min J(w) = arg max u⊤Su

α,u α,u where S = N1 􏰃Ni=1 xix⊤i , which is a common formulation of PCA.

Exercise 3: Programming (40 P)

Download the programming files on ISIS and follow the instructions.

</div>
</div>
</div>
<div class="page" title="Page 3">
<div class="section">
<div class="section">
<div class="layoutArea">
<div class="column">
Exercise sheet 4 (programming) [SoSe 2021] Machine Learning 2

</div>
</div>
<div class="layoutArea">
<div class="column">
Implementing an Autoencoder

In this exercise, we would like to train on a popular face dataset, a sparse auto-encoder. We consider the simple two-layer autoencoder network:

zi = max(0, V xi + b) (layer 1) x^i = Wzi + a (layer 2)

where W , V are matrices of parameters of the encoder and the decoder, and b, a are additional bias parameters. We seek to maximize the objective:

</div>
</div>
<div class="layoutArea">
<div class="column">
1N 1N h1N−1 min ∑∥xi −x^i∥2 +λ⋅ ∑∥zi∥1 +ε⋅∑( ∑[zi]j)

</div>
<div class="column">
+η⋅ ∥W∥2 F

</div>
</div>
<div class="layoutArea">
<div class="column">
i=1 i=1  

</div>
</div>
<div class="layoutArea">
<div class="column">
WNNN

</div>
<div class="column">


</div>
</div>
<div class="layoutArea">
<div class="column">
reconstruction sparsity

</div>
<div class="column">
i=1 “entropy”

</div>
<div class="column">
regularization

</div>
</div>
<div class="layoutArea">
<div class="column">
The objective is composed of four terms: The reconstruction term is the standard mean square error between the data points and their reconstructions. The sparsity term applies a l1-norm to drive activation to zero in the representation. The “entropy” term that ensures that at least a few examples activate each source dimension. The regularization term ensures that the sparsity term remains effective.

Loading the dataset

We first load the Labeled Faces in the Wild (LFW) dataset. The LFW is a popular face recognition dataset which is readily available from scikit learn. When loading the dataset, we specify some image downscaling in order to limit the computation resources. The following code visualizes a few images that we have extracted from the LFW dataset.

</div>
</div>
<div class="layoutArea">
<div class="column">
j=1 

</div>
</div>
<div class="layoutArea">
<div class="column">
In [1]:

</div>
<div class="column">
import sklearn,sklearn.datasets import matplotlib

%matplotlib inline

from matplotlib import pyplot as plt

<pre>data = sklearn.datasets.fetch_lfw_people(resize=0.3)['images']
</pre>
<pre>plt.figure(figsize=(12,2.5))
plt.imshow(data[:32].reshape(2,16,37,28).transpose(0,2,1,3).reshape(2*37,16*28),cmap=
'gray')
plt.show()
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
Implementing the autoencoder (20 P)

We now would like to train an autoencoder on this data. As a first step, we standardize the data, which is a usual step before training a ML model. (Note that contrarily to other component analyses such as ICA, the data does not need to be whitened.)

In [2]: X = data.reshape(len(data),-1) X = X – X.mean(axis=0)

</div>
</div>
<div class="layoutArea">
<div class="column">
X = X / X.std()

</div>
</div>
</div>
</div>
</div>
<div class="page" title="Page 4">
<div class="section">
<div class="layoutArea">
<div class="column">
To learn the autoencoder, we need to optimize the objective function above. This can be done using by gradient descent, or some enhanced gradient-based optimizer such as Adam. Because a manual computation of the gradients can be difficult and error-prone, we will make use of automatic differentiation readily provided by the PyTorch software. PyTorch uses its own structures for storing the data and the model parameters. (You can consult the tutorials at https://pytorch.org/tutorials/ (https://pytorch.org/tutorials/) to learn the basics.)

We first convert the data into a PyTorch tensor.

In [3]: import torch

X = torch.FloatTensor(X)

Recall that the four terms that compose the objective function are given by:

</div>
</div>
<div class="layoutArea">
<div class="column">
rec= N ∑∥xi −x^i∥2 i=1

h1N −1 ent = ∑ ( ∑[zi]j)

Task:

Create the function get_objective_terms that computes these terms.

The function receives as input:

</div>
</div>
<div class="layoutArea">
<div class="column">
A FloatTensor A FloatTensor A FloatTensor A FloatTensor A FloatTensor

</div>
<div class="column">
X of size m × d containing a data minibatch of m examples. V of size d × h containing the weights of the encoder.

W of size h × d containing the weights of the decoder.

b of size h containing the bias of the encoder.

a of size d containing the bias of the decoder.

</div>
</div>
<div class="layoutArea">
<div class="column">
1N 1N

</div>
</div>
<div class="layoutArea">
<div class="column">
j=1 Ni=1

</div>
<div class="column">
spa=(N ∑∥zi∥1) i=1

reg = ∥W∥2 F

</div>
</div>
<div class="layoutArea">
<div class="column">
In your function, the parameter ε can be hardcoded to 0.01. The function should return the four terms ( rec , spa , ent , reg ) of the objective. (These terms will be merged later on in a single objective function.) While implementing the get_objective_terms function, make sure to use PyTorch functions so that the gradient information necessary for automatic differentiation is retained. For example, converting arrays to numpy will not work as this will remove the gradient information.

In [4]: def get_objective_terms(X,V,W,b,a):

# —————————————————– # TODO: replace by your code

# —————————————————– import solution

<pre>                  rec,spa,ent,reg = solution.get_objective_terms(X,V,W,b,a)
</pre>
<pre>                  # -----------------------------------------------------
</pre>
return rec,spa,ent,reg

Training the autoencoder

Now that the terms of the objective function have been implemented, the model can be trained to minimize the objective. The code belowcallsthefunction get_objective_terms repeatedly(onceperiteration).Automaticdifferentiationisusedtocomputethe gradient, and we use Adam (a state-of-the-art optimizer for neural networks) to optimize the parameters. The number of units in the representation is hard-coded to h = 400, and we use the parameter η = 1 for the regularizer.

</div>
</div>
</div>
</div>
<div class="page" title="Page 5">
<div class="section">
<div class="layoutArea">
<div class="column">
In [5]:

</div>
<div class="column">
<pre>import torch.optim
import torch.nn
import numpy
</pre>
def train(X,lambd=0): d = X.shape[1]

h = 400

eps = 0.01 * lambd # hard-coded parameter

eta = 1 * lambd # hard-coded parameter

<pre>    V = torch.nn.Parameter(d**-.5*torch.randn([d,h]))
    W = torch.nn.Parameter(torch.zeros([h,d]))
    b = torch.nn.Parameter(torch.zeros([h]))
    a = torch.nn.Parameter(torch.zeros([d]))
</pre>
optimizer = torch.optim.Adam((V,W,b,a), lr=0.0001)

print(‘%7s %8s %8s %8s %8s’%(‘nbit’,’rec’,’spa’,’ent’,’reg’)) for i in range(0,10001):

<pre>        optimizer.zero_grad()
</pre>
<pre>        x= X[numpy.random.permutation(len(X))[:100]]
</pre>
<pre>        rec,spa,ent,reg = get_objective_terms(x,V,W,b,a)
</pre>
<pre>        (rec + lambd*spa + eps*ent + eta*reg).backward()
</pre>
if i%1000 == 0: print(‘%7d %8.2f %8.2f %8.2f %8.2f’%(i,rec.data,spa.data,ent. data,reg.data))

optimizer.step() return V,W,b,a

</div>
</div>
<div class="layoutArea">
<div class="column">
Dense Autoencoder

We first train an autoencoder with parameter λ = 0, that is, a standard autoencoder without sparsity. The parameters of the learn autoencoder are stored in the variables V , W , b , a . Running the code may take a few minutes. You may temporarily reduce the number of iterations when testing your implementation.

</div>
</div>
<div class="layoutArea">
<div class="column">
In [7]: V1,W1,b1,a1 = train(X,lambd=0)

</div>
</div>
<div class="layoutArea">
<div class="column">
<pre> nbit      rec
    0  1045.22
 1000   128.94
 2000    86.48
 3000    63.23
 4000    62.44
 5000    54.00
 6000    54.19
 7000    46.84
 8000    40.91
 9000    36.63
10000    30.43
</pre>
</div>
<div class="column">
<pre>   spa      ent
153.52  1099.49
458.57   356.76
456.00   366.91
438.00   377.32
450.24   381.65
450.36   370.90
466.40   362.62
436.62   389.91
441.50   391.96
423.16   407.59
425.86   407.38
</pre>
</div>
<div class="column">
<pre>   reg
  0.00
 48.74
 75.44
 94.49
110.32
124.42
137.63
150.74
164.41
179.50
196.64
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
We observe that the reconstruction term decreases strongly, indicating that the autoencoder becomes increasingly better at reconstructing the data. The sparsity term, however, increases, indicating that the standard autoencoder does not learn a sparse representation.

Sparse Autoencoder

We now would like to train a sparse autoencoder. For this, we set the sparsity parameter to λ = 1 and re-run the training procedure. We store the learned parameters in another set of variables.

</div>
</div>
</div>
</div>
<div class="page" title="Page 6">
<div class="layoutArea">
<div class="column">
In [8]: V2,W2,b2,a2 = train(X,lambd=1)

</div>
</div>
<div class="layoutArea">
<div class="column">
<pre> nbit      rec
    0  1086.80
 1000   176.41
 2000   155.85
 3000   146.50
 4000   142.15
 5000   131.36
 6000   136.57
 7000   136.83
 8000   125.52
 9000   132.11
10000   127.77
</pre>
</div>
<div class="column">
<pre>   spa      ent
157.35  1077.17
170.94  1313.81
159.27  1674.05
154.00  1950.38
155.02  2055.47
156.07  2003.10
157.66  1966.48
154.05  2090.28
147.72  2086.20
159.23  1948.78
156.40  2025.83
</pre>
</div>
<div class="column">
<pre>  reg
 0.00
79.80
76.04
72.20
70.96
70.35
70.26
69.95
69.97
69.84
69.77
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
We observe that setting the parameter λ to a non-zero keeps the sparsity term low, which indicates that a sparser representation has been learned. In turn, we also loose a bit of reconstruction accuracy compared to the original autoencoder. This can be expected since the sparsity imposes additional constraints on the solution.

Analyzing autoencoder sparsity (10 P)

As a first analysis, we would like to verify how truly sparse the representation we have learned is.

Task:

Create a line plot where the two lines represents all activations (for the 25 first examples in the dataset) sorted from largest to smallest of the respective autoencoder models.

</div>
</div>
<div class="layoutArea">
<div class="column">
In [9]:

</div>
<div class="column">
# —————————————————– # TODO: replace by your code

# —————————————————– import solution solution.plot_sparsity(X[:25],V1,V2,b1,b2)

<pre># -----------------------------------------------------
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
We observe that the sparse autoencoder has a much larger proportion of weights that are close to zero. Hence, the our model has learned a sparse representation. One possible use of sparsity is to compress the data while retaining most of the information.

Inspecting the representation (10 P)

As a second analysis, we would like to visualize what the decoder has learned.

Task:

Write code that displays the first 64 decoding filters of the two models, in a similar mosaic format as it was used above to display some examples from the dataset.

</div>
</div>
</div>
<div class="page" title="Page 7">
<div class="layoutArea">
<div class="column">
In [10]:

</div>
<div class="column">
# —————————————————– # TODO: replace by your code

# —————————————————– import solution

<pre>solution.view_decoder(W1,W2)
</pre>
<pre># -----------------------------------------------------
</pre>
<pre>decoder weights for the standard autoencoder
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
<pre>decoder weights for the sparse autoencoder
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
We observe that the filters of the standard autoencoder are quite difficult to interpret, whereas the sparse autoencoder produces filters with a stronger focus a single facial or background features such as the mouth, the nose, the bottom left/right corners, or the overall lighting condition. The features of the sparse autoencoder are also more interpretable for a human.

</div>
</div>
</div>
<div class="page" title="Page 8"></div>
<div class="page" title="Page 9"></div>
