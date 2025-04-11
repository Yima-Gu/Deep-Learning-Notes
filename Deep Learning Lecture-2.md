### Brain and Neuron

感知机*Perceptron*是神经元的一个相当简单的数学模型，包括：输入、权重、激活函数、输出。其实是在空间超平面上嵌入了一个非线性函数。
$$
\hat{y} = g(\sum_{i=1}^{n}  x_i \theta_i +\theta_0)
$$
感知机在神经网络中也叫单元*unit*，是神经网络的基本组成单元。但是这样的单元会比人的神经元简单很多。
#### PLA
在上述的激活函数中，最先使用的是符号函数，这个函数是不光滑的。可以使用下面的算法来进行训练：
![[{5ABA24F6-F5A5-48C8-AE9F-1E8524E35979}.png]]
在理论推导中，可以计算PLA的**收敛率**：（在线性可分的情况下）
$\gamma$是最优间隔*the best-case margin*，计算的是训练样本与超平面的距离之间的最小值。
$$
\exists v \in \mathbb{R}^d \quad \text{s.t.}\,\gamma \leq \frac{y_i(v\cdot x_i)}{||v||}
$$
$R$是数据集的半径，即样本数据向量模的最大值，$d$是数据集的维度。那么PLA的收敛率为： 最多经过$\frac{R^2}{\gamma^2}$次迭代就可以收敛。
- $\gamma$越大，收敛越快
- $R$越大，收敛越慢

##### Expresiveness of Perceptron
感知机是一个线性分类器，只能解决线性可分的问题。如果数据不是线性可分的，那么感知机就无法解决（例如异或问题，但其实异或不是基本的布尔运算，可以用与或非表达）。

#### Multi-layer Perceptron
多层感知机*Multi-layer Perceptron*是感知机的扩展，可以解决非线性问题。多层感知机的结构是：输入层、隐藏层、输出层（输入层并不算一层）。
*感知机之间的链接方式相比人脑而言也是较为简单的。*
在表达时，可以发现是**稀疏的**，也就是每一层并不是与前面的所有的感知机相连。多层感知机的表达能力较强，这时需要增加感知机的层数。

##### Comention

![[{9A215F21-185B-4C55-B872-413D388E5321}.png]]

![[{E178BAD1-DE9C-44E5-A665-D550AE3A9558}.png]]

- 用上标表示层数，用下标表示感知机的编号
- $\theta_{ij}^{(l)}$表示第$l$层的第$i$个感知机的第$j$个输入的权重
- $b_j^{(l)}$表示第$l$层的第$j$个感知机的偏置
- $a_j^{(l)}$表示第$l$层的第$j$个感知机的输出（在激活之后的数值）
- $z_j^{(l)}$表示第$l$层的第$j$个感知机的输入（经过线性变换之后的数值）
- $J(\theta)$表示损失函数*Loss Function*

在上面的图中，边的个数就是参数的个数。

##### Activation Function

- **Sigmoid函数**：$g(z) = \sigma(z)= \frac{1}{1+e^{-z}}$
采用有界的函数，可以将输出限制在0-1之间，避免数值爆炸。但是在基于梯度的计算中，会出现梯度消失（梯度饱和），在两侧的范围内梯度会接近于0。

- **ReLU函数**：$g(z) = max(0,z)$
ReLU函数是一个分段函数，可以避免**梯度消失**的问题。但是在训练时，会出现**神经元死亡**的问题，即神经元的输出一直为0。

- **GeLu函数**：$g(z) = z \cdot \Phi(z) = z \cdot \frac{1}{2} (1 + \text{erf}(\frac{z}{\sqrt{2}}))$
用Guass分布的累计函数对上述进行加权。$\Phi(z)$是标准正态分布的累计分布函数*CDF*。在一些较为复杂的模型中（GPT-3、Bert）都有使用。

在网络的输出层，使用的激活函数由问题决定。如果是回归问题，可以使用线性函数；在有界的输出情况下，可以使用Sigmoid函数；在多分类问题中，可以使用Softmax函数。

- **Softmax函数**：$g(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{k} e^{z_j}}$
Softmax函数是一个多分类的激活函数，可以将输出的值转化为概率值。分类问题是随机实验中的伯努利实验*Categorical Distribution*。
缺点为：“赢者通吃”，即最大的值会被放大，其他的值会被压缩，有*over confidence*的问题（即某个分类的概率过大）。同时有数值稳定性问题，即数值计算时可能会出现数值爆炸的问题。
改进为：
$$
g(z)_i = \frac{e^{z_i - \max(z)}}{\sum_{j=1}^{k} e^{z_j - \max(z)}}
$$
上述改进能解决数值稳定性问题，但是对于*over confidence*问题还是存在。 ^b5bcbb

##### Cost Function

任何一个衡量预测与实际值之间的差异的函数都可以称为损失函数。在这里使用的是交叉熵损失函数*Cross Entropy Loss*：
$$
J(y,\hat{y}) = - \sum_{i=1}^{n} y_i \log(\hat{y}_i)
$$
作代入得到：
$$
\min J(\theta)= -\frac{1}{m} \sum_{i=1}^{m} \sum_{j=1}^{k} \mathbf{1}\{y^{(i)}=j\} \log\frac{\exp{ z_j^{(n_l)}}}{\sum_{j'=1}^{k} \exp{ z_{j'}^{(n_l)}}} \quad (*)
$$

*\*上述公式中的m为样本数目，k为类别数目，$z_{j}^{(n_l)}$为最后一层的第j个感知机的输入。对于实际类别采用独热编码，即只有在对应类别取值为1。*

##### Statistical View of Softmax

考虑投掷m次骰子，其中第$i$个得到$j$的概率为$q_{ij}$。在*Softmax*中对于概率进行建模（用数据进行估计，对于分类估计的参数进行逼近）：
$$
q_{ij} = P(y_i = j\,| \mathbf{x_i} ; \mathbf{W} )
$$
在给定的结果$\{y_1,...,y_m\}$下，概率值（似然函数）为：
$$
\mathcal{L}(\mathbf{W};\mathcal{D})=\prod_{i=1}^{m} \prod_{j=1}^{k} P(y_i=j|q_{ij})^{\mathbf{1}\{y_i = j\}} = \prod_{i=1}^{m} \prod_{j=1}^{k}P(y_i = j\,| \mathbf{x_i} ; \mathbf{W} ) ^{\mathbf{1}\{y_i = j\}}
$$
*$\mathbf{W}$是模型的参数，上面的式子是在这样的建模和数据下得到结果的可能性，也就是统计中的似然函数。这样的过程类似于统计中的参数估 计。*

做极大似然估计：
$$
\mathcal{L}(\mathbf{W};\mathcal{D}) =\max_{w_1 \dots w_k} \prod_{i=1}^{m} \prod_{j=1}^{k} P(y_i = j\,| \mathbf{x_i} ; \mathbf{W} ) ^{\mathbf{1}\{y_i = j\}}
$$
取负对数：
$$
J(\mathbf{W}) = \min_{w_1 \dots w_k}- \log \mathcal{L}(\mathbf{W};\mathcal{D}) = - \sum_{i=1}^{m} \sum_{j=1}^{k} \mathbf{1}\{y_i = j\} \log P(y_i = j\,| \mathbf{x_i} ; \mathbf{W} )
$$
上述的式子就是交叉熵损失函数。上面的过程其实是在认为分类是$i.i.d.$的伯努利分布的极大似然估计。

### Gradient Descent

对于不是直接依赖的导数的计算较为复杂，对于最后一层的导数计算较为简单（是直接依赖）。对于前面层的参数的导数在这里使用**链式法则**来进行计算。

对于最后一层的参数的导数计算：
$$
\frac{\partial J(\theta ,b)}{\partial z_j^{(n_l)}} = - (\mathbf{1}\{y^{(i)}=j\} -P(y^{(i)}=j|\mathbf{x}^{(i)};\theta,b))) 
$$
可以发现梯度是真是的概率减去预测的概率。

#### Step 1: Forward Propagation

输入样本计算得到的输出值，这个过程是一个前向传播的过程。

#### Step 2: Backward Propagation

将损失函数带有的错误信息向前传播
$$
\frac{J(\theta)}{\theta_1}= \frac{\partial J(\theta)}{\partial \hat{y}} \frac{\partial \hat{y}}{\partial z} \frac{\partial z}{\partial \theta_1}
$$
除了需要求解的导数的参数，其他的都是计算的中间值。BP是一个动态规划算法。

#### Computing the Residual

第$l$层的第$i$个结点的残差*Residual*的定义为：
$$
\delta_i^{(l)} = \frac{\partial J(\theta)}{\partial z_i^{(l)}}
$$
对于最后一层的残差，计算较为简单：
$$
\delta_i^{(n_l)} = \frac{\partial}{\partial z_i^{(n_l)} } J(\theta) = \frac{\partial }{\partial \hat{y}_i}J(\theta) g'(z_i^{(n_l)})
$$
利用链式法则对激活函数求导即可。
对于隐藏层的导数计算：
$$
\delta_i^{(l)} = \frac{\partial J(\theta)}{\partial z_i^{(l)}} = \sum_{j=1}^{n_{l+1}} \frac{\partial J(\theta)}{\partial z_j^{(l+1)}} \frac{\partial z_j^{(l+1)}}{\partial z_i^{(l)}} = \sum_{j=1}^{n_{l+1}} \delta_j^{(l+1)} \theta_{ij}^{(l)} g'(z_i^{(l)})
$$
$$
\delta_i^{(l)}= \sum_{j=1}^{n_{l+1}} \delta_j^{(l+1)} \theta_{ji}^{(l)} g'(z_j^{(l)})
$$
上述公式实现了**传递**的过程。


#### Step 3: Update Parameters
对于参数更新的过程：
$$
\frac{\partial J(\theta)}{\partial \theta_{ij}^{(l)}} = \frac{\partial J(\theta)}{\partial z_j^{(l+1)}} \frac{\partial z_j^{(l+1)}}{\partial \theta_{ij}^{(l)}} = \delta_j^{(l+1)} a_i^{(l)}
$$
$$
\frac{\partial J(\theta)}{\partial b_j^{(l)}} = \delta_j^{(l+1)}
$$
##### Automatic Differentiation
在实际的计算中，可以使用自动微分的方法来进行计算。自动微分是一种计算导数的方法，可以分为两种：
- **Symbolic Differentiation**：通过符号的方式来计算导数，这种方法计算的精确度较高，但是计算的速度较慢。
- **Numerical Differentiation**：通过数值的方式来计算导数，这种方法计算的速度较快，但是计算的精确度较低。

在计算图中，将每一个计算层的反向传播的导数保存在软件包中，这样可以减少计算的时间。实际的应用中，对于计算图进行拓扑排序，然后进行反向传播的计算。

#### Optimization in Practice

##### **Dropout**

在训练的过程中，随机的将一些神经元的权重置为0（丢弃），这样可以减少过拟合的问题。在操作的过程中，按照一定的概率$p$对神经元进行丢弃。在某一层未被丢弃的神经元的激活值值乘以$\frac{1}{1-p}$，这样可以保持期望值不变。

##### Weight Initialization

对于权重的初始化，一般使用Guass分布可以使用一些方法来进行初始化，例如：
**Xavier Initialization** ( linear activations )：
$$
Var(W)= \frac{1}{n_{in}}
$$
假设输入的数据$x_j$满足均值为0，方差为$\gamma$，$n_{in}$是这一个神经元对应的输入的神经元的个数。
在线性组合之后，可以得到：
$$
h_i=\sum_{j=1}^{n_{in}} w_{ij} x_j
$$
可以认为$w_{ij}$是独立同分布的并且均值为0方差为$\sigma^2$那么计算得到：
$$
\mathbb{E}[h_i]=0 \quad \mathbb{E}[h_i^2] = n_{in} \sigma^2 \gamma
$$
这样在经过一个层之后数据的方差会改变，为了保持方差不变，可以使用上述的初始化方法。

**He Initialization**：(ReLU activations)
$$
Var(W)= \frac{2}{n_{in}}
$$
[[权重初始化.pdf]]
其中$n_{in}$是这一个神经元对应的输入的神经元的个数。


##### Baby Sitting Learning

在训练的过程中，首先在较小的数据集上进行过拟和（在这个训练集上的损失函数接近0）

**学习率**
- 如果一个网络训练的过程中，损失函数不变或变大，那么可能是学习率过大，可以减小学习率。
- 学习率较小，可能会导致训练的过程较慢，可以增大学习率。

**数值爆炸**：
- 尽量使神经元不陷入饱和区，使用上述权重的初始化方法，可以很好缓解。
- 使得输入经过一定的归一化处理，可以尽量避免数值爆炸的问题。

验证误差曲线和训练误差曲线之间的差距较大，可能是过拟合的问题。可以进行早停。现在已经可以使验证误差趋近于渐近线。

##### Batch Normalization

对于输入的数据进行归一化处理，可以加快训练的速度，同时可以减少梯度消失的问题。在训练的过程中，对于每一个batch的数据进行归一化处理，可以使得数据的分布更加稳定。
$$
\hat{x} = \frac{x - \mu}{\sigma}
$$
这是一个非参数化方法。可以加入可学习的参数：
$$
y = \gamma \hat{x} + \beta
$$
其中$\mu$和$\sigma$是对于每一个mini-batch的均值和方差。

在CNN中，对每一个batch中的n个$w \times h$的特征图进行归一化处理，可以使得数据的分布更加稳定。

上述是在训练的过程中使用的，在测试过程中使用不了称为**训练推理失配***train inference mismatch*。可以使用EMA（指数滑动平均）的方法来进行替代。

上述要求n大概是16，在比较大的模型中，可能显存不够。上述方法有一个替代的方法*Layer Normalization*，对于每一个样本进行归一化处理。

在使用了*Batch Normalization*之后，仍然有协变量偏移*covariate shift*的问题。但是在使用*Batch Normalization*之后，*Lipchitz*系数变化更加平稳，海森矩阵也更加稳定。上述可以用数学严格证明。上述操作并不是简单的归一化，而是使得表示的函数族更加光滑，一个光滑的、凸的函数更容易优化。
- Lipchitz:
$$
\left\|\nabla_{y_j} \hat{\mathcal{L}}\right\|^2 \leq \frac{\gamma^2}{\sigma_j^2}\left(\left\|\nabla_{y_j}\right\|^2-\frac{1}{m}\left(1, \nabla_{y_j} \mathcal{L}\right)^2-\frac{1}{m}\left(\nabla_{y_j} \mathcal{L}, \hat{y}_j\right)^2\right)
$$
- Smoothness:
$$
\gamma<\sigma \text { in experiments }
$$
- Hessian matrix

$$
\left(\nabla_{y_j} \hat{\mathcal{L}}\right)^T \frac{\partial \hat{\mathcal{L}}}{\partial y_j \partial y_j}\left(\nabla_{y_j} \hat{\mathcal{L}}\right) \leq \frac{\gamma^2}{\sigma_j^2}\left(\left(\nabla_{y_j} \mathcal{L}\right)^T \frac{\partial \mathcal{L}}{\partial y_j \partial y_j}\left(\nabla_{y_j} \mathcal{L}\right)-\frac{\gamma}{m \sigma^2}\left(\nabla_{y_j} \mathcal{L}, \hat{y}_j\right)\left\|\nabla_{y_j} \hat{\mathcal{L}}\right\|^2\right)
$$

##### Group Normalization

![[{139D48AB-F664-4CE8-AC05-97B772908A85}.png]]
在*Group Normalization*中，对于每一个通道的特征图进行归一化处理，这样可以减少计算的复杂度。是轻量化CNN的方法。在一定数据量较大的情况下可以达到和*Batch Normalization*差不多的结果。

### Generalization and Capacity

- 网络结构不同网络效果不同，如相同的层数下，全连接网络的参数量大但是和卷积网络的效果差不多。
- 相同的网络结构，参数量不同，参数量多的网络效果更好。

#### Theorem (Arbitrarily large neural networks can approximate any function)

理论可以表述为：对于任意的连续函数，存在一个足够大的神经网络可以近似这个函数。
![[{79D51D58-D7B7-485E-9A0F-5F615FE27545}.png]]
上面表示两层神经网络可以逼近任意的连续函数，要求这个函数$\sigma$不是多项式函数。

![[{862689B6-2775-4822-8FAC-B0450B360BA0}.png]]
上面的定理表示神经网络的宽度也很重要，可以通过增加神经元的数量来逼近函数。

 在空间折叠的问题中，表明**深度比宽度更加重要**。
