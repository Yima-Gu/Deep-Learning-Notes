

**人工智能**：模拟人类的智能
**机器学习**：不显式编程来实现
**深度学习**：使用神经网络

对于深度学习总是能找到一些amazing的例子，同时也有一些失败的例子。

深度学习是在数据中**获取高度抽象**的特征，主要使用神经网络。在深度学习中实现的功能类似于坐标系变换。对于同一个问题进行变换可以进行不同的处理。在机器学习过程中的规则系统仍然是很重要的。

深度学习的表示为计算图

### Framework

在学习算法中一定要指定函数族，然后使用数据来拟合函数族。要找出**最优的函数族**。不同的机器学习模型的区别主要在于**假设空间**（候选函数族）的不同。

过拟和在指定函数族时已经确定。

泛化误差公式：
$$
\varepsilon_{test} \leq \hat{{\varepsilon}}_{train} + \sqrt{\frac{complexity}{n}}
$$
要求：训练误差小，模型复杂度小，数据量大
**当且仅当需要的时候才增加模型复杂度**

### Model

$$
\arg \min O(D;\theta) = \sum_{i=1}^{N} L(y_i, f(x_i);\theta) + \Omega(\theta)
$$
上述公式要求：在训练样本上预测值与真实值的误差尽可能小。

对于分类问题而言，经常使用的是交叉熵损失函数（logistic）和hinge损失函数。

$\Omega(\theta)$是惩罚项，希望模型尽量简单。在使得$\theta$尽量小的过程中，某些特征的权重会变为0，这样就可以实现特征选择。同时也可以减少模型的复杂性。

**学习类似于统计中的参数估计**

![[{65C91395-95B8-4EF2-8AF6-3F86705FD81B}.png]]

梯度下降的方法。

具体例子：线性回归

$$
O(D;\theta) = \sum_{i=1}^{N} (y_i - \theta^T x_i)^2 + \lambda \theta^T \theta = (Y - X\theta)^T (Y - X\theta) + \lambda \theta^T \theta
$$
上述公式是关于$\theta$的二次函数，可以通过求导得到最优解。[[matrixcookbook.pdf]]
$$
\frac{\partial O(D;\theta)}{\partial \theta} = -2 X^T (Y - X\theta) + 2 \lambda \theta = 0
$$
$$
\hat{\theta} = (X^T X + \lambda I)^{-1} X^T Y
$$
但是上述算法的复杂性为$O(n^3)$，不适用于大数据集。在软件工程中，通常使用梯度下降法。

**Logistic Regression**

$$
O(D;\theta) = \sum_{i=1}^{n} \log(1 + \exp(-y_i \theta^T x_i)) + \lambda ||\theta||_{1} = F(D;\theta) + \lambda ||\theta||_{1}
$$
这里使用的是近端梯度下降法 *proximal gradient descent*。其中的1-范数是各个分量的绝对值之和。[机器学习 | 近端梯度下降法 (proximal gradient descent)](https://www.zhihu.com/tardis/zm/art/82622940?source_id=1005)
在一些不可导的情况下如hinge损失函数，可以使用次梯度。

**Softmax Regression**

实现多分类问题，计算每个类别的概率即可，对于某个样本进行多次打分，取最高分对应的类别。

Softmax Function:
[[Deep Learning Lecture-2#^b5bcbb]]
$$
P(y|x,\theta) = \frac{\exp(\theta_y^T x)}{\sum_{r=1}^{C} \exp(\theta_{r}^T x)}
$$
线性模型，用超平面对样本进行打分，然后使用softmax函数进行归一化。

$$
O(D;\theta) = -\sum_{i=1}^{n} \log P(y_i|x_i;\theta) + \lambda ||\theta||_{1}
$$
上述式子其实是对数似然函数，最大化对数似然函数等价于最小化交叉熵损失函数。


#### Aproximation and Estimation

假设空间只能表现一个有限的函数集合，有时候真值函数不一定在假设空间中。这时候假设空间中最好的函数与真值函数之间的差距称为**近似误差***Approximation Error*。学习得到的函数与空间中最好的函数的误差为**估计误差** *Estimation Error*。

### Model Selection

*All models are wrong but some are useful.*

CASH: Combined Algorithm Selection and Hyperparameter optimization
目标是：选择最好的模型和超参数
$$
A^*_{\lambda^*} = \arg \min_{A \in \mathcal{A}} \min_{\lambda \in \Lambda^{(i)}} \mathcal{L}(A_{\lambda}^{(i)},D_{train},D_{valid})
$$




