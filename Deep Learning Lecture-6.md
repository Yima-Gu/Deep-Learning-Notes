
## Transformers

### Transformers: Attention is All You Need

[[Deep Learning Lecture-5#Attention]]

再次理解Attention的概念：类似于”查字典“的操作，对于Query $q$, Key $k$和Value $v$，计算相关性，也就是重要性，对于输出序列中的第$i$个输出有价值的信息：
$$
w_{ij} = a(q_i, k_j)
$$
其中$a$是一个函数，可以是内积、*Additive Attention*等。对于输出序列中的第$i$个输出，计算当前的输出的$q_i$，计算与输入序列中的$k_j$的相关性，然后对于$v_j$进行加权求和（这是一种寻址操作），得到的$c_i$是查字典所得到的信息：
$$
c_i = \sum_{j=1}^T w_{ij}v_j
$$
**希望找到一种更好的计算方法**。

在[[Deep Learning Lecture-5#RNN with Attention]]中问题在于：
- 太复杂的模型
- 某种意义上使用的Attention已经足够使用，不再需要循环网络
- 循环网络的计算是串行的，不能有效加速
#### Self-Attention

计算的是同一条序列中的不同位置之间的相关性，也就是自注意力。对于输入序列中的第$i$个位置，计算与其他位置的相关性，然后对于所有的位置进行加权求和：
规定Query $Q = [q_1 \dots q_n]$，Key $K = [k_1 \dots k_n]$，Value $V = [v_1 \dots v_k]$，则：

![[Pasted image 20250323133751.png]]

#### Scaled Dot-Product Attention

我们认为使用一个网络来计算相关性太复杂了，当两个向量是相同维度的时候可以直接计算内积。在这里，在计算先引入参数，使得其维度是一样的，从而可以计算内积：

*Scaled Dot-Product* :
$$
a(q,k) = \frac{q^T k}{\sqrt{d_k}}
$$
使得变换前后的方差是一样的，这样可以使得梯度更加稳定，否则可能进入激活函数的饱和区。
$$
Attention(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$
上面的式子将得到的$n \times n$的矩阵进行softmax操作，在归一化的过程中，**是某一个query在所有的key上的注意力分配一定是$\mathbf{1}$**。后面是对于Value的加权求和。

![[Pasted image 20250323141917.png]]

对于同一组输入，经过不同的线性变换得到的不同的Query、Key和Value，在样本数量为$m$的情况下，可以进行计算：

$$
\begin{aligned}
&W^Q \in \mathbb{R}^{d_k \times d_{\text{input}}}, \\
&W^K \in \mathbb{R}^{d_k \times d_{\text{input}}}, \\
&W^V \in \mathbb{R}^{d_v \times d_{\text{input}}} \\
&Q = X W^Q \in \mathbb{R}^{m \times d_k}, \\
&K = X W^K \in \mathbb{R}^{m \times d_k}, \\
&V = X W^V \in \mathbb{R}^{m \times d_v}. \\
&QK^T \in \mathbb{R}^{m \times m}, \\
&\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{m \times m}, \\
&\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \in \mathbb{R}^{m \times d_v}.

\end{aligned}
$$

 **维度总结表**

| 矩阵/操作                          | 维度                          | 说明                                    |
| ------------------------------ | --------------------------- | ------------------------------------- |
| 输入矩阵 $X$                       | $m \times d_{\text{input}}$ | 包含 $m$ 个样本，每个样本维度为 $d_{\text{input}}$ |
| 查询矩阵 $Q$                       | $m \times d_k$              | 每个样本的查询向量维度为 $d_k$                    |
| 键矩阵 $K$                        | $m \times d_k$              | 每个样本的键向量维度为 $d_k$                     |
| 值矩阵 $V$                        | $m \times d_v$              | 每个样本的值向量维度为 $d_v$                     |
| 注意力得分矩阵 $QK^T$                 | $m \times m$                | 样本间的注意力强度矩阵                           |
| 最终输出 $\text{Attention}(Q,K,V)$ | $m \times d_v$              | 聚合所有样本的加权值信息，输出维度为 $d_v$              |

#### Multi-Head Attention

注意到上面的注意力的表达能力是相当有限的，在language model同一个词和其他不同的词之间可能有很多种不同的关系，仅仅用一种简单的关系来表示是不够的。所以我们引入多头注意力，希望能在不同的侧面上进行表达。
$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O
$$
其中：
$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) 
$$
其中$W_i^Q, W_i^K, W_i^V$是不同的线性变换，$W^O$是最后的线性变换，最后进行的维度的规约操作。
与CNN相比，CNN的不同的通道之间与上一层的每一个通道之间都是有连接的；但是在这里，不同的头之间是没有连接的，这样可以使得不同的头可以关注不同的信息。

不同的头之间是可以并行计算的，这样可以加速计算；但是缺点是内存占用会很大。

#### Position-wise Feed-Forward Networks

上面的操作除了SoftMax之外都是线性的，这样的表达能力会有限的。这里使用的SoftMax生成的是稀疏的权重。所以我们引入一个全连接层，使得其表达能力更强，这里使用的其实是一个卷积网络，用不同的卷积核来进行卷积操作，然后使用ReLU激活函数。

> 这里在进行什么操作？
> 是不是将一个词语的向量经过一个线性变换之后再激活成一个标量
> 为什么需要这样的操作？

![[Pasted image 20250323145847.png]]

我们希望不同的词之间的特征尽量不要进行相互影响，所以这里使用的CNN是尽可能干净地学习到一个词语内部的特征。上下文关系在Attention之间已经较好地解决了。这里用的卷积核实际上是RNN中的不同的位置（时间）上有相同参数的假设。**这个思想提高了transformer的表达能力**

#### Residual Connection

在上面的操作中，这些操作都是有排列不变性。
残差是一个标准的操作，这样可以让网络更好地记录位置编码。

#### Layer Normalization

目的是使得每一层经过Attention和Feed-Forward之后的输出的分布是一样的，这样可以使得梯度更加稳定。
[[Deep Learning Lecture-5#Layer Normalization]]

![[Pasted image 20250323151056.png]]

#### Positional Encoding

位置信息是顺序信息的一种泛化的形式。如果采用独热编码，这是一种类别信息而不是一个顺序信息，不同的是不可以比的。所以引入*position embedding*，这是一个矩阵，效果类似于一个查找表。查找操作在这里就是一个矩阵乘上一个独热编码的操作，这是因为GPU在矩阵乘法操作上是非常高效的。
但是独热编码会带来下面的问题
- **高维稀疏性**：  独热编码的维度等于序列最大长度（如512），导致向量稀疏且计算效率低下（尤其对长序列）。
- **无法泛化到未见长度**：  若训练时序列最大长度为512，模型无法处理更长的序列


**引入归纳偏好**：
- 每个位置的编码应该是独一无二的且是确定的
- 认为两个位置的距离应该是一致的
- 应该生成一个有界的值，位置数随着序列长度的增加而增加

Google的实现是使用的正弦和余弦函数的组合：
$$
e_i(2j) = \sin\left(\frac{i}{10000^{2j/d_{\text{model}}}}\right)
$$
$$
e_i(2j+1) = \cos\left(\frac{i}{10000^{2j/d_{\text{model}}}}\right)
$$
上述公式中的$i$指的句子中的第$i$个位置，$j$指的是位置编码的维度，$d_{\text{model}}$是位置编码的维度。这样的编码是满足上面的归纳偏好的。

#### Encoder

编码器中使用的是多头注意力、逐位置前馈网络和位置编码。在这个编码器中是一个直筒式的网络，好处是调参较为简单。

缺点：
- 二次复杂度
- 参数量过大
- 很多的头是冗余的

训练阶段要使用多个头，发现有些头的权重较低，可以在推理阶段去掉这些头。

#### Decoder

##### Autoregressive

![[Pasted image 20250323161530.png]]

预测阶段一定要使用滚动预测，这是一个自回归的状态，但是这是一个串行的操作，会比较慢。但是在训练阶段这样是不能接受的，我希望训练的不同阶段可以并行计算，但是这里要求在一开始输入所有的序列，所以这里需要**遮挡**。
在算Attention的时候，对于当前的位置，只能看到之前的位置，不能看到之后的位置。

![[Pasted image 20250323163151.png]]

在编码器上是不能用的，因为防止解码器在训练时利用未来的目标序列信息（即“作弊”），确保模型逐步生成的能力与推理阶段一致。训练过程中仍然需要真实标签作为目标输出，但掩码限制了模型在生成当前词时对未来的访问。

#### Encoder-Decoder Attention

计算的是解码器的输出和编码器的输出之间的相关性，这里的Query是解码器的输出，Key和Value是编码器的输出。

![[Pasted image 20250323185530.png]]


![[Pasted image 20250323185720.png]]

注意这里是将编码器的输出输入到解码器中的每一层的Encoder-Decoder Attention中。这里是神经网络中的**特征重用**思想，并且解码器中的网络是直筒式的，所以这些特征是可以重用的。

#### RNN vs. Transformer

- RNN是串行的，Transformer是并行的
- 对于有严格偏序关系的序列，RNN可能更适合
- 对于长序列，Transformer更适合
- 对于较小的数据量，Transformers参数量较大，表现可能不如RNN

![[Pasted image 20250323190623.png]]

![[Pasted image 20250323190856.png]]
### X-formers Variance with Improvements

[[2106.04554] A Survey of Transformers](https://arxiv.org/abs/2106.04554)

![[Pasted image 20250323191632.png]]

![[Pasted image 20250323191637.png]]

#### Lineariezd Attention

#### Flow Attention

### GPT: Generative Pre-trained Transformer

#### Transfer Learning

先将一个模型预训练好，然后在特定的任务上进行微调。一般而言，预训练的过程是无监督的，优点是可以使用大规模数据。

#### Pre-Training

![[Pasted image 20250324190821.png]]

- 直接使用的是Transformers中的block，但是这里使用12层
- 只使用decoder没有encoder，因为这不是一个机器翻译的任务
- 在计算损失函数的过程中，使用的似然函数是最大似然估计，在实际中使用一个参数化的网络来近似需要的概率。

#### Supervised Fine-Tuning

对于不同的任务，需要更换模型的输出头，并且还要使用新的损失函数。关注上下文建模。
![[Pasted image 20250324193704.png]]

最后是使用无监督训练的损失函数和有监督训练的损失函数的加权和，这是一个**多任务学习**。当微调的数据比较少的时候，可以使用无监督训练的损失函数的权重较大。

对于不同的下游任务，要进行任务适配*Task Specific Adaptation*。对于不同的下游任务，可以使用不同的头。
![[Pasted image 20250324194137.png]]

#### GPT-2 & GPT-3

Zero-shot learning：在没有看到训练数据的情况下，直接在测试集上进行预测。通过在预训练阶段使用大规模的数据，可以使得模型具有更好的泛化能力，这样可以提高在一些常见问题上的表现。

### BERT: Bidirectional Encoder Representations from Transformers

与GPT不同的是，BERT是双向的，可以看到上下文的信息。
![[Pasted image 20250324195035.png]]

BERT在encoder阶段就使用了mask，这样可以使得模型在训练的时候不会看到未来的信息。在训练的过程中随机地mask掉一些词，然后预测这些词。如果遮挡的词太少，那么模型得到的训练不够， 如果遮挡的词太多，那么得到的上下文就很少。
在训练的过程中就使用了102种语言。
特征工程：使用了更多的特征，引入了更多的embedding
是多个任务的联合训练，这样可以使得模型更加通用。

#### RoBERTa: A Robustly Optimized BERT Pretraining Approach

经过充分的调参和更长的训练时间，使得模型的表现更好。
证明了BERT中的下句预测是没有用的，因为在RoBERTa中去掉了这个任务。
mask的pattern可以动态调整

#### ALBERT: A Lite BERT for Self-supervised Learning of Language Representations

低秩分解，减少参数量
![[Pasted image 20250324203227.png]]

跨层参数共享：可以让模型更加稳定

#### T5: Text-to-Text Transfer Transformer

迁移是泛化的高级形式：可以将多种文本任务统一为文本到文本的形式，这样可以使得模型更加通用。

架构层面的创新：
![[Pasted image 20250324203639.png]]

这里使用的是prefix-LM，这样可以使得模型更加通用。

### Vision Transformer

#### ViT

将一个图像变成一个patch
增加一个Position Embedding，于是得到各个patch的特征的加权平均。
 主要的贡献是将图像转换为序列，从而可以使用transformers来进行建模。在这个之前，普遍的观点是transformers只能用于文本数据，而CNN用于图像数据。

#### Swim Transformer

将CNN中的一些归纳偏好引入，可以使用局部的注意力，但是在一定程度上能捕捉全局的信息，通过Shifted Window Mechanism来实现。
层次化特征：
![[Pasted image 20250324205304.png]]

密集预测任务对于层次化特征需求更高，于是这个模型的表现是更好地。

#### DETR

![[Pasted image 20250324205845.png]]

![[Pasted image 20250324205921.png]]

### Fundation Models

 