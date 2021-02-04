Formulas

Algorithms:

* Factorization Machine

2-order FM is commonly userd.

$$FM(x) = w_0 + \sum_{i=1}^n{w_ix_i}+ \sum_{i=1}^n\sum_{j=i+1}^n<v_i,v_j>x_ix_j$$

​     		  $= w_0 + \sum_{i=1}^nw_ix_i+\frac{1}{2}\sum_{f=1}^k((\sum_{i=1}^nv_{i,f}x_i)^2-\sum_{i=1}^n{v_{i,f}^2x_i^2})$

Gradients of FM:

1-order part is simple.

2-order part: $$\frac{\partial f}{\partial v_i}=x_i * \sum_{i=1}^nv_{i,f}x_i - v_{i,f}x_i^2$$



Optimizer:

* AdaGrad

SGD uses a gloabl learning rate for all weights, which may cause difficults as each individual weight may have different converge rate.

Adagrad use different Learning rate for each individual weight.

$w^{t+1} = w^t-\frac{l}{\sqrt{\sum_{i=0}^t(g^i)^2}}g^t$

注：adagrad使用每个变量的历史累计梯度的平方和的倒数，作为当前学习率的加权。

* RMSProp

RMSProp is similar to AdaGrad, the only different is the formula of accumulate gradients.

For Adagrad: $r=r+g^2$

For RMSProp: $r=p*r+(1-p)*g^2$

注：RMSProp在计算梯度累计时，使用了衰减系数p来减少间隔较久的历史梯度的影响，更加鲁棒。

* AdaDelta



Loss:

* MSE

$MSE=\frac{1}{n}\sum_{i=1}^n(\hat{y_i}-y_i)^2$

* LogLoss（Binary）

$LogLoss = -\frac{1}{n}\sum_{i=1}^n(y_i\log{p_i}+(1-y_i)\log{(1-p_i)})$

