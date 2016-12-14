import tensorflow as tf # tensorflow 를 ts라는 이름으로 import해온다.
from tensorflow.examples.tutorials.mnist import input_data # tensorflow.examples.tutorials.mnist 에서 input_data를 import 해온다.
#  
# Dataset loading
mnist = input_data.read_data_sets("./samples/MNIST_data/", one_hot=True) #데이터를 다운로드 해온다.

# Set up model
x = tf.placeholder(tf.float32, [None, 784]) # 심볼릭 변수들을 사용하여 상호작용하는 작업들을 기술합니다. [None, 784] 형태의 부정소숫점으로 이루어진 2차원 텐서로 표현합니다
W = tf.Variable(tf.zeros([784, 10])) # W 와 b 를 0으로 채워진 텐서들로 초기화합니다. 우리가 W와 b 를 학습할 것이기 때문에, 그것들이 무엇으로 초기화되었는지는 크게 중요하지 않습니다.
b = tf.Variable(tf.zeros([10])) # W의 형태가 [784, 10] 임을 주의합시다. 우리는 784차원의 이미지 벡터를 곱하여 10차원 벡터의 증거를 만들것이기 때문입니다. b는 [10]의 형태이므로 출력에 더할 수 있습니다.
y = tf.nn.softmax(tf.matmul(x, W) + b) # tf.matmul(x, W) 표현식으로 xx 와 WW를 곱합니다. 이 값은 WxWx가 있던 우리 식에서 곱한 결과에서 뒤집혀 있는데, 이것은 xx가 여러 입력으로 구성된 2D 텐서일 경우를 다룰 수 있게 하기 위한 잔재주입니다. 그 다음 b를 더하고, 마지막으로 tf.nn.softmax 를 적용합니다.

y_ = tf.placeholder(tf.float32, [None, 10]) # 교차 엔트로피를 구현하기 위해 우리는 우선적으로 정답을 입력하기 위한 새 placeholder를 추가해야 합니다:

cross_entropy = -tf.reduce_sum(y_*tf.log(y)) # 그 다음 교차 엔트로피 −∑y′log(y) 를 구현할 수 있습니다. 첫번째로, tf.log는 y의 각 원소의 로그값을 계산합니다. 그 다음, y_ 의 각 원소들에, 각각에 해당되는 tf.log(y)를 곱합니다. 마지막으로, tf.reduce_sum은 텐서의 모든 원소를 더합니다.
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy) # 이 경우 TensorFlow에게 학습도를 0.01로 준 경사 하강법(gradient descent) 알고리즘을 이용하여 교차 엔트로피를 최소화하도록 명령했습니다. 경사 하강법은 TensorFlow 가 각각의 변수들을 비용을 줄이는 방향으로 약간씩 바꾸는 간단한 방법입니다.

# Session
init = tf.initialize_all_variables() # 실행 전 마지막으로 우리가 만든 변수들을 초기화하는 작업을 추가해야 합니다.

sess = tf.Session() # 이제 세션에서 모델을 시작하고 변수들을 초기화하는 작업을 실행할 수 있습니다:
sess.run(init)

# Learning
for i in range(1000): # 확인 차원에서 1000번 반복
  batch_xs, batch_ys = mnist.train.next_batch(100) # 각 반복 단계마다, 학습 세트로부터 100개의 무작위 데이터들의 일괄 처리(batch)들을 가져옵니다. placeholders를 대체하기 위한 일괄 처리 데이터에 train_step 피딩을 실행합니다.
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}) # 무작위 데이터의 작은 일괄 처리를 이용하는 방법을 확률적 교육(stochastic training) 이라고 합니다 -- 이 경우에는 확률적 경사 하강법입니다. 이상적으로는 무엇을 해야 할지에 대해 더 나은 직관을 줄 수 있도록 학습의 모든 단계에 모든 데이터를 사용하고 싶습니다만, 그 과정은 비쌉니다. 그래서 그 대신에 우리는 각 시간마다 다른 서브셋을 사용합니다. 이렇게 하면 저렴하면서도 거의 비슷한 효과를 볼 수 있습니다.

# Validation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) # tf.argmax는 특정한 축을 따라 가장 큰 원소의 색인을 알려주는 엄청나게 유용한 함수입니다. 예를 들어 tf.argmax(y,1) 는 진짜 라벨이 tf.argmax(y_,1) 일때 우리 모델이 각 입력에 대하여 가장 정확하다고 생각하는 라벨입니다. 우리는 tf.equal 을 이용해 예측이 실제와 맞았는지 확인할 수 있습니다.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # 이 결과는 부울 리스트를 줍니다. 얼마나 많은 비율로 맞았는지 확인하려면, 부정소숫점으로 캐스팅한 후 평균값을 구하면 됩니다. 예를 들어, [True, False, True, True]는 [1,0,1,1] 이 되고 평균값은 0.75가 됩니다.

# Result should be approximately 91%.
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})) # 테스트 데이터를 대상으로 정확도를 확인해 봅시다.