'use strict'
const tf = require('@tensorflow/tfjs')
const co = require('co')
const { getTrainData, getTestData } = require('./data')

co(async function () {
  const model = tf.sequential()
  model.add(tf.layers.dense({ units: 10, activation: 'relu', inputShape: [4] }))
  model.add(tf.layers.dense({ units: 10, activation: 'relu' }))
  model.add(tf.layers.dense({ units: 3, activation: 'softmax' }))

  const optimizer = tf.train.adam(0.001)
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  })

  const trainData = await getTrainData()
  const testData = await getTestData()

  await model.fit(trainData.inputs, trainData.labels, {
    epochs: 300,
    validationData: [testData.inputs, testData.labels]
  })

  const res = model.predict(testData.inputs[3])
  console.log(res, testData.labels[3])
}).catch(console.error)
