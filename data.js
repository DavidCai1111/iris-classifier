'use strict'
const fs = require('fs')
const tf = require('@tensorflow/tfjs')
const csv = require('neat-csv')

async function getTrainData () {
  return await getData('./data/iris_training.csv')
}

async function getTestData () {
  return await getData('./data/iris_test.csv')
}

async function getData (dataPath) {
  const parsedCSV = await csv(fs.readFileSync(dataPath))

  let inputs = []
  let labels = []

  parsedCSV.forEach((row, i) => {
    inputs.push(row['4'], row['30'], row['setosa'], row['versicolor'])

    const label = row['virginica']
    if (label === 0) labels.push(1, 0, 0)
    if (label === 1) labels.push(0, 1, 0)
    else labels.push(0, 0, 1)
  })

  return {
    inputs: tf.tensor(inputs, [parsedCSV.length, 4]),
    labels: tf.tensor(labels, [parsedCSV.length, 3])
  }
}

module.exports = {
  getTrainData,
  getTestData
}
