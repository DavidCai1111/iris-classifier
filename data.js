'use strict'
const fs = require('fs')
const path = require('path')
const csv = require('neat-csv')

async function getTrainData () {
  return await getData('./data/iris_training.csv')
}

async function getTestData () {
  return await getData('./data/iris_test.csv')
}

async function getData (dataPath) {
  const parsedCSV = await csv(fs.readFileSync(dataPath))

  const inputs = {
    sepalLength: new Float32Array(parsedCSV.length),
    sepalWidth: new Float32Array(parsedCSV.length),
    petalLength: new Float32Array(parsedCSV.length),
    petalWidth: new Float32Array(parsedCSV.length)
  }

  const labels = []

  parsedCSV.forEach((row, i) => {
    inputs.sepalLength[i] = row['4']
    inputs.sepalWidth[i] = row['30']
    inputs.petalLength[i] = row['setosa']
    inputs.petalWidth[i] = row['versicolor']

    labels.push(row['virginica'])
  })

  return { inputs, labels }
}

module.exports = {
  getTrainData,
  getTestData
}
