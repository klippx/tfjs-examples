/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'

import { CartPole } from './cart_pole'
import { SaveablePolicyNetwork } from './saveablePolicyNetwork'
import { mean, sum } from './utils'

const getElementById = (id: string) => {
  const el = document.getElementById(id)
  if (el === null) {
    throw new Error(`Could not find element with id ${id}`)
  }
  return el
}

const appStatus = getElementById('app-status') as HTMLSpanElement
const storedModelStatusInput = getElementById(
  'stored-model-status'
) as HTMLInputElement
const hiddenLayerSizesInput = getElementById(
  'hidden-layer-sizes'
) as HTMLInputElement
const createModelButton = getElementById('create-model') as HTMLButtonElement
const deleteStoredModelButton = getElementById(
  'delete-stored-model'
) as HTMLButtonElement
const cartPoleCanvas = getElementById('cart-pole-canvas') as HTMLCanvasElement

const numIterationsInput = getElementById('num-iterations') as HTMLInputElement
const gamesPerIterationInput = getElementById(
  'games-per-iteration'
) as HTMLInputElement
const discountRateInput = getElementById('discount-rate') as HTMLInputElement
const maxStepsPerGameInput = getElementById(
  'max-steps-per-game'
) as HTMLInputElement
const learningRateInput = getElementById('learning-rate') as HTMLInputElement
const renderDuringTrainingCheckbox = getElementById(
  'render-during-training'
) as HTMLInputElement

const trainButton = getElementById('train') as HTMLButtonElement
const testButton = getElementById('test') as HTMLButtonElement
const iterationStatus = getElementById('iteration-status') as HTMLLabelElement
const iterationProgress = getElementById(
  'iteration-progress'
) as HTMLProgressElement
const trainStatus = getElementById('train-status') as HTMLLabelElement
const trainSpeed = getElementById('train-speed') as HTMLSpanElement
const trainProgress = getElementById('train-progress') as HTMLProgressElement

const stepsContainer = getElementById('steps-container')

// Module-global instance of policy network.
let policyNet: SaveablePolicyNetwork | null
let stopRequested = false
// Objects and functions to support display of cart pole status during training.
let renderDuringTraining = true

/**
 * Display a message to the info div.
 *
 * @param {string} message The message to be displayed.
 */
function logStatus(message: string) {
  appStatus.textContent = message
}

/**
 * A function invokved at the end of a training iteration.
 *
 * @param {number} iterationCount A count of how many iterations has completed
 *   so far in the current round of training.
 * @param {*} totalIterations Total number of iterations to complete in the
 *   current round of training.
 */
function onIterationEnd(iterationCount: number, totalIterations: number) {
  trainStatus.textContent = `Iteration ${iterationCount} of ${totalIterations}`
  trainProgress.value = (iterationCount / totalIterations) * 100
}

// Objects and function to support the plotting of game steps during training.
let meanStepValues: Array<{ x: number; y: number }> = []
function plotSteps() {
  tfvis.render.linechart(
    stepsContainer,
    { values: meanStepValues },
    {
      xLabel: 'Training Iteration',
      yLabel: 'Mean Steps Per Game',
      width: 400,
      height: 300,
    }
  )
}

function disableModelControls() {
  trainButton.textContent = 'Stop'
  testButton.disabled = true
  deleteStoredModelButton.disabled = true
}

function enableModelControls() {
  trainButton.textContent = 'Train'
  testButton.disabled = false
  deleteStoredModelButton.disabled = false
}

async function updateUIControlState() {
  const modelInfo = await SaveablePolicyNetwork.checkStoredModelStatus()
  if (modelInfo == null) {
    storedModelStatusInput.value = 'No stored model.'
    deleteStoredModelButton.disabled = true
  } else {
    storedModelStatusInput.value = `Saved@${modelInfo.dateSaved.toISOString()}`
    deleteStoredModelButton.disabled = false
    createModelButton.disabled = true
  }
  createModelButton.disabled = policyNet != null
  hiddenLayerSizesInput.disabled = policyNet != null
  trainButton.disabled = policyNet == null
  testButton.disabled = policyNet == null
  renderDuringTrainingCheckbox.checked = renderDuringTraining
}

export async function setUpUI() {
  const cartPole = new CartPole()

  if ((await SaveablePolicyNetwork.checkStoredModelStatus()) != null) {
    policyNet = await SaveablePolicyNetwork.loadModel()
    logStatus('Loaded policy network from IndexedDB.')
    hiddenLayerSizesInput.value = policyNet.hiddenLayerSizes().toString()
  }
  await updateUIControlState()

  renderDuringTrainingCheckbox.addEventListener('change', () => {
    renderDuringTraining = renderDuringTrainingCheckbox.checked
  })

  createModelButton.addEventListener('click', async () => {
    try {
      const hiddenLayerSizes = hiddenLayerSizesInput.value
        .trim()
        .split(',')
        .map((v) => {
          const num = Number.parseInt(v.trim())
          if (!(num > 0)) {
            throw new Error(
              `Invalid hidden layer sizes string: ` +
                `${hiddenLayerSizesInput.value}`
            )
          }
          return num
        })
      policyNet = new SaveablePolicyNetwork(hiddenLayerSizes)
      console.log('DONE constructing new instance of SaveablePolicyNetwork')
      await updateUIControlState()
    } catch (err) {
      logStatus(`ERROR: ${err.message}`)
    }
  })

  deleteStoredModelButton.addEventListener('click', async () => {
    if (confirm(`Are you sure you want to delete the locally-stored model?`)) {
      await policyNet?.removeModel()
      policyNet = null
      await updateUIControlState()
    }
  })

  trainButton.addEventListener('click', async () => {
    if (trainButton.textContent === 'Stop') {
      stopRequested = true
    } else {
      disableModelControls()
      if (policyNet === null) {
        throw new Error(`Invalid policyNet: ${policyNet}`)
      }
      try {
        const trainIterations = Number.parseInt(numIterationsInput.value)
        if (!(trainIterations > 0)) {
          throw new Error(`Invalid number of iterations: ${trainIterations}`)
        }
        const gamesPerIteration = Number.parseInt(gamesPerIterationInput.value)
        if (!(gamesPerIteration > 0)) {
          throw new Error(
            `Invalid # of games per iterations: ${gamesPerIteration}`
          )
        }
        const maxStepsPerGame = Number.parseInt(maxStepsPerGameInput.value)
        if (!(maxStepsPerGame > 1)) {
          throw new Error(`Invalid max. steps per game: ${maxStepsPerGame}`)
        }
        const discountRate = Number.parseFloat(discountRateInput.value)
        if (!(discountRate > 0 && discountRate < 1)) {
          throw new Error(`Invalid discount rate: ${discountRate}`)
        }
        const learningRate = Number.parseFloat(learningRateInput.value)

        logStatus(
          'Training policy network... Please wait. ' +
            'Network is saved to IndexedDB at the end of each iteration.'
        )
        const optimizer = tf.train.adam(learningRate)

        meanStepValues = []
        onIterationEnd(0, trainIterations)
        let t0 = new Date().getTime()
        stopRequested = false
        for (let i = 0; i < trainIterations; ++i) {
          const gameSteps = await policyNet.train(
            cartPole,
            optimizer,
            discountRate,
            gamesPerIteration,
            maxStepsPerGame,
            async (system) => {
              if (renderDuringTraining) {
                system.render(cartPoleCanvas)
                await tf.nextFrame() // Unblock UI thread.
              }
            },
            (gameCount, totalGames) => {
              iterationStatus.textContent = `Game ${gameCount} of ${totalGames}`
              iterationProgress.value = (gameCount / totalGames) * 100
              if (gameCount === totalGames) {
                iterationStatus.textContent = 'Updating weights...'
              }
            }
          )
          const t1 = new Date().getTime()
          const stepsPerSecond = sum(gameSteps) / ((t1 - t0) / 1e3)
          t0 = t1
          trainSpeed.textContent = `${stepsPerSecond.toFixed(1)} steps/s`
          meanStepValues.push({ x: i + 1, y: mean(gameSteps) })
          console.log(`# of tensors: ${tf.memory().numTensors}`)
          plotSteps()
          onIterationEnd(i + 1, trainIterations)
          await tf.nextFrame() // Unblock UI thread.
          await policyNet.saveModel()
          await updateUIControlState()

          if (stopRequested) {
            logStatus('Training stopped by user.')
            break
          }
        }
        if (!stopRequested) {
          logStatus('Training completed.')
        }
      } catch (err) {
        logStatus(`ERROR: ${err.message}`)
      }
      enableModelControls()
    }
  })

  testButton.addEventListener('click', async () => {
    disableModelControls()
    let isDone = false
    const cartPole = new CartPole()
    cartPole.setRandomState()
    let steps = 0
    stopRequested = false
    while (!isDone) {
      steps++
      tf.tidy(() => {
        if (policyNet === null) {
          throw new Error(`Invalid policyNet: ${policyNet}`)
        }
        const action = policyNet.getActions(cartPole.getStateTensor())[0]
        logStatus(
          `Test in progress. ` +
            `Action: ${action === 1 ? '<--' : ' -->'} (Step ${steps})`
        )
        isDone = cartPole.update(action)
        cartPole.render(cartPoleCanvas)
      })
      await tf.nextFrame() // Unblock UI thread.
      if (stopRequested) {
        break
      }
    }
    if (stopRequested) {
      logStatus(`Test stopped by user after ${steps} step(s).`)
    } else {
      logStatus(`Test finished. Survived ${steps} step(s).`)
    }
    console.log(`# of tensors: ${tf.memory().numTensors}`)
    enableModelControls()
  })
}
