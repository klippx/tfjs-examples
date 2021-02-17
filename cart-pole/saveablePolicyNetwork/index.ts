import * as tf from '@tensorflow/tfjs'
import { PolicyNetwork } from './policyNetwork'

// The IndexedDB path where the model of the policy network will be saved.
const MODEL_SAVE_PATH_ = 'indexeddb://cart-pole-v1'

/**
 * A subclass of PolicyNetwork that supports saving and loading.
 */
export class SaveablePolicyNetwork extends PolicyNetwork {
  /**
   * Constructor of SaveablePolicyNetwork
   *
   * @param {number | number[]} hiddenLayerSizes
   */
  constructor({
    layersModel,
    sizes,
  }: {
    layersModel?: tf.LayersModel
    sizes?: {
      hiddenLayerSizes: number | number[]
      inputSize: number
      outputSize: number
    }
  }) {
    if (layersModel !== undefined) {
      super({ layersModel })
    } else if (sizes !== undefined) {
      super({ sizes })
    }
  }

  /**
   * Save the model to IndexedDB.
   */
  async saveModel() {
    return await this.policyNet.save(MODEL_SAVE_PATH_)
  }

  /**
   * Load the model fom IndexedDB.
   *
   * @returns The instance of loaded `SaveablePolicyNetwork`.
   * @throws {Error} If no model can be found in IndexedDB.
   */
  static async loadModel() {
    const modelsInfo = await tf.io.listModels()
    if (MODEL_SAVE_PATH_ in modelsInfo) {
      console.log(`Loading existing model...`)
      const layersModel = await tf.loadLayersModel(MODEL_SAVE_PATH_)
      console.log(`Loaded model from ${MODEL_SAVE_PATH_}`)
      return new SaveablePolicyNetwork({ layersModel })
    } else {
      throw new Error(`Cannot find model at ${MODEL_SAVE_PATH_}.`)
    }
  }

  /**
   * Check the status of locally saved model.
   *
   * @returns If the locally saved model exists, the model info as a JSON
   *   object. Else, `undefined`.
   */
  static async checkStoredModelStatus() {
    const modelsInfo = await tf.io.listModels()
    return modelsInfo[MODEL_SAVE_PATH_]
  }

  /**
   * Remove the locally saved model from IndexedDB.
   */
  async removeModel() {
    return await tf.io.removeModel(MODEL_SAVE_PATH_)
  }

  /**
   * Get the sizes of the hidden layers.
   *
   * @returns If the model has only one hidden layer,
   *   return the size of the layer as a single number. If the model has
   *   multiple hidden layers, return the sizes as an Array of numbers.
   */
  hiddenLayerSizes() {
    const sizes: number | number[] = []
    for (let i = 0; i < this.policyNet.layers.length - 1; ++i) {
      // `units` does not exist on Layer (!?)
      const layer = this.policyNet.layers[i] as any
      sizes.push(layer.units)
    }
    return sizes.length === 1 ? sizes[0] : sizes
  }
}
