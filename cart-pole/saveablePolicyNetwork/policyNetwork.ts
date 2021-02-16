import * as tf from '@tensorflow/tfjs'

import { System } from './system'

/**
 * Policy network for controlling the cart-pole system.
 *
 * The role of the policy network is to select an action based on the observed
 * state of the system. In this case, the action is the leftward or rightward
 * force and the observed system state is a four-dimensional vector, consisting
 * of cart position, cart velocity, pole angle and pole angular velocity.
 *
 */
export class PolicyNetwork {
  public policyNet: tf.Sequential | tf.LayersModel
  private currentActions_: any

  /**
   * Constructor of PolicyNetwork.
   *
   * @param {number | number[] | tf.LayersModel} hiddenLayerSizes
   *   Can be any of the following
   *   - Size of the hidden layer, as a single number (for a single hidden
   *     layer)
   *   - An Array of numbers (for any number of hidden layers).
   *   - An instance of tf.LayersModel.
   */
  constructor(hiddenLayerSizesOrModel) {
    if (hiddenLayerSizesOrModel instanceof tf.LayersModel) {
      this.policyNet = hiddenLayerSizesOrModel
    } else {
      this.policyNet = this.createPolicyNetwork(hiddenLayerSizesOrModel)
    }
  }

  /**
   * Create the underlying model of this policy network.
   *
   * @param {number | number[]} hiddenLayerSizes Size of the hidden layer, as
   *   a single number (for a single hidden layer) or an Array of numbers (for
   *   any number of hidden layers).
   */
  createPolicyNetwork(hiddenLayerSizes: number | number[]) {
    if (!Array.isArray(hiddenLayerSizes)) {
      hiddenLayerSizes = [hiddenLayerSizes]
    }
    // this.policyNet = tf.sequential()
    const network = tf.sequential()
    hiddenLayerSizes.forEach((hiddenLayerSize, i) => {
      network.add(
        tf.layers.dense({
          units: hiddenLayerSize,
          activation: 'elu',
          // `inputShape` is required only for the first layer.
          inputShape: i === 0 ? [4] : undefined,
        })
      )
    })
    // The last layer has only one unit. The single output number will be
    // converted to a probability of selecting the leftward-force action.
    network.add(tf.layers.dense({ units: 1 }))
    return network
  }

  /**
   * Train the policy network's model.
   *
   * @param {System} cartPoleSystem The cart-pole system object to use during
   *   training.
   * @param {tf.train.Optimizer} optimizer An instance of TensorFlow.js
   *   Optimizer to use for training.
   * @param {number} discountRate Reward discounting rate: a number between 0
   *   and 1.
   * @param {number} numGames Number of game to play for each model parameter
   *   update.
   * @param {number} maxStepsPerGame Maximum number of steps to perform during
   *   a game. If this number is reached, the game will end immediately.
   * @returns The number of steps completed in the `numGames` games
   *   in this round of training.
   */
  interface
  async train<T extends System>(
    cartPoleSystem: T,
    optimizer: tf.Optimizer,
    discountRate: number,
    numGames: number,
    maxStepsPerGame: number,
    render: (system: T) => Promise<void>,
    onGameEnd: (gameCount: number, totalGames: number) => void
  ) {
    const allGradients: { [name: string]: Array<tf.Tensor[]> } = {}
    const allRewards: Array<number[]> = []
    const gameSteps: Array<number> = []
    onGameEnd(0, numGames)
    for (let i = 0; i < numGames; ++i) {
      // Randomly initialize the state of the cart-pole system at the beginning
      // of every game.
      cartPoleSystem.setRandomState()
      const gameRewards: Array<number> = []
      const gameGradients: { [name: string]: tf.Tensor[] } = {}
      for (let j = 0; j < maxStepsPerGame; ++j) {
        // For every step of the game, remember gradients of the policy
        // network's weights with respect to the probability of the action
        // choice that lead to the reward.
        const gradients = tf.tidy(() => {
          const inputTensor = cartPoleSystem.getStateTensor()
          return this.getGradientsAndSaveActions(inputTensor).grads
        })

        this.pushGradients(gameGradients, gradients)
        const action = this.currentActions_[0]
        const isDone = cartPoleSystem.update(action)

        await render(cartPoleSystem)

        if (isDone) {
          // When the game ends before max step count is reached, a reward of
          // 0 is given.
          gameRewards.push(0)
          break
        } else {
          // As long as the game doesn't end, each step leads to a reward of 1.
          // These reward values will later be "discounted", leading to
          // higher reward values for longer-lasting games.
          gameRewards.push(1)
        }
      }
      onGameEnd(i + 1, numGames)
      gameSteps.push(gameRewards.length)
      this.pushGradients(allGradients, gameGradients)
      allRewards.push(gameRewards)
      await tf.nextFrame()
    }

    tf.tidy(() => {
      // The following line does three things:
      // 1. Performs reward discounting, i.e., make recent rewards count more
      //    than rewards from the further past. The effect is that the reward
      //    values from a game with many steps become larger than the values
      //    from a game with fewer steps.
      // 2. Normalize the rewards, i.e., subtract the global mean value of the
      //    rewards and divide the result by the global standard deviation of
      //    the rewards. Together with step 1, this makes the rewards from
      //    long-lasting games positive and rewards from short-lasting
      //    negative.
      // 3. Scale the gradients with the normalized reward values.
      const normalizedRewards = discountAndNormalizeRewards(
        allRewards,
        discountRate
      )
      // Add the scaled gradients to the weights of the policy network. This
      // step makes the policy network more likely to make choices that lead
      // to long-lasting games in the future (i.e., the crux of this RL
      // algorithm.)
      optimizer.applyGradients(
        scaleAndAverageGradients(allGradients, normalizedRewards)
      )
    })
    tf.dispose(allGradients)
    return gameSteps
  }

  getGradientsAndSaveActions(inputTensor: tf.Tensor2D) {
    const f = () =>
      tf.tidy(() => {
        const [logits, actions] = this.getLogitsAndActions(inputTensor)
        this.currentActions_ = actions.dataSync()
        const labels = tf.sub(
          1,
          tf.tensor2d(this.currentActions_, actions.shape)
        )
        return tf.losses.sigmoidCrossEntropy(labels, logits).asScalar()
      })
    return tf.variableGrads(f)
  }

  getCurrentActions() {
    return this.currentActions_
  }

  /**
   * Get policy-network logits and the action based on state-tensor inputs.
   *
   * @param {tf.Tensor} inputs A tf.Tensor instance of shape `[batchSize, 4]`.
   * @returns {[tf.Tensor, tf.Tensor]}
   *   1. The logits tensor, of shape `[batchSize, 1]`.
   *   2. The actions tensor, of shape `[batchSize, 1]`.
   */
  getLogitsAndActions(inputs: tf.Tensor): [tf.Tensor, tf.Tensor2D] {
    return tf.tidy(() => {
      // FIXME: as tf.Tensor<tf.Rank>
      const logits = this.policyNet.predict(inputs) as tf.Tensor

      // Get the probability of the leftward action.
      const leftProb = tf.sigmoid(logits)

      // Probabilites of the left and right actions.
      // FIXME: as tf.Tensor2D
      const leftRightProbs = tf.concat(
        [leftProb, tf.sub(1, leftProb)],
        1
      ) as tf.Tensor2D
      const actions: tf.Tensor2D = tf.multinomial(
        leftRightProbs,
        1,
        undefined,
        true
      ) as tf.Tensor2D
      return [logits, actions]
    })
  }

  /**
   * Get actions based on a state-tensor input.
   *
   * @param {tf.Tensor} inputs A tf.Tensor instance of shape `[batchSize, 4]`.
   * @param {Float32Array} inputs The actions for the inputs, with length
   *   `batchSize`.
   */
  getActions(inputs: tf.Tensor) {
    return this.getLogitsAndActions(inputs)[1].dataSync()
  }

  /**
   * Push a new dictionary of gradients into records.
   *
   * @param {{[varName: string]: tf.Tensor[]}} record The record of variable
   *   gradient: a map from variable name to the Array of gradient values for
   *   the variable.
   * @param {{[varName: string]: tf.Tensor}} gradients The new gradients to push
   *   into `record`: a map from variable name to the gradient Tensor.
   */
  pushGradients(
    record: { [varName: string]: Array<tf.Tensor | tf.Tensor[]> },
    gradients: { [varName: string]: tf.Tensor | tf.Tensor[] }
  ) {
    for (const key in gradients) {
      if (key in record) {
        record[key].push(gradients[key])
      } else {
        record[key] = [gradients[key]]
      }
    }
  }
}

/**
 * Discount the reward values.
 *
 * @param {number[]} rewards The reward values to be discounted.
 * @param {number} discountRate Discount rate: a number between 0 and 1, e.g.,
 *   0.95.
 * @returns The discounted reward values as a 1D tf.Tensor.
 */
function discountRewards(rewards: number[], discountRate: number) {
  const discountedBuffer = tf.buffer([rewards.length])
  let prev = 0
  for (let i = rewards.length - 1; i >= 0; --i) {
    const current = discountRate * prev + rewards[i]
    discountedBuffer.set(current, i)
    prev = current
  }
  return discountedBuffer.toTensor()
}

/**
 * Discount and normalize reward values.
 *
 * This function performs two steps:
 *
 * 1. Discounts the reward values using `discountRate`.
 * 2. Normalize the reward values with the global reward mean and standard
 *    deviation.
 *
 * @param {number[][]} rewardSequences Sequences of reward values.
 * @param {number} discountRate Discount rate: a number between 0 and 1, e.g.,
 *   0.95.
 * @returns The discounted and normalize reward values as an Array of tf.Tensor.
 */
function discountAndNormalizeRewards(
  rewardSequences: Array<number[]>,
  discountRate: number
) {
  return tf.tidy(() => {
    const discounted: tf.Tensor[] = []
    for (const sequence of rewardSequences) {
      discounted.push(discountRewards(sequence, discountRate))
    }
    // Compute the overall mean and stddev.
    const concatenated = tf.concat(discounted)
    const mean = tf.mean(concatenated)
    const std = tf.sqrt(tf.mean(tf.square(concatenated.sub(mean))))
    // Normalize the reward sequences using the mean and std.
    const normalized = discounted.map((rs) => rs.sub(mean).div(std))
    return normalized
  })
}

/**
 * Scale the gradient values using normalized reward values and compute average.
 *
 * The gradient values are scaled by the normalized reward values. Then they
 * are averaged across all games and all steps.
 *
 * @param {{[varName: string]: tf.Tensor[][]}} allGradients A map from variable
 *   name to all the gradient values for the variable across all games and all
 *   steps.
 * @param {tf.Tensor[]} normalizedRewards An Array of normalized reward values
 *   for all the games. Each element of the Array is a 1D tf.Tensor of which
 *   the length equals the number of steps in the game.
 * @returns Scaled and averaged gradients for the variables.
 */
function scaleAndAverageGradients(
  allGradients: { [varName: string]: Array<tf.Tensor[]> },
  normalizedRewards: tf.Tensor[]
) {
  return tf.tidy(() => {
    const gradients: { [varName: string]: tf.Tensor } = {}
    for (const varName in allGradients) {
      gradients[varName] = tf.tidy(() => {
        // Stack gradients together.
        const varGradients = allGradients[varName].map((varGameGradients) =>
          tf.stack(varGameGradients)
        )
        // Expand dimensions of reward tensors to prepare for multiplication
        // with broadcasting.
        const expandedDims: Array<number> = []
        for (let i = 0; i < varGradients[0].rank - 1; ++i) {
          expandedDims.push(1)
        }
        const reshapedNormalizedRewards = normalizedRewards.map((rs) =>
          rs.reshape(rs.shape.concat(expandedDims))
        )
        for (let g = 0; g < varGradients.length; ++g) {
          // This mul() call uses broadcasting.
          varGradients[g] = varGradients[g].mul(reshapedNormalizedRewards[g])
        }
        // Concatenate the scaled gradients together, then average them across
        // all the steps of all the games.
        return tf.mean(tf.concat(varGradients, 0), 0)
      })
    }
    return gradients
  })
}
