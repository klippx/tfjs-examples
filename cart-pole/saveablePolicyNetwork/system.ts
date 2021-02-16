import * as tf from '@tensorflow/tfjs'

export abstract class System {
  abstract setRandomState(): void
  abstract getStateTensor(): tf.Tensor2D
  abstract update(action: number): boolean
}
