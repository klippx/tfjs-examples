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

/**
 * Implementation based on: http://incompleteideas.net/book/code/pole.c
 */

import * as tf from '@tensorflow/tfjs'
import { RenderableSystem } from './renderableSystem'

/**
 * Cart-pole system simulator.
 *
 * In the control-theory sense, there are four state variables in this system:
 *
 *   - x: The 1D location of the cart.
 *   - xDot: The velocity of the cart.
 *   - theta: The angle of the pole (in radians). A value of 0 corresponds to
 *     a vertical position.
 *   - thetaDot: The angular velocity of the pole.
 *
 * The system is controlled through a single action:
 *
 *   - leftward or rightward force.
 */
export class CartPole implements RenderableSystem {
  // Constants that characterize the system.
  private gravity: number
  private massCart: number
  private massPole: number
  private totalMass: number
  private cartWidth: number
  private cartHeight: number
  private length: number
  private poleMoment: number
  private forceMag: number
  private tau: number

  // Threshold values, beyond which a simulation will be marked as failed.
  private xThreshold: number
  private thetaThreshold: number

  // The control-theory state variables of the cart-pole system.
  // Cart position, meters.
  private x: number
  // Cart velocity.
  private xDot: number
  // Pole angle, radians.
  private theta: number
  // Pole angle velocity.
  private thetaDot: number

  /**
   * Construct`or of this.
   */
  constructor() {
    // Constants that characterize the system.
    this.gravity = 9.8
    this.massCart = 1.0
    this.massPole = 0.1
    this.totalMass = this.massCart + this.massPole
    this.cartWidth = 0.2
    this.cartHeight = 0.1
    this.length = 0.5
    this.poleMoment = this.massPole * this.length
    this.forceMag = 10.0
    this.tau = 0.02 // Seconds between state updates.

    // Threshold values, beyond which a simulation will be marked as failed.
    this.xThreshold = 2.4
    this.thetaThreshold = (12 / 360) * 2 * Math.PI

    // The control-theory state variables of the cart-pole system.
    this.x = Math.random() - 0.5
    this.xDot = (Math.random() - 0.5) * 1
    this.theta = (Math.random() - 0.5) * 2 * ((6 / 360) * 2 * Math.PI)
    this.thetaDot = (Math.random() - 0.5) * 0.5
  }

  /**
   * Set the state of the cart-pole system randomly.
   */
  setRandomState() {
    // Cart position, meters.
    this.x = Math.random() - 0.5
    // Cart velocity.
    this.xDot = (Math.random() - 0.5) * 1
    // Pole angle, radians.
    this.theta = (Math.random() - 0.5) * 2 * ((6 / 360) * 2 * Math.PI)
    // Pole angle velocity.
    this.thetaDot = (Math.random() - 0.5) * 0.5
  }

  /**
   * Get current state as a tf.Tensor of shape [1, 4].
   */
  getStateTensor() {
    return tf.tensor2d([[this.x, this.xDot, this.theta, this.thetaDot]])
  }

  /**
   * Update the cart-pole system using an action.
   * @param action Only the sign of `action` matters.
   *   A value > 0 leads to a rightward force of a fixed magnitude.
   *   A value <= 0 leads to a leftward force of the same fixed magnitude.
   */
  update(action: number) {
    const force = action > 0 ? this.forceMag : -this.forceMag

    const cosTheta = Math.cos(this.theta)
    const sinTheta = Math.sin(this.theta)

    const temp =
      (force + this.poleMoment * this.thetaDot * this.thetaDot * sinTheta) /
      this.totalMass
    const thetaAcc =
      (this.gravity * sinTheta - cosTheta * temp) /
      (this.length *
        (4 / 3 - (this.massPole * cosTheta * cosTheta) / this.totalMass))
    const xAcc = temp - (this.poleMoment * thetaAcc * cosTheta) / this.totalMass

    // Update the four state variables, using Euler's metohd.
    this.x += this.tau * this.xDot
    this.xDot += this.tau * xAcc
    this.theta += this.tau * this.thetaDot
    this.thetaDot += this.tau * thetaAcc

    return this.isDone()
  }

  /**
   * Determine whether this simulation is done.
   *
   * A simulation is done when `x` (position of the cart) goes out of bound
   * or when `theta` (angle of the pole) goes out of bound.
   *
   * @returns Whether the simulation is done.
   */
  isDone() {
    return (
      this.x < -this.xThreshold ||
      this.x > this.xThreshold ||
      this.theta < -this.thetaThreshold ||
      this.theta > this.thetaThreshold
    )
  }

  /**
   * Render the current state of the system on an HTML canvas.
   *
   * @param {HTMLCanvasElement} canvas The instance of HTMLCanvasElement on which
   *   the rendering will happen.
   */
  render(canvas: HTMLCanvasElement) {
    if (!canvas.style.display) {
      canvas.style.display = 'block'
    }
    const X_MIN = -this.xThreshold
    const X_MAX = this.xThreshold
    const xRange = X_MAX - X_MIN
    const scale = canvas.width / xRange

    const context = canvas.getContext('2d')
    if (context === null) {
      throw new Error('Where did the canvas go?')
    }
    context.clearRect(0, 0, canvas.width, canvas.height)
    const halfW = canvas.width / 2

    // Draw the cart.
    const railY = canvas.height * 0.8
    const cartW = this.cartWidth * scale
    const cartH = this.cartHeight * scale

    const cartX = this.x * scale + halfW

    context.beginPath()
    context.strokeStyle = '#000000'
    context.lineWidth = 2
    context.rect(cartX - cartW / 2, railY - cartH / 2, cartW, cartH)
    context.stroke()

    // Draw the wheels under the cart.
    const wheelRadius = cartH / 4
    for (const offsetX of [-1, 1]) {
      context.beginPath()
      context.lineWidth = 2
      context.arc(
        cartX - (cartW / 4) * offsetX,
        railY + cartH / 2 + wheelRadius,
        wheelRadius,
        0,
        2 * Math.PI
      )
      context.stroke()
    }

    // Draw the pole.
    const angle = this.theta + Math.PI / 2
    const poleTopX = halfW + scale * (this.x + Math.cos(angle) * this.length)
    const poleTopY =
      railY - scale * (this.cartHeight / 2 + Math.sin(angle) * this.length)
    context.beginPath()
    context.strokeStyle = '#ffa500'
    context.lineWidth = 6
    context.moveTo(cartX, railY - cartH / 2)
    context.lineTo(poleTopX, poleTopY)
    context.stroke()

    // Draw the ground.
    const groundY = railY + cartH / 2 + wheelRadius * 2
    context.beginPath()
    context.strokeStyle = '#000000'
    context.lineWidth = 1
    context.moveTo(0, groundY)
    context.lineTo(canvas.width, groundY)
    context.stroke()

    const nDivisions = 40
    for (let i = 0; i < nDivisions; ++i) {
      const x0 = (canvas.width / nDivisions) * i
      const x1 = x0 + canvas.width / nDivisions / 2
      const y0 = groundY + canvas.width / nDivisions / 2
      const y1 = groundY
      context.beginPath()
      context.moveTo(x0, y0)
      context.lineTo(x1, y1)
      context.stroke()
    }

    // Draw the left and right limits.
    const limitTopY = groundY - canvas.height / 2
    context.beginPath()
    context.strokeStyle = '#ff0000'
    context.lineWidth = 2
    context.moveTo(1, groundY)
    context.lineTo(1, limitTopY)
    context.stroke()
    context.beginPath()
    context.moveTo(canvas.width - 1, groundY)
    context.lineTo(canvas.width - 1, limitTopY)
    context.stroke()
  }
}
