import { System } from './saveablePolicyNetwork/system'

/**
 * Abstract class definition that the simulated model must adhere to
 */
export abstract class RenderableSystem extends System {
  abstract render(canvas: HTMLCanvasElement): void
}
