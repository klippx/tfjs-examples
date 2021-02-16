import { System } from './saveablePolicyNetwork/system'

export abstract class RenderableSystem extends System {
  abstract render(canvas: HTMLCanvasElement): void
}
