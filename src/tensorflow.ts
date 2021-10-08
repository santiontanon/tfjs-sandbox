/*
This file is a hack. It just defines a fake API for Tensorflow, 
so that I can use the TypeScript compiler, without installing the
Tensorflow.js package, and hence, I do not need any package manager,
nor additional dependencies.
*/

declare module tf {

  export type Kwargs = {
    // tslint:disable-next-line:no-any
    [key: string]: any
  };

  export class Tensor {
    array();
    print();

    shape:number[];
  }
  export class Tensor1D extends Tensor {
  }
  export class Tensor2D extends Tensor {
  }
  export class Tensor3D extends Tensor {
  }
  export class Tensor4D extends Tensor {
  }
  export class SymbolicTensor {
    apply(inputs: Tensor|Tensor[]|SymbolicTensor|SymbolicTensor[],
          kwargs?: Kwargs): Tensor|Tensor[]|SymbolicTensor|SymbolicTensor[];
    countParams();
    shape:number[];
  }
  export class LayersModel {
    compile(kwargs);
    fit(x, y, kwargs);
    predict(tensor);
    getWeights();
  }
  export class History {
    epoch;
    history;
  }


  export function setBackend(name:string);
  export function tidy(f);

  export function scalar(array: any, dtype?: string): Tensor
  export function tensor(array: any, shape:any): Tensor;
  export function tensor1d(array: any): Tensor1D;
  export function tensor2d(array: any): Tensor2D;
  export function tensor2d(array: any, shape: any, dtype?: string): Tensor2D;
  export function tensor3d(array: any): Tensor3D;
  export function tensor3d(array: any, shape: any): Tensor3D;
  export function tensor4d(array: any): Tensor3D;
  export function tensor4d(array: any, shape: any): Tensor3D;

  export function cast(array: tf.Tensor, dtype: string): tf.Tensor;
  export function expandDims(array: tf.Tensor, axis: number): tf.Tensor;
  export function squeeze(array: tf.Tensor, axis?: number): tf.Tensor;
  export function reshape(array: tf.Tensor, shape: number[]): tf.Tensor;
  export function transpose(array: tf.Tensor, perm?: number[]): tf.Tensor;
  export function slice(array: tf.Tensor, begin:number[], size:number[]): tf.Tensor;
  export function gather(x: tf.Tensor, indices: tf.Tensor, axis?:number): tf.Tensor;

  export function ones(shape: number[]): tf.Tensor;
  export function ones(shape: number[], dtype: string): tf.Tensor;

  export function add(a1: tf.Tensor, a2: tf.Tensor): tf.Tensor;
  export function sub(a1: tf.Tensor, a2: tf.Tensor): tf.Tensor;
  export function div(a1: tf.Tensor, a2: tf.Tensor): tf.Tensor;
  export function mul(a1: tf.Tensor, a2: tf.Tensor, axis?:number): tf.Tensor;
  export function sqrt(a1: tf.Tensor): tf.Tensor;
  export function sum(a1: tf.Tensor, axis?:number): tf.Tensor;
  export function mean(a1: tf.Tensor, axis?:number): tf.Tensor;
  export function logicalNot(a1: tf.Tensor): tf.Tensor;
  export function argMax(a1: tf.Tensor, axis?:number): tf.Tensor;
  export function equal(a1: tf.Tensor, a2: tf.Tensor): tf.Tensor;
  export function greaterEqual(a1: tf.Tensor, a2: tf.Tensor): tf.Tensor;
  export function maximum(a1: tf.Tensor, a2: tf.Tensor): tf.Tensor;
  export function matMul(a: tf.Tensor, b:tf.Tensor, transposeA?: boolean, transposeB?: boolean): tf.Tensor;
  export function einsum(equation: string, ...tensors): tf.Tensor;
  export function softmax(a: tf.Tensor, axis?: number): tf.Tensor;
  export function oneHot(a: tf.Tensor, depth:number): tf.Tensor;

  export function input(config: any): SymbolicTensor;
  export function model(config: any): LayersModel;
  module layers {
    export function dense(config: any): tf.layers.Layer;
    export function embedding(config: any): tf.layers.Layer;
    export function dropout(config: any): tf.layers.Layer;
    export function layerNormalization(config: any): tf.layers.Layer;
    export function add(config: any): tf.layers.Layer;
    export function softmax(config: any): tf.layers.Layer;
    export function flatten(): tf.layers.Layer;

    export class Layer {
      built:boolean;
      constructor(kwargs);
      apply(inputs: Tensor|Tensor[]|SymbolicTensor|SymbolicTensor[],
            kwargs?: Kwargs): Tensor|Tensor[]|SymbolicTensor|SymbolicTensor[];
      addWeight(name, shape, dtype?, initializer?, regularizer?, trainable?, constraint?);
      countParams();
    }
  }
  module linalg {
    export function bandPart(x:tf.Tensor): tf.Tensor;
    export function bandPart(x:tf.Tensor, numLower:number, numUpper:number): tf.Tensor;
  }
  module train {
    export function sgd(learningRate:number);
  }
  module metrics {
    export function categoricalCrossentropy(yTrue:tf.Tensor, yPred:tf.Tensor): tf.Tensor;
    export function categoricalAccuracy(yTrue:tf.Tensor, yPred:tf.Tensor): tf.Tensor;
  }
  module initializers {
    class Initializer {

    }
    export function randomUniform(args): tf.initializers.Initializer;
  }
}
