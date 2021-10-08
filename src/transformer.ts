/*
The transformer model defined in this file is basically adapted from the 
Tensorflow Transformer tutorial here: https://www.tensorflow.org/text/tutorials/transformer
Additionally, relative attention and a copy decoder (following this paper: https://arxiv.org/abs/2108.04378)
have been added.
*/


function isNumeric(value) {
    return /^-?\d+$/.test(value);
}


/*
Auxiliary function to "positionalEncoding" below.
*/
function getAngles(
    pos:number[][],
    i:number[][],
    dModel:number) : number[][]
{
  for(let idx2:number = 0;idx2<i[0].length;idx2++) {
    i[0][idx2] = 1.0 / Math.pow(10000.0, 2 * Math.floor(i[0][idx2]/2) / dModel);
  }
  let angles:number[][] = [];
  for(let idx:number = 0;idx<pos.length;idx++) {
    angles.push([]);
    for(let idx2:number = 0;idx2<i[0].length;idx2++) {
      angles[idx].push(pos[idx][0] * i[0][idx2]);
    }
  }
  return angles
}


/*
Generates absolute position encodings of dimensionality "dModel" for
sequences of length "seqLen".
*/
function positionalEncoding(seqLen:number, dModel:number) : tf.Tensor3D
{
  let arg1:number[][] = [];
  let arg2:number[][] = [[]];
  for(let i:number = 0;i<seqLen;i++) arg1.push([i]);
  for(let i:number = 0;i<dModel;i++) arg2[0].push(i);
  let angleRads:number[][] = getAngles(arg1, arg2, dModel);

  // Even indices of the array contain "sin", odd contain "cos":
  for(let i:number = 0;i<angleRads.length;i++) {
    for(let j:number = 0;j<angleRads[i].length;j++) {
      if ((j % 2) == 0) {
        // Even:
        angleRads[i][j] = Math.sin(angleRads[i][j]);
      } else {
        // Odd:
        angleRads[i][j] = Math.cos(angleRads[i][j]);
      }
    }
  }
    
  let posEncoding:number[][][] = [];
  posEncoding.push(angleRads);
  return tf.tensor3d(posEncoding, [1, seqLen, dModel]);
}


function createPaddingMask(seq:tf.Tensor2D) : tf.Tensor4D
{
  seq = tf.cast(tf.equal(seq, tf.tensor1d([0])), "float32");
  // Add extra dimensions to add the padding to the attention logits.
  return tf.expandDims(tf.expandDims(seq, 1), 1);  // (batchSize, 1, 1, seqLen)
}


function createLookAheadMask(seqLen:number) : tf.Tensor2D
{
  let mask:tf.Tensor2D = tf.sub(tf.tensor1d([1]), 
                                tf.linalg.bandPart(tf.ones([seqLen, seqLen]), -1, 0));
  return mask;  // (seqLen, seqLen)
}


function createMasks(inp:tf.Tensor2D, tar:tf.Tensor2D) : [tf.Tensor4D, tf.Tensor4D, tf.Tensor4D]
{
  // Encoder padding mask
  let encPaddingMask:tf.Tensor4D = createPaddingMask(inp);
  
  // Used in the 2nd attention block in the decoder.
  // This padding mask is used to mask the encoder outputs.
  let decPaddingMask:tf.Tensor4D = createPaddingMask(inp);
  
  // Used in the 1st attention block in the decoder.
  // It is used to pad and mask future tokens in the input received by 
  // the decoder.
  let lookAheadMask:tf.Tensor2D = createLookAheadMask(tar.shape[1]);
  let decTargetPaddingMask:tf.Tensor4D = createPaddingMask(tar);
  let combinedMask:tf.Tensor4D = tf.maximum(decTargetPaddingMask, lookAheadMask);
  
  return [encPaddingMask, combinedMask, decPaddingMask];

}


function createRelativeIds(
  inpLen:number,
  tarLen:number,
  relativeRadius:number,
  dec2endIds:boolean) : [tf.Tensor2D, tf.Tensor2D, tf.Tensor2D]
{
  let encRelativeIds:number[][] = [];
  for(let i:number = 0;i<inpLen;i++) {
    encRelativeIds.push([]);
    for(let j:number = 0;j<inpLen;j++) {
      let diff:number = i - j
      diff = relativeRadius + Math.min(Math.max(diff, -relativeRadius), relativeRadius)
      encRelativeIds[i].push(diff);
    }
  }
  
  let decRelativeIds:number[][] = [];
  for(let i:number = 0;i<tarLen-1;i++) {
    decRelativeIds.push([]);
    for(let j:number = 0;j<tarLen-1;j++) {
      let diff:number = i - j
      diff = relativeRadius + Math.min(Math.max(diff, -relativeRadius), relativeRadius)
      decRelativeIds[i].push(diff);
    }
  }

  let dec2EncRelativeIds:number[][] = [];
  for(let i:number = 0;i<tarLen-1;i++) {
    dec2EncRelativeIds.push([]);
    for(let j:number = 0;j<inpLen;j++) {
      if (dec2endIds) {
        let diff:number = i - j
        diff = relativeRadius + Math.min(Math.max(diff, -relativeRadius), relativeRadius)
        dec2EncRelativeIds[i].push(diff);
      } else {
        dec2EncRelativeIds[i].push(relativeRadius);
      }
    }
  }

  return [tf.tensor2d(encRelativeIds, [inpLen, inpLen], "int32"),
          tf.tensor2d(decRelativeIds, [tarLen-1, tarLen-1], "int32"), 
          tf.tensor2d(dec2EncRelativeIds, [tarLen-1, inpLen], "int32")];
}


class CreateMasksLayer extends tf.layers.Layer {
  maxInputLength:number;
  maxTargetLength:number;

  constructor(config) {
    super(config);
    this.maxInputLength = config.maxInputLength;
    this.maxTargetLength = config.maxTargetLength;
  }

  computeOutputShape(inputShape) {
    return [[1, 1, 1, this.maxInputLength],
            [1, 1, this.maxTargetLength, this.maxTargetLength],
            [1, 1, 1, this.maxTargetLength]];
  }

  call(x) {
    return tf.tidy(() => {
      return createMasks(x[0], x[1]);
    });
  }

  getClassName() { return "CreateMasksLayer"; }
}


class OneHotLayer extends tf.layers.Layer {
  depth:number;

  constructor(config) {
    super(config)
    this.depth = config.depth;
  }

  computeOutputShape(inputShape) {
    let outputShape:number[] = [];
    for(let d of inputShape) {
      outputShape.push(d);
    }
    outputShape.push(this.depth);
    return outputShape;
  }

  call(x) { 
    return tf.tidy(() => {
      return tf.oneHot(x[0], this.depth);
    });
  }

  getClassName() { return "OneHotLayer"; }  
}


/*
Calculate the attention weights.
- If relativeIds or relativeEmbeddings is None, then this is equivalent to
  regular scaled dot product attention (without relative attention).
- q, k, v must have matching leading dimensions.
- k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
- The mask has different shapes depending on its type(padding or look ahead) 
  but it must be broadcastable for addition.
- relativeIds is an optional [seq_len_q, seq_len_k] with the relative ids.
- relativeEmbeddings is an optional dense layer that converts OHE ids to
  embeddings.
- relativeBiases: optional dense layer that generates a bias term for the
  attention logits based on the relative_ids

Args:
  q: query shape == (..., seq_len_q, depth)
  k: key shape == (..., seq_len_k, depth)
  v: value shape == (..., seq_len_v, depth_v)
  mask: Float tensor with shape broadcastable 
        to (..., seq_len_q, seq_len_k). Defaults to None.
  relative_ids == (seq_len_q, seq_len_k). Defaults to None.
  
Returns:
  output, attentionWeights
*/
class ScaledDotProductRelativeAttention extends tf.layers.Layer {
  useRelativePositionBiases:boolean = false;
  useRelativePositionEmbeddings:boolean = false;
  relativeVocabSize:number;
  depth:number;
  relativeEmbeddings = undefined;
  relativeBiases = undefined;
  relativeIds:tf.Tensor = undefined;
  useMask:boolean = true;

  constructor(config) {
    super(config)
    if ("relativeIds" in config) {
      this.relativeIds = config.relativeIds;
    }
    if ("mask" in config) {
      this.useMask = config.mask;
    }
    if ("useRelativePositionBiases" in config) {
      this.useRelativePositionBiases = config.useRelativePositionBiases;
    }
    if ("useRelativePositionEmbeddings" in config) {
      this.useRelativePositionEmbeddings = config.useRelativePositionEmbeddings;
    }
    if (this.useRelativePositionBiases) {
      this.relativeVocabSize = config.relativeVocabSize;
    }
    if (this.useRelativePositionEmbeddings) {
      this.relativeVocabSize = config.relativeVocabSize;
      this.depth = config.depth;
    }
  }

  computeOutputShape(inputShape) {
    // console.log("ScaledDotProductRelativeAttention.inputShape:" + inputShape);
    let seq_len_q:number = inputShape[0][inputShape[0].length-2];
    let seq_len_k:number = inputShape[1][inputShape[1].length-2];
    let depth_v:number = inputShape[2][inputShape[1].length-1];
    let output_shape:number[] = [];
    let scaledAttentionLogits_shape:number[] = [];
    for(let i:number = 0;i<inputShape[0].length-2;i++) {
      output_shape.push(inputShape[0][i]);
      scaledAttentionLogits_shape.push(inputShape[0][i]);
    }
    output_shape.push(seq_len_q);
    output_shape.push(depth_v);
    scaledAttentionLogits_shape.push(seq_len_q);
    scaledAttentionLogits_shape.push(seq_len_k);

    return [output_shape, scaledAttentionLogits_shape];
  }

  public build(inputShape): void {
    if (this.useRelativePositionBiases) {
      this.relativeBiases = this.addWeight(
          "relativeBiases", [this.relativeVocabSize, 1], "float32", tf.initializers.randomUniform({}));
    }
    if (this.useRelativePositionEmbeddings) {
      this.relativeEmbeddings = this.addWeight(
          "relativeEmbeddings", [this.relativeVocabSize, this.depth], "float32", tf.initializers.randomUniform({}));
    }
    this.built = true;
  }

  call(input)
  {
    return tf.tidy(() => {
      let q:tf.Tensor = input[0];
      let k:tf.Tensor = input[1];
      let v:tf.Tensor = input[2];

      let matmul_qk:tf.Tensor = tf.matMul(q, k, false, true);  // (..., seq_len_q, seq_len_k)
      
      if (this.relativeIds != undefined) {
        if (this.relativeEmbeddings != undefined) {
          let r = tf.gather(this.relativeEmbeddings.read(), this.relativeIds);

          // let matmul_qrel:tf.Tensor = tf.einsum("bhqd,qkd->bhqk", q, r);
          let q2 = tf.reshape(q, [q.shape[0], q.shape[1], q.shape[2], 1, q.shape[3]]);
          let matmul_qrel = tf.sum(tf.mul(q2, r, -1), -1);

          matmul_qk  = tf.add(matmul_qk, matmul_qrel);
        }
        if (this.relativeBiases != undefined) {
          matmul_qk = tf.add(matmul_qk, tf.squeeze(tf.gather(this.relativeBiases.read(), this.relativeIds), -1));
        }
      }
      
      // Scale matmul_qk
      let dk:tf.Tensor = tf.cast(tf.tensor1d([k.shape[k.shape.length-1]]), "float32");
      let scaledAttentionLogits:tf.Tensor = tf.div(matmul_qk, tf.sqrt(dk));

      // Add the mask to the scaled tensor.
      if (this.useMask) {
        let mask:tf.Tensor = input[3];
        scaledAttentionLogits = tf.add(scaledAttentionLogits, tf.mul(mask, tf.tensor1d([-1e9])));
      }

      // Softmax is normalized on the last axis (seq_len_k) so that the scores
      // add up to 1.
      let attentionWeights:tf.Tensor = tf.softmax(scaledAttentionLogits, -1);  // (..., seq_len_q, seq_len_k)
      let output:tf.Tensor = tf.matMul(attentionWeights, v);  // (..., seq_len_q, depth_v)

      return [output, scaledAttentionLogits];
    });
  }

  getClassName() { return "ScaledDotProductRelativeAttention"; }  
}


class SplitHeads extends tf.layers.Layer {
  numHeads:number;
  depth:number;

  constructor(config) {
    super(config)
    this.numHeads = config.numHeads;
    this.depth = config.depth;
  }

  computeOutputShape(inputShape) {
    return [inputShape[0], this.numHeads, inputShape[1], this.depth];
  }

  /*
    Split the last dimension into (numHeads, depth).
    Transpose the result such that the shape is (batchSize, numHeads, seq_len, depth)
  */
  call(inputs) { 
    return tf.tidy(() => {
      let x = inputs[0];
      let batchSize:number = x.shape[0];
      x = tf.reshape(x, [batchSize, -1, this.numHeads, this.depth]);
      return tf.transpose(x, [0, 2, 1, 3]);
    });
  }

  getClassName() { return "SplitHeads"; }  
}


class MultiHeadAttentionReshape extends tf.layers.Layer {
  constructor(config) {
    super(config);
  }

  computeOutputShape(inputShape) {
    return [inputShape[0], inputShape[2], inputShape[1]*inputShape[3]];
  }

  call(inputs) {
    return tf.tidy(() => {
      let scaledAttention = inputs[0];
      let shape:number[] = scaledAttention.shape;
      scaledAttention = tf.transpose(scaledAttention, [0, 2, 1, 3]);  // (batchSize, seq_len_q, numHeads, depth)
      let concatAttention = tf.reshape(scaledAttention, [shape[0], -1, shape[1]*shape[3]])  // (batchSize, seq_len_q, dModel)
      return concatAttention;
    });
  }

  getClassName() { return "MultiHeadAttentionAuxiliar1";}
}


class MultiHeadAttention {
  numHeads:number;
  dModel:number;
  relativeVocabSize:number;
  depth:number;
  wq:tf.layers.Layer;
  wk:tf.layers.Layer;
  wv:tf.layers.Layer;
  dense:tf.layers.Layer;

  splitHeads:tf.layers.Layer;
  scaledDotProductRelativeAttention:tf.layers.Layer;
  multiHeadAttentionReshape:tf.layers.Layer;

  constructor(
    dModel:number,
    numHeads:number,
    relativeRadius:number,
    useRelativePositionBiases:boolean,
    useRelativePositionEmbeddings:boolean,
    useAbsolutePositionEncodings:boolean,
    relativeIds:tf.Tensor,
    namePrefix:string) 
  {
    this.dModel = dModel
    this.numHeads = numHeads
    this.relativeVocabSize = relativeRadius*2+1    
    this.depth = Math.floor(this.dModel / this.numHeads);

    this.wq = tf.layers.dense({units: this.dModel, name:namePrefix+"-mha-wq"});
    this.wk = tf.layers.dense({units: this.dModel, name:namePrefix+"-mha-wk"});
    this.wv = tf.layers.dense({units: this.dModel, name:namePrefix+"-mha-wv"});
    this.dense = tf.layers.dense({units: this.dModel, name:namePrefix+"-mha-dense"})

    this.splitHeads = new SplitHeads({numHeads:this.numHeads, depth:this.depth});
    this.scaledDotProductRelativeAttention = new ScaledDotProductRelativeAttention({useRelativePositionBiases:useRelativePositionBiases, 
                                                                                    useRelativePositionEmbeddings:useRelativePositionEmbeddings,
                                                                                    relativeIds:relativeIds,
                                                                                    mask:true,
                                                                                    relativeVocabSize:this.relativeVocabSize,
                                                                                    depth:this.depth,
                                                                                    namePrefix:namePrefix+"-mha-internal",
                                                                                    name:namePrefix+"-mha-internal"});
    this.multiHeadAttentionReshape = new MultiHeadAttentionReshape({});
  }

  call(v: tf.Tensor, k: tf.Tensor, q: tf.Tensor, mask:tf.Tensor) : [tf.Tensor, tf.Tensor]
  {
    q = <tf.Tensor>this.wq.apply(q)  // (batchSize, seq_len, dModel)
    k = <tf.Tensor>this.wk.apply(k)  // (batchSize, seq_len, dModel)
    v = <tf.Tensor>this.wv.apply(v)  // (batchSize, seq_len, dModel)

    q = <tf.Tensor>this.splitHeads.apply(q)  // (batchSize, numHeads, seq_len_q, depth)
    k = <tf.Tensor>this.splitHeads.apply(k)  // (batchSize, numHeads, seq_len_k, depth)
    v = <tf.Tensor>this.splitHeads.apply(v)  // (batchSize, numHeads, seq_len_v, depth)

    let tmp:tf.Tensor[] = <tf.Tensor[]>this.scaledDotProductRelativeAttention.apply([q, k, v, mask]);
    let scaledAttention:tf.Tensor = tmp[0];  // (batchSize, numHeads, seq_len_q, depth)
    let attentionLogits:tf.Tensor = tmp[1];
    let concatAttention:tf.Tensor = <tf.Tensor>this.multiHeadAttentionReshape.apply(scaledAttention);
    let output:tf.Tensor = <tf.Tensor>this.dense.apply(concatAttention)  // (batchSize, seq_len_q, dModel)        
    return [output, attentionLogits]
  }
}


class PointWiseFeedForwardNetwork {
  dense1:tf.layers.Layer;
  dense2:tf.layers.Layer;

  constructor(dModel:number, dFF:number, namePrefix:string)
  {
    this.dense1 = tf.layers.dense({units: dFF, activation: "relu", name:namePrefix+"-ffn1"});  // (batchSize, seq_len, dff)
    this.dense2 = tf.layers.dense({units: dModel, name:namePrefix+"-ffn2"});  // # (batchSize, seq_len, dModel)
  }

  call(x:tf.Tensor) : tf.Tensor
  {
    return <tf.Tensor>this.dense2.apply(this.dense1.apply(x));
  }
}


class EncoderLayer {
  mha:MultiHeadAttention;
  ffn:PointWiseFeedForwardNetwork;
  layernorm1:tf.layers.Layer;
  layernorm2:tf.layers.Layer;
  layernorm1add:tf.layers.Layer;
  layernorm2add:tf.layers.Layer;
  dropout1:tf.layers.Layer;
  dropout2:tf.layers.Layer;

  constructor(
    dModel:number,
    numHeads:number,
    dFF:number, 
    dropoutRate:number, 
    relativeRadius:number,
    useRelativePositionBiases:boolean,
    useRelativePositionEmbeddings:boolean,
    useAbsolutePositionEncodings:boolean,
    relativeIds:tf.Tensor,
    namePrefix:string)
  {
    this.mha = new MultiHeadAttention(dModel, numHeads, relativeRadius, 
      useRelativePositionBiases, useRelativePositionEmbeddings,
      useAbsolutePositionEncodings, relativeIds, namePrefix);
    this.ffn = new PointWiseFeedForwardNetwork(dModel, dFF, namePrefix);

    this.layernorm1 = tf.layers.layerNormalization({epsilon:1e-6, name:namePrefix+"-layernorm1"});
    this.layernorm2 = tf.layers.layerNormalization({epsilon:1e-6, name:namePrefix+"-layernorm2"});
    this.layernorm1add = tf.layers.add({});
    this.layernorm2add = tf.layers.add({});
    
    this.dropout1 = tf.layers.dropout({rate:dropoutRate});
    this.dropout2 = tf.layers.dropout({rate:dropoutRate});
  }

  call(
    x: tf.Tensor, 
    mask:tf.Tensor) : [tf.Tensor, tf.Tensor]
  {
    let mhaOut:[tf.Tensor, tf.Tensor] = 
      this.mha.call(x, x, x, mask);
    let attnOutput:tf.Tensor = mhaOut[0];
    let attentionLogits:tf.Tensor = mhaOut[1];
    attnOutput = <tf.Tensor>this.dropout1.apply(attnOutput);
    let out1:tf.Tensor = <tf.Tensor>this.layernorm1.apply(this.layernorm1add.apply([x, attnOutput]));  // (batchSize, input_seq_len, dModel)
    
    let ffnOutput:tf.Tensor = this.ffn.call(out1);  // (batchSize, input_seq_len, dModel)
    ffnOutput = <tf.Tensor>this.dropout2.apply(ffnOutput)
    let out2:tf.Tensor = <tf.Tensor>this.layernorm2.apply(this.layernorm2add.apply([out1, ffnOutput]));  // (batchSize, input_seq_len, dModel)
    
    return [out2, attentionLogits];
  }
}


class DecoderLayer {
  mha1:MultiHeadAttention;
  mha2:MultiHeadAttention;
  ffn:PointWiseFeedForwardNetwork;
  layernorm1:tf.layers.Layer;
  layernorm2:tf.layers.Layer;
  layernorm3:tf.layers.Layer;
  layernorm1add:tf.layers.Layer;
  layernorm2add:tf.layers.Layer;
  layernorm3add:tf.layers.Layer;
  dropout1:tf.layers.Layer;
  dropout2:tf.layers.Layer;
  dropout3:tf.layers.Layer;

  constructor(
    dModel:number,
    numHeads:number,
    dFF:number, 
    dropoutRate:number, 
    relativeRadius:number,
    useRelativePositionBiases:boolean,
    useRelativePositionEmbeddings:boolean,
    useAbsolutePositionEncodings:boolean,
    dec_relativeIds:tf.Tensor,
    dec2enc_relativeIds:tf.Tensor,
    namePrefix:string)
  {
    this.mha1 = new MultiHeadAttention(dModel, numHeads, relativeRadius, 
      useRelativePositionBiases, useRelativePositionEmbeddings,
      useAbsolutePositionEncodings, dec_relativeIds, namePrefix+"-dec2dec");
    this.mha2 = new MultiHeadAttention(dModel, numHeads, relativeRadius, 
      useRelativePositionBiases, useRelativePositionEmbeddings,
      useAbsolutePositionEncodings, dec2enc_relativeIds, namePrefix+"-dec2enc");

    this.ffn = new PointWiseFeedForwardNetwork(dModel, dFF, namePrefix);

    this.layernorm1 = tf.layers.layerNormalization({epsilon:1e-6, name:namePrefix+"-layernorm1"});
    this.layernorm2 = tf.layers.layerNormalization({epsilon:1e-6, name:namePrefix+"-layernorm2"});
    this.layernorm3 = tf.layers.layerNormalization({epsilon:1e-6, name:namePrefix+"-layernorm3"});
    this.layernorm1add = tf.layers.add({});
    this.layernorm2add = tf.layers.add({});
    this.layernorm3add = tf.layers.add({});
    
    this.dropout1 = tf.layers.dropout({rate:dropoutRate});
    this.dropout2 = tf.layers.dropout({rate:dropoutRate});
    this.dropout3 = tf.layers.dropout({rate:dropoutRate});
  }

  call(
    x: tf.Tensor, 
    encOutput: tf.Tensor,
    lookAheadMask:tf.Tensor,
    paddingMask:tf.Tensor) : [tf.Tensor, tf.Tensor, tf.Tensor]
  {
    let mhaOut1:[tf.Tensor, tf.Tensor] = 
      this.mha1.call(x, x, x, lookAheadMask);
    let attn1:tf.Tensor = mhaOut1[0];
    let attentionLogitsBlock1:tf.Tensor = mhaOut1[1];
    attn1 = <tf.Tensor>this.dropout1.apply(attn1);
    let out1:tf.Tensor = <tf.Tensor>this.layernorm1.apply(this.layernorm1add.apply([attn1, x]));  // (batchSize, input_seq_len, dModel)


    let mhaOut2:[tf.Tensor, tf.Tensor] = 
      this.mha2.call(encOutput, encOutput, out1, paddingMask);
    let attn2:tf.Tensor = mhaOut2[0];
    let attentionLogitsBlock2:tf.Tensor = mhaOut2[1];
    attn2 = <tf.Tensor>this.dropout2.apply(attn2);
    let out2:tf.Tensor = <tf.Tensor>this.layernorm2.apply(this.layernorm2add.apply([attn2, out1]));  // (batchSize, target_seq_len, dModel)
    
    let ffnOutput:tf.Tensor = this.ffn.call(out2)  // (batchSize, target_seq_len, dModel)
    ffnOutput = <tf.Tensor>this.dropout3.apply(ffnOutput);
    let out3:tf.Tensor = <tf.Tensor>this.layernorm3.apply(this.layernorm3add.apply([ffnOutput, out2]));  // (batchSize, target_seq_len, dModel)
    
    return [out3, attentionLogitsBlock1, attentionLogitsBlock2];
  }
}


class EncoderAuxiliarLayer1 extends tf.layers.Layer {
  useAbsolutePositionEncodings:boolean;
  dModel:number;
  posEncoding:tf.Tensor;

  constructor(config) {
    super(config);
    this.useAbsolutePositionEncodings = config.useAbsolutePositionEncodings;
    this.dModel = config.dModel;
    this.posEncoding = config.posEncoding;
  }

  computeOutputShape(inputShape) { 
    return inputShape; 
  }

  call(inputs) { 
    return tf.tidy(() => {
      let x = tf.mul(inputs[0], tf.sqrt(tf.scalar(this.dModel, "float32")));
      if (this.useAbsolutePositionEncodings) {
        let posEncodingSlice:tf.Tensor = tf.slice(this.posEncoding, [0,0,0], [-1, x.shape[1], -1]);
        x = tf.add(x, posEncodingSlice);
      }
      return x;
    });
  }

  getClassName() { return "EncoderAuxiliarLayer1"; }
}


class Encoder {
  dModel:number;
  numLayers:number;
  useAbsolutePositionEncodings:boolean;
  sharedLayers:boolean;

  embedding:tf.layers.Layer;
  dropout:tf.layers.Layer;
  encoderLayers:EncoderLayer[] = [];

  encoderAuxiliarLayer1:tf.layers.Layer;

  constructor(
    numLayers:number, 
    dModel:number,
    numHeads:number, 
    dFF:number, 
    inputVocabSize:number,
    maximumPositionEncoding:number, 
    dropoutRate:number,
    relativeRadius:number,
    useRelativePositionBiases:boolean,
    useRelativePositionEmbeddings:boolean,
    useAbsolutePositionEncodings:boolean,
    sharedLayers:boolean,
    posEncoding:tf.Tensor,
    enc_relativeIds:tf.Tensor)
  {
    this.dModel = dModel;
    this.numLayers = numLayers;
    this.useAbsolutePositionEncodings = useAbsolutePositionEncodings;
    this.sharedLayers = sharedLayers;
    
    this.embedding = tf.layers.embedding({inputDim:inputVocabSize, outputDim:dModel, name:"encoder-embedding"});
        
    if (this.sharedLayers) {
      let layer:EncoderLayer = new EncoderLayer(
        dModel, numHeads, dFF, dropoutRate, 
        relativeRadius,
        useRelativePositionBiases,
        useRelativePositionEmbeddings,
        useAbsolutePositionEncodings,
        enc_relativeIds,
        "encoder-layer");
      for(let i:number = 0;i<numLayers;i++) {
        this.encoderLayers.push(layer);
      }
    } else {
      for(let i:number = 0;i<numLayers;i++) {
        this.encoderLayers.push(new EncoderLayer(
          dModel, numHeads, dFF, dropoutRate, 
          relativeRadius,
          useRelativePositionBiases,
          useRelativePositionEmbeddings,
          useAbsolutePositionEncodings,
          enc_relativeIds,
          "encoder-layer-"+i));          
      }
    }
  
    this.dropout = tf.layers.dropout({rate:dropoutRate});

    this.encoderAuxiliarLayer1 = new EncoderAuxiliarLayer1({useAbsolutePositionEncodings:useAbsolutePositionEncodings,
                                                            dModel:dModel,
                                                            posEncoding:posEncoding});
  }

  call(x, mask:tf.Tensor) : tf.Tensor /*[tf.Tensor, tf.Tensor[]]*/
  {    
    // adding embedding and position encoding.
    x = this.embedding.apply(x)  // (batchSize, input_seq_len, dModel)
    x = this.encoderAuxiliarLayer1.apply(x);
    x = <tf.Tensor>this.dropout.apply(x);

    for(let i:number=0;i<this.numLayers;i++) {
      let tmp:[tf.Tensor, tf.Tensor] = this.encoderLayers[i].call(x, mask);
      x = tmp[0];
    }
    return x;  // x: (batchSize, input_seq_len, dModel)
  }
}


class Decoder
{
  dModel:number;
  numLayers:number;
  useAbsolutePositionEncodings:boolean;
  sharedLayers:boolean;
  posEncoding:tf.Tensor;

  embedding:tf.layers.Layer;
  dropout:tf.layers.Layer;
  decoderLayers:DecoderLayer[] = [];

  decoderAuxiliarLayer1:tf.layers.Layer;

  constructor(
    numLayers:number, 
    dModel:number,
    numHeads:number, 
    dFF:number, 
    targetVocabSize:number,
    maximumPositionEncoding:number, 
    dropoutRate:number,
    relativeRadius:number,
    useRelativePositionBiases:boolean,
    useRelativePositionEmbeddings:boolean,
    useAbsolutePositionEncodings:boolean,
    sharedLayers:boolean,
    posEncoding:tf.Tensor,
    dec_relativeIds:tf.Tensor,
    dec2enc_relativeIds:tf.Tensor)
  {
    this.dModel = dModel;
    this.numLayers = numLayers;
    this.useAbsolutePositionEncodings = useAbsolutePositionEncodings;
    this.sharedLayers = sharedLayers;
    this.posEncoding = posEncoding;

    this.embedding = tf.layers.embedding({inputDim:targetVocabSize, outputDim:dModel, name:"decoder-embedding"});

    if (this.sharedLayers) {
      let layer:DecoderLayer = new DecoderLayer(
        dModel, numHeads, dFF, dropoutRate,
        relativeRadius,
        useRelativePositionBiases,
        useRelativePositionEmbeddings,
        useAbsolutePositionEncodings,
        dec_relativeIds,
        dec2enc_relativeIds,
        "decoder-layer");
      for(let i:number = 0;i<numLayers;i++) {
        this.decoderLayers.push(layer);
      }
    } else {
      for(let i:number = 0;i<numLayers;i++) {
        this.decoderLayers.push(new DecoderLayer(
          dModel, numHeads, dFF, dropoutRate,
          relativeRadius,
          useRelativePositionBiases,
          useRelativePositionEmbeddings,
          useAbsolutePositionEncodings,
          dec_relativeIds,
          dec2enc_relativeIds,
          "decoder-layer-"+i));
      }
    }

    this.dropout = tf.layers.dropout({rate:dropoutRate});

    this.decoderAuxiliarLayer1 = new EncoderAuxiliarLayer1({useAbsolutePositionEncodings:useAbsolutePositionEncodings,
                                                            dModel:dModel,
                                                            posEncoding:this.posEncoding});
  }

  call(x:tf.Tensor, encOutput:tf.Tensor,
       lookAheadMask:tf.Tensor, paddingMask:tf.Tensor) : tf.Tensor
  {
    x = <tf.Tensor>this.embedding.apply(x)  // (batchSize, input_seq_len, dModel)
    x = <tf.Tensor>this.decoderAuxiliarLayer1.apply(x);
    x = <tf.Tensor>this.dropout.apply(x);

    for(let i:number=0;i<this.numLayers;i++) {
      let tmp:[tf.Tensor, tf.Tensor, tf.Tensor] = this.decoderLayers[i].call(
        x, encOutput, 
        lookAheadMask, paddingMask);
      x = tmp[0];
    }
    return x;  // (batchSize, target_seq_len, dModel)
  }
}


class FinalLayerCopyBlend extends tf.layers.Layer {
  constructor(config) {
    super(config);
  }

  computeOutputShape(inputShape) {
    return inputShape[0];
  }

  call(x) { 
    return tf.tidy(() => {
      let finalOutput = x[0];
      let copyOutput = x[1];
      let copyOutputWeight = x[2];
      return tf.add(tf.mul(tf.sub(tf.tensor1d([1.0]), copyOutputWeight), finalOutput),
                    tf.mul(copyOutputWeight, copyOutput));
    });
  }

  getClassName() { return "FinalLayerCopyBlend"; }  
}


class Transformer
{
  dModel:number;
  vocabSize:number;
  maxInputLength:number;
  maxTargetLength:number;
  relativeRadius:number;
  useRelativeDec2EncRelativePositions:boolean;
  useCopyDecoder:boolean;

  enc_relativeIds:tf.Tensor;
  dec_relativeIds:tf.Tensor;
  dec2enc_relativeIds:tf.Tensor;
  posEncoding:tf.Tensor;

  createMasks:tf.layers.Layer;
  encoder:Encoder;
  decoder:Decoder;
  finalLayer:tf.layers.Layer;
  finalLayerSoftmax:tf.layers.Layer;
  finalLayerCopy:tf.layers.Layer;
  finalLayerCopyWeight:tf.layers.Layer;
  finalLayerCopyOneHot:tf.layers.Layer;
  scaledDotProductRelativeAttention:tf.layers.Layer;
  finalLayerCopyBlend:tf.layers.Layer;

  model:tf.LayersModel;

  constructor(
    numLayers:number,
    dModel:number,
    numHeads:number,
    dFF:number,
    vocabSize:number, 
    maxInputLength:number,
    maxTargetLength:number,
    dropoutRate:number,
    relativeRadius:number,
    useRelativePositionBiases:boolean,
    useRelativePositionEmbeddings:boolean,
    useAbsolutePositionEncodings:boolean,
    useRelativeDec2EncRelativePositions:boolean,
    sharedLayers:boolean,
    useCopyDecoder:boolean)
  {
    this.dModel = dModel;
    this.vocabSize = vocabSize;
    this.maxInputLength = maxInputLength;
    this.maxTargetLength = maxTargetLength;
    this.relativeRadius = relativeRadius;
    this.useRelativeDec2EncRelativePositions = useRelativeDec2EncRelativePositions;
    this.useCopyDecoder = useCopyDecoder
    this.posEncoding = positionalEncoding(Math.max(this.maxInputLength, this.maxTargetLength), this.dModel);

    let ids = createRelativeIds(this.maxInputLength, this.maxTargetLength+1, this.relativeRadius, this.useRelativeDec2EncRelativePositions);
    this.enc_relativeIds = ids[0];
    this.dec_relativeIds = ids[1];
    this.dec2enc_relativeIds = ids[2];
    this.createMasks = new CreateMasksLayer({maxInputLength:this.maxInputLength, maxTargetLength:this.maxTargetLength});
    this.encoder = new Encoder(numLayers, dModel, numHeads, dFF, 
                           vocabSize, maxInputLength, dropoutRate,
                           relativeRadius,
                           useRelativePositionBiases,
                           useRelativePositionEmbeddings,
                           useAbsolutePositionEncodings,
                           sharedLayers,
                           this.posEncoding,
                           this.enc_relativeIds);
    this.decoder = new Decoder(numLayers, dModel, numHeads, dFF, 
                           vocabSize, maxTargetLength, dropoutRate,
                           relativeRadius,
                           useRelativePositionBiases,
                           useRelativePositionEmbeddings,
                           useAbsolutePositionEncodings,
                           sharedLayers,
                           this.posEncoding,
                           this.dec_relativeIds,
                           this.dec2enc_relativeIds);
    this.finalLayer = tf.layers.dense({units: vocabSize, name:"finalLayer"})
    this.finalLayerSoftmax = tf.layers.softmax({axis:-1});
    if (this.useCopyDecoder) {
      this.finalLayerCopy = tf.layers.dense({units: dModel, name:"finalLayerCopy"})
      this.finalLayerCopyWeight = tf.layers.dense({units: 1, activation:"sigmoid", name:"finalLayerCopyWeight"})
      this.finalLayerCopyOneHot = new OneHotLayer({depth:this.vocabSize});
      this.scaledDotProductRelativeAttention = new ScaledDotProductRelativeAttention({mask:false});
    }
    this.finalLayerCopyBlend = new FinalLayerCopyBlend({});

    let input = tf.input({shape:[maxInputLength], dtype:"int32"});
    let target = tf.input({shape:[maxTargetLength], dtype:"int32"});
    let masks = this.createMasks.apply([input, target]);
    let enc_paddingMask = masks[0];
    let lookAheadMask = masks[1];
    let dec_paddingMask = masks[2];
    let encOutput = this.encoder.call(input, enc_paddingMask)
    let decOutput = this.decoder.call(<tf.Tensor><unknown>target, encOutput, lookAheadMask, dec_paddingMask);
    let finalOutput = this.finalLayerSoftmax.apply(this.finalLayer.apply(decOutput));  // (batchSize, tar_seq_len, vocab_size)

    // Copy decoder:
    if (this.useCopyDecoder) {
      let copyOutputQuery = this.finalLayerCopy.apply(decOutput);  // (batchSize, tar_seq_len, dModel)
      let copyOutputWeight = this.finalLayerCopyWeight.apply(decOutput);

      let tmp = this.scaledDotProductRelativeAttention.apply(
          [<tf.SymbolicTensor>copyOutputQuery,  // (batchSize, tar_seq_len, dModel)
           <tf.SymbolicTensor><unknown>encOutput,        // (batchSize, inp_seq_len, dModel)
           <tf.SymbolicTensor>this.finalLayerCopyOneHot.apply(input)]);  // (batchSize, inp_seq_len, vocab_size)
      let copyOutput = tmp[0];
      finalOutput = this.finalLayerCopyBlend.apply([finalOutput, copyOutput, copyOutputWeight]);
    }

    // let output = this.call(
      // <tf.Tensor><unknown>input, <tf.Tensor><unknown>target);
    this.model = tf.model({inputs: [input, target], outputs: finalOutput});
    this.model.compile({loss: this.loss,
                        metrics: [this.tokenAccuracy, this.sequenceAccuracy],
                        optimizer: "adam"});
  }


  // Pass-through function to the internal Tensorflow method
  fit(x, y, args) : Promise<tf.History>
  {
    return this.model.fit(x, y, args);
  }


  // Pass-through function to the internal Tensorflow method.
  // This method just does one forward pass, so, it basicallly does "teacher forcing". 
  // For getting the actual prediction the Transformer would get on it's own, use the 
  // "predictIteratively" method below.
  predict(input) : tf.Tensor
  {
    return this.model.predict(input);
  }


  loss(yTrueOH, yPred) {
    let mask = tf.cast(tf.logicalNot(tf.equal(tf.argMax(yTrueOH, -1), tf.scalar(0))), "float32")
    let loss = tf.metrics.categoricalCrossentropy(tf.cast(yTrueOH, "float32"), yPred);
    return tf.div(tf.sum(tf.mul(loss, mask)), tf.sum(mask))
  }


  tokenAccuracy(yTrueOH, yPred) {
    let mask = tf.cast(tf.logicalNot(tf.equal(tf.argMax(yTrueOH, -1), tf.scalar(0))), "float32")
    let acc = tf.metrics.categoricalAccuracy(tf.cast(yTrueOH, "float32"), yPred);
    return tf.div(tf.sum(tf.mul(acc, mask)), tf.sum(mask))
  }


  sequenceAccuracy(yTrueOH, yPred) {
    let mask = tf.cast(tf.logicalNot(tf.equal(tf.argMax(yTrueOH, -1), tf.scalar(0))), "float32")
    let acc = tf.metrics.categoricalAccuracy(tf.cast(yTrueOH, "float32"), yPred);
    let accsum = tf.sum(tf.mul(acc, mask), -1)
    let masksum = tf.sum(mask, -1)
    return tf.mean(tf.cast(tf.greaterEqual(accsum, masksum), "float32"))
  }


  countParams(print:boolean=false)
  {
    let count:number = 0;  // total count
    let encoder_count:number = 0;
    let decoder_count:number = 0;
    for(let w of this.model.getWeights()) {
      count += w.size;
      if (w.name.startsWith("encoder")) encoder_count += w.size;
      if (w.name.startsWith("decoder")) decoder_count += w.size;
      if (print && w.size>0) {
        console.log(w.name + ": " + w.size)
      }
    }
    return [count, encoder_count, decoder_count];
  }


  // If you instantiate more than one model, weights will start having "_1", "_2", etc. suffixes
  getCanonicalWeightName(name:string) : string
  {
    let idx:number = name.lastIndexOf("_");
    if (idx >= 0) {
      let suffix:string = name.substring(idx+1);
      if (isNumeric(suffix)) {
        return name.substring(0, idx);
      }
    }
    return name;
  }


  async getTWeights()
  {
    let dict:{[key:string]:any[]} = {};
    for(let w of this.model.getWeights()) {
      let name = this.getCanonicalWeightName(w.name);
      dict[name] = [await w.array(), w.shape];
    }
    return dict;
  }


  setWeights(dict)
  {
    for(let w of this.model.getWeights()) {
      let name = this.getCanonicalWeightName(w.name);
      if (name in dict) {
        let value = tf.tensor(dict[name][0], dict[name][1]);
        w.assign(value)
      } else {
        throw new Error("weight " + w.name + " not found in dictionary!")
      }
    }
  }


  decodeTokenIDs(tokenIDs:number[], vocab:string[])
  {
    let decoded:string = "";
    for(let tokenID of tokenIDs) {
      decoded += vocab[tokenID];
    }
    return decoded;
  }


  decodeTokenIDsBatch(tokenIDsBatch:number[][], vocab:string[])
  {
    let result:string[] = [];
    for(let tokenIDs of tokenIDsBatch) {
      result.push(this.decodeTokenIDs(tokenIDs, vocab));
    }
    return result;
  }


  decodeBatchPrediction(output:number[][][], vocab:string[])
  {
    let result:string[] = [];
    for(let prediction of output) {
      let predictionResult:string = "";
      for(let tokenPrediction of prediction) {
        let maxIdx:number = 0;
        for(let i:number = 1;i<tokenPrediction.length;i++) {
          if (tokenPrediction[i] > tokenPrediction[maxIdx]) {
            maxIdx = i;
          }
        }
        predictionResult += vocab[maxIdx] + " ";
      }
      result.push(predictionResult);
    }
    return result;
  }


  // The "predict" method above uses teacher forcing. This, slower, one iteratively
  // predicts one token at a time without teacher forcing.
  async predictIteratively(input:number[], inputLength:number, targetLength:number) : Promise<number[]>
  {
    // Pad the input and target:
    let target:number[] = [2];  // <START> token 
    // Pad the input:
    while(input.length < inputLength) input.push(0);
    while(target.length < targetLength) target.push(0);
    let inputTensor = tf.tensor2d([input], [1, input.length]);
    for(let i:number = 1;i<targetLength;i++) {
      let targetTensor = tf.tensor2d([target], [1, target.length]);
      let predictions = this.model.predict([inputTensor, targetTensor]);
      let predictionArray:number[][][] = await predictions.array();
      let maxIdx:number = 0;
      for(let j:number = 1;j<predictionArray[0][i-1].length;j++) {
        if (predictionArray[0][i-1][j] > predictionArray[0][i-1][maxIdx]) {
          maxIdx = j;
        }
      }
      target[i] = maxIdx;
      if (maxIdx == 3) return target;  // <END> token
    }
    return target;
  }


  async evaluateOnTestSet(testset:[string, string][], generator:DatasetGenerator,
                          inputLength:number, targetLength:number) : Promise<number[]>
  {
    let tokenAccuracy:number = 0.0;
    let tokenCounts:number = 0.0;
    let sequenceAccuracy:number = 0.0;
    let sequenceCounts:number = 0.0;
    for(let instance of testset) {
      let input:number[] = generator.tokenize(instance[0]);
      let target:number[] = generator.tokenize(instance[1]);
      let predicted:number[] = await this.predictIteratively(input, inputLength, targetLength);
      tokenCounts += target.length - 1;  // "- 1", since the first symbol is "<START>", which does not count.
      sequenceCounts += 1.0;
      let anyWrong:boolean = false;
      // We start at "1", since the first symbol is "<START>", which does not count.
      for(let i = 1;i<target.length;i++) {
        if (predicted.length > i && target[i] == predicted[i]) {
          tokenAccuracy += 1.0;
        } else {
          anyWrong = true;
        }
      }
      if (!anyWrong) sequenceAccuracy += 1;
    }
    return [tokenAccuracy/tokenCounts, sequenceAccuracy/sequenceCounts];
  }

}

