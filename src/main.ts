// tf.setBackend("webgl")
tf.setBackend("cpu")

async function main()
{
  let div = document.getElementById("log");

  let symbols:string[] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]; 
  let generator:DuplicationDatasetGenerator = new DuplicationDatasetGenerator(symbols);
  // let trainingset:[string, string][] = generator.generateSet(16*64, 1, 4);
  // let trainingset:[string, string][] = generator.generateSet(128*64, 1, 4);
  let trainingset:[string, string][] = generator.generateSet(512*64, 1, 4);
  let iid_testset:[string, string][] = generator.generateSet(256, 1, 4);
  let gen_testset:[string, string][] = generator.generateSet(256, 4, 6);

  let maxInputLength:number = 7;
  let maxTargetLength:number = 14;
  let batchSize = 64;
  let trainingsetTensors = generator.stringDatasetToTensors(trainingset, maxInputLength, maxTargetLength);
  let iid_testsetTensors = generator.stringDatasetToTensors(iid_testset, maxInputLength, maxTargetLength);
  // let gen_testsetTensors = generator.stringDatasetToTensors(gen_testset, maxInputLength, maxTargetLength);

  div.innerHTML += "Creating model...<br>";
  let vocabSize:number = generator.vocab.length;
  let model = new Transformer(
    1, // numLayers
    16, // dModel
    2, // numHeads
    64, // dFF
    vocabSize, // vocabSize
    maxInputLength, // maxInputLength
    maxTargetLength, // maxTargetLength
    0.1, // dropoutRate
    16, // relativeRadius
    true, // useRelativePositionBiases
    true, // useRelativePositionEmbeddings
    true, // useAbsolutePositionEncodings
    false, // useRelativeDec2EncRelativePositions
    true, // sharedLayers
    true, // useCopyDecoder
    );

  /*
  let model2 = new Transformer(
    1, // numLayers
    16, // dModel
    2, // numHeads
    64, // dFF
    vocabSize, // vocabSize
    maxInputLength, // maxInputLength
    maxTargetLength, // maxTargetLength
    0.1, // dropoutRate
    16, // relativeRadius
    true, // useRelativePositionBiases
    true, // useRelativePositionEmbeddings
    true, // useAbsolutePositionEncodings
    false, // useRelativeDec2EncRelativePositions
    true, // sharedLayers
    true, // useCopyDecoder
    );
  */

  let input = tf.slice(trainingsetTensors[0], [0,0], [1,-1]);
  let target = tf.slice(trainingsetTensors[1], [0,0], [1,-1]);
  let label = tf.slice(trainingsetTensors[2], [0,0], [1,-1]);
  // Calling the model for the first time, to make sure everything is initizlized...
  model.predict([input, target]);

  div.innerHTML += "Model parameter count: " + model.countParams()[0] + "<br>";
  // let weights = await getTransformerWeights(model)
  // console.log(weights);
  // let weightsString:string = JSON.stringify(weights);
  // let weights2 = JSON.parse(weightsString);

  // console.log(await transformerPredict(model, [4, 5, 6], maxInputLength, maxTargetLength));

  div.innerHTML += "Training... ";
  let html_tmp = div.innerHTML;
  let nBatches:number = Math.floor(trainingset.length / batchSize);
  let startTime:number = new Date().getTime();
  await model.fit([trainingsetTensors[0], trainingsetTensors[1]], tf.oneHot(trainingsetTensors[2], vocabSize), {
     epochs: 1,
     batchSize: batchSize,
     validationData: [[iid_testsetTensors[0], iid_testsetTensors[1]], tf.oneHot(iid_testsetTensors[2], vocabSize)],
     yieldEvery: "batch",
     callbacks: {onBatchEnd: (batch, logs) => {
                  let time:number = new Date().getTime();
                  let timePerBatch:number = (time - startTime) / (batch+1)
                  let str:string = "batch " + batch + "/" + nBatches + "<br>";
                  str += "<p style='margin-left: 40px'>loss = " + logs.loss + "<br>";
                  str += "token accuracy = " + logs.tokenAccuracy + "<br>";
                  str += "sequence accuracy = " + logs.sequenceAccuracy + "<br>";
                  str += "time per batch = " + (timePerBatch/1000) + "s<br>";
                  str += "ETA: = " + ((nBatches-batch)*timePerBatch/1000) + "s</p>";
                  div.innerHTML = html_tmp + str;
                }}
    }).then(info => {
      div.innerHTML += "Final token accuracy: " + info.history.transformerTokenAccuracy +
                       ", Final sequence accuracy: " + info.history.transformerSequenceAccuracy + "<br><br>";
    });

  div.innerHTML += "Making a prediction after training: <br>";
  let predictions = model.predict([input, target]);
  div.innerHTML += "<p style='margin-left: 40px'>";
  div.innerHTML += "input: " + model.decodeTokenIDsBatch(await input.array(), generator.vocab) + "<br>";
  div.innerHTML += "target: " + model.decodeTokenIDsBatch(await target.array(), generator.vocab) + "<br>";
  div.innerHTML += "label: " + model.decodeTokenIDsBatch(await label.array(), generator.vocab) + "<br>";
  // div.innerHTML += "prediction: " + predictions + "<br>";
  div.innerHTML += "decoded: " + model.decodeBatchPrediction(await predictions.array(), generator.vocab) + "</p>";

  // console.log(await transformerPredict(model, [4, 5, 6], maxInputLength, maxTargetLength));

  div.innerHTML += "Evaluating on the iid set... ";
  let iid_eval:number[] = await model.evaluateOnTestSet(iid_testset, generator, maxInputLength, maxTargetLength);
  div.innerHTML += "token accuracy: " + iid_eval[0] + ", sequence accuracy: " + iid_eval[1] + "<br>";
  let gen_eval:number[] = await model.evaluateOnTestSet(gen_testset, generator, maxInputLength, maxTargetLength);
  div.innerHTML += "Evaluating on the gen set... ";
  div.innerHTML += "token accuracy: " + gen_eval[0] + ", sequence accuracy: " + gen_eval[1] + "<br>";

  // weights = await getTransformerWeights(model)
  // console.log(weights);

  // setTransformerWeights(model2, weights);

  // weights = await getTransformerWeights(model2)
  // console.log(weights);

  // predictions = model2.predict([input, target]);
  // console.log("input: " + decodeTokenIDsBatch(await input.array(), vocab))
  // console.log("target: " + decodeTokenIDsBatch(await target.array(), vocab))
  // console.log("label: " + decodeTokenIDsBatch(await label.array(), vocab))
  // console.log("prediction: " + predictions);
  // console.log("decoded: " + decodeTransformerBatchPrediction(await predictions.array(), vocab));

}
main();
