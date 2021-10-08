class DatasetGenerator {
  padSymbol:string = "-";
  sepSymbol:string = "<SEP>";
  startSymbol:string = "<START>";
  endSymbol:string = "<END>";

  padIdx:number = 0;
  sepIdx:number = 1;
  startIdx:number = 2;
  endIdx:number = 3;

  vocab:string[] = [this.padSymbol, this.sepSymbol, this.startSymbol, this.endSymbol];


  tokenize(input:string) : number[]
  {
    let tokenIDs:number[] = [];
    for(let token of input.split(" ")) {
      let tokenID:number = this.vocab.indexOf(token);
      if (tokenID == -1) throw new Error("Token '" + token + "' is not in the vocab in input '"+input+"'!");
      tokenIDs.push(tokenID);
    }
    return tokenIDs;
  }


  stringDatasetToTensors(data:[string,string][], maxInputLength:number, maxOutputLength:number) : tf.Tensor2D[]
  {
    let inputTensor:number[][] = [];
    let targetTensor:number[][] = []; 
    let labelsTensor:number[][] = [];  // labels is like target, but shifted one to the left (for teacher forcing)
    for(let instance of data) {
      let instanceInputTensor:number[] = this.tokenize(instance[0]);
      let instanceTargetTensor:number[] = this.tokenize(instance[1]);
      let instanceLabelsTensor:number[] = [];
      for(let i:number = 1;i<instanceTargetTensor.length;i++) instanceLabelsTensor.push(instanceTargetTensor[i]);
      while(instanceInputTensor.length < maxInputLength) instanceInputTensor.push(0);
      while(instanceTargetTensor.length < maxOutputLength) instanceTargetTensor.push(0);
      while(instanceLabelsTensor.length < maxOutputLength) instanceLabelsTensor.push(0);
      inputTensor.push(instanceInputTensor);
      targetTensor.push(instanceTargetTensor);
      labelsTensor.push(instanceLabelsTensor);
    }
    return [tf.cast(tf.tensor2d(inputTensor), "int32"),
            tf.cast(tf.tensor2d(targetTensor), "int32"),
            tf.cast(tf.tensor2d(labelsTensor), "int32")];
  }
}



class DuplicationDatasetGenerator extends DatasetGenerator {
  symbols:string[] = null;

  constructor(a_symbols:string[])
  {
    super();
    this.symbols = a_symbols;
    for(let symbol in this.symbols) {
      this.vocab.push(symbol);
    }    
  }
  

  generateSet(size:number, minSymbols:number, maxSymbols:number) : [string, string][]
  {
    let examples:[string,string][] = [];

    for(let i:number = 0;i<size;i++) {
      let input:string = "";
      let numSymbols:number = Math.floor(Math.random() * (maxSymbols - minSymbols) + minSymbols);
      for(let j:number = 0;j<numSymbols;j++) {
        let symbolIdx:number = Math.floor(Math.random() * this.symbols.length);
        input += this.symbols[symbolIdx] + " ";
      }
      let output:string = this.startSymbol + " " + input + input + this.endSymbol;
      examples.push([input.trim(), output.trim()]);
    }

    return examples;
  }
}
