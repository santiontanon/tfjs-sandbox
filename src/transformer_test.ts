function compare2dArrays(a1:number[][], a2:number[][]): boolean
{
  if (a1.length != a2.length) return false;
  for(let i:number = 0;i<a1.length;i++) {
    if (a1[i].length != a2[i].length) return false;
    for(let j:number = 0;j<a1[i].length;j++) {
      if (a1[i][j] != a2[i][j]) return false;
    }
  }
  return true;
}


function compare2dArraysApprox(a1:number[][], a2:number[][], tolerance:number): boolean
{
  if (a1.length != a2.length) return false;
  for(let i:number = 0;i<a1.length;i++) {
    if (a1[i].length != a2[i].length) return false;
    for(let j:number = 0;j<a1[i].length;j++) {
      if (Math.abs(a1[i][j] - a2[i][j]) >= tolerance) return false;
    }
  }
  return true;
}


function compare3dArraysApprox(a1:number[][][], a2:number[][][], tolerance:number): boolean
{
  if (a1.length != a2.length) return false;
  for(let i:number = 0;i<a1.length;i++) {
    if (!compare2dArraysApprox(a1[i], a2[i], tolerance)) return false;
  }
  return true;
}


function compare4dArraysApprox(a1:number[][][][], a2:number[][][][], tolerance:number): boolean
{
  if (a1.length != a2.length) return false;
  for(let i:number = 0;i<a1.length;i++) {
    if (!compare3dArraysApprox(a1[i], a2[i], tolerance)) return false;
  }
  return true;
}


// Tests:
function test_getAngles()
{
  let arg1:number[][] = [[0], [1], [2]];
  let arg2:number[][] = [[0, 1, 2, 3]];
  let arg3:number = 4;
  let actual:number[][] = getAngles(arg1, arg2, arg3);
  let expected:number[][] = [[0, 0, 0, 0], [1, 1, 0.01, 0.01], [2, 2, 0.02, 0.02]];
  if (!compare2dArrays(actual, expected)) {
    console.error("test_getAngles failed!")
  }
}


async function test_positionalEncoding()
{
  let actual:number[][][] = await positionalEncoding(3, 4).array();
  let expected:number[][][] = 
      [[[ 0.        ,  1.        ,  0.        ,  1.        ],
        [ 0.84147096,  0.5403023 ,  0.00999983,  0.99995   ],
        [ 0.9092974 , -0.41614684,  0.01999867,  0.9998    ]]];
  if (!compare2dArraysApprox(actual[0], expected[0], 0.00001)) {
    console.error("test_positionalEncoding failed!")
  }
}


async function test_createPaddingMask()
{
  let x:tf.Tensor2D = tf.tensor2d([[1,2,3,0], [1,1,0,0]], [2,4]);
  let actual:number[][][][] = await createPaddingMask(x).array();
  let expected:number[][][][] = [[[[0,0,0,1]]], [[[0,0,1,1]]]];
  if (!compare4dArraysApprox(actual, expected, 0.00001)) {
    console.error("test_createPaddingMask failed!")
    console.error("actual: " + actual)
    console.error("expected: " + expected)
  }
}


async function test_createLookAheadMask()
{
  let actual:number[][] = await createLookAheadMask(4).array();
  let expected:number[][] = 
      [[0., 1., 1., 1.],
       [0., 0., 1., 1.],
       [0., 0., 0., 1.],
       [0., 0., 0., 0.]];
  if (!compare2dArraysApprox(actual, expected, 0.00001)) {
    console.error("test_positionalEncoding failed!")
  }
}


async function test_createMasks()
{
  let inp:tf.Tensor2D = tf.tensor2d([[1,2,3,0],[1,1,0,0]], [2,4]);
  let tar:tf.Tensor2D = tf.tensor2d([[4,5],[3,0]], [2,2]);
  let actual:[tf.Tensor4D, tf.Tensor4D, tf.Tensor4D] = createMasks(inp, tar);
  let actual1:number[][][][] = await actual[0].array();
  let actual2:number[][][][] = await actual[1].array();
  let actual3:number[][][][] = await actual[2].array();
  let expected1:number[][][][] = 
    [[[[0., 0., 0., 1.]]],
     [[[0., 0., 1., 1.]]]];
  let expected2:number[][][][] = 
    [[[[0., 1.], [0., 0.]]],
     [[[0., 1.], [0., 1.]]]];
  let expected3:number[][][][] = 
    [[[[0., 0., 0., 1.]]],
     [[[0., 0., 1., 1.]]]];
  if (!compare4dArraysApprox(actual1, expected1, 0.00001) ||
      !compare4dArraysApprox(actual2, expected2, 0.00001) ||
      !compare4dArraysApprox(actual3, expected3, 0.00001)) {
    console.error("test_createMasks failed!")
  }     
}


async function test_createRelativeIds()
{
  let actual:[tf.Tensor2D, tf.Tensor2D, tf.Tensor2D] = createRelativeIds(4, 3, 2, true);
  let actualFalse:[tf.Tensor2D, tf.Tensor2D, tf.Tensor2D] = createRelativeIds(4, 3, 2, false);
  let actual1:number[][] = await actual[0].array();
  let actual2:number[][] = await actual[1].array();
  let actual3:number[][] = await actual[2].array();
  let actual3False:number[][] = await actualFalse[2].array();
  let expected1:number[][] = 
    [[2, 1, 0, 0],
     [3, 2, 1, 0],
     [4, 3, 2, 1],
     [4, 4, 3, 2]];
  let expected2:number[][] = 
    [[2, 1],
     [3, 2]];
  let expected3:number[][] = 
    [[2, 1, 0, 0],
     [3, 2, 1, 0]];
  let expected3False:number[][] = 
    [[2, 2, 2, 2],
     [2, 2, 2, 2]];
  if (!compare2dArraysApprox(actual1, expected1, 0.00001) ||
      !compare2dArraysApprox(actual2, expected2, 0.00001) ||
      !compare2dArraysApprox(actual3, expected3, 0.00001) ||
      !compare2dArraysApprox(actual3False, expected3False, 0.00001)) {
    console.error("test_createRelativeIds failed!")
  }          
}


// Test the functions:
// test_getAngles();
// test_positionalEncoding();
// test_createPaddingMask();
// test_createLookAheadMask();
// test_createMasks();
// test_createRelativeIds();
