// Matrix multiplication shader with tiling for performance
// Supports both standard and transposed matrices

struct MatmulParams {
    M: u32,           // Rows of A
    N: u32,           // Cols of B
    K: u32,           // Cols of A / Rows of B
    transA: u32,      // Transpose A
    transB: u32,      // Transpose B
    alpha: f32,       // Scale factor
    beta: f32,        // Output scale factor
}

@group(0) @binding(0) var<storage, read> matrixA: array<f32>;
@group(0) @binding(1) var<storage, read> matrixB: array<f32>;
@group(0) @binding(2) var<storage, read_write> matrixC: array<f32>;
@group(0) @binding(3) var<uniform> params: MatmulParams;

// Tile size optimized for browser compatibility
const TILE_SIZE: u32 = 16u;

var<workgroup> tileA: array<f32, 256>; // TILE_SIZE * TILE_SIZE
var<workgroup> tileB: array<f32, 256>;

fn getA(row: u32, col: u32) -> f32 {
    if (params.transA != 0u) {
        return matrixA[col * params.M + row];
    } else {
        return matrixA[row * params.K + col];
    }
}

fn getB(row: u32, col: u32) -> f32 {
    if (params.transB != 0u) {
        return matrixB[col * params.K + row];
    } else {
        return matrixB[row * params.N + col];
    }
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>) {

    let row = global_id.y;
    let col = global_id.x;

    // Check bounds
    if (row >= params.M || col >= params.N) {
        return;
    }

    var sum = 0.0;

    // Tiled matrix multiplication
    let numTiles = (params.K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t = 0u; t < numTiles; t = t + 1u) {
        // Load tile of A
        let tileRow = local_id.y;
        let tileCol = local_id.x;
        let aRow = workgroup_id.y * TILE_SIZE + tileRow;
        let aCol = t * TILE_SIZE + tileCol;

        if (aRow < params.M && aCol < params.K) {
            tileA[tileRow * TILE_SIZE + tileCol] = getA(aRow, aCol);
        } else {
            tileA[tileRow * TILE_SIZE + tileCol] = 0.0;
        }

        // Load tile of B
        let bRow = t * TILE_SIZE + tileRow;
        let bCol = workgroup_id.x * TILE_SIZE + tileCol;

        if (bRow < params.K && bCol < params.N) {
            tileB[tileRow * TILE_SIZE + tileCol] = getB(bRow, bCol);
        } else {
            tileB[tileRow * TILE_SIZE + tileCol] = 0.0;
        }

        workgroupBarrier();

        // Compute partial dot product
        for (var k = 0u; k < TILE_SIZE; k = k + 1u) {
            sum = sum + tileA[tileRow * TILE_SIZE + k] * tileB[k * TILE_SIZE + tileCol];
        }

        workgroupBarrier();
    }

    // Write result with alpha/beta scaling
    let index = row * params.N + col;
    if (params.beta != 0.0) {
        matrixC[index] = params.alpha * sum + params.beta * matrixC[index];
    } else {
        matrixC[index] = params.alpha * sum;
    }
}

// Optimized matrix-vector multiplication
struct MatvecParams {
    M: u32,
    N: u32,
    transA: u32,
    alpha: f32,
    beta: f32,
}

@group(0) @binding(0) var<storage, read> matrix: array<f32>;
@group(0) @binding(1) var<storage, read> vector: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
@group(0) @binding(3) var<uniform> matvecParams: MatvecParams;

@compute @workgroup_size(256, 1, 1)
fn matvec(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;

    if (row >= matvecParams.M) {
        return;
    }

    var sum = 0.0;

    for (var i = 0u; i < matvecParams.N; i = i + 1u) {
        let matrixVal = matrix[row * matvecParams.N + i];
        let vectorVal = vector[i];
        sum = sum + matrixVal * vectorVal;
    }

    if (matvecParams.beta != 0.0) {
        result[row] = matvecParams.alpha * sum + matvecParams.beta * result[row];
    } else {
        result[row] = matvecParams.alpha * sum;
    }
}

// Batch matrix multiplication
struct BatchMatmulParams {
    batchSize: u32,
    M: u32,
    N: u32,
    K: u32,
    strideA: u32,
    strideB: u32,
    strideC: u32,
}

@group(0) @binding(0) var<storage, read> batchA: array<f32>;
@group(0) @binding(1) var<storage, read> batchB: array<f32>;
@group(0) @binding(2) var<storage, read_write> batchC: array<f32>;
@group(0) @binding(3) var<uniform> batchParams: BatchMatmulParams;

@compute @workgroup_size(16, 16, 1)
fn batchMatmul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch = global_id.z;
    let row = global_id.y;
    let col = global_id.x;

    if (batch >= batchParams.batchSize || row >= batchParams.M || col >= batchParams.N) {
        return;
    }

    var sum = 0.0;

    let aOffset = batch * batchParams.strideA;
    let bOffset = batch * batchParams.strideB;

    for (var k = 0u; k < batchParams.K; k = k + 1u) {
        let a = batchA[aOffset + row * batchParams.K + k];
        let b = batchB[bOffset + k * batchParams.N + col];
        sum = sum + a * b;
    }

    let cOffset = batch * batchParams.strideC;
    batchC[cOffset + row * batchParams.N + col] = sum;
}
