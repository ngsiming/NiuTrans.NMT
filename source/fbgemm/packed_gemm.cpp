#include "../niutensor/tensor/XTensor.h"
#include "../niutensor/tensor/core/CHeader.h"
#include "../niutensor/tensor/function/FHeader.h"
#include <fbgemm/QuantUtils.h>
#include <fbgemm/Fbgemm.h>
#include "packed_gemm.h"

using namespace nts;
using namespace fbgemm;

// Memory blocking factors (parameters) for packing into AVX2 int8
static const BlockingFactors Packed8Avx2BlockingFactors = {
    PackingTraits<int8_t, int32_t, inst_set_t::avx2>::MR,
    PackingTraits<int8_t, int32_t, inst_set_t::avx2>::NR,
    PackingTraits<int8_t, int32_t, inst_set_t::avx2>::NR_MIN,
    PackingTraits<int8_t, int32_t, inst_set_t::avx2>::ROW_INTERLEAVE,
    PackingTraits<int8_t, int32_t, inst_set_t::avx2>::MCB,
    PackingTraits<int8_t, int32_t, inst_set_t::avx2>::KCB,
    PackingTraits<int8_t, int32_t, inst_set_t::avx2>::NCB
};

// Memory blocking factors (parameters) for packing into AVX512 int8
static const BlockingFactors Packed8Avx512BlockingFactors = {
    PackingTraits<int8_t, int32_t, inst_set_t::avx512>::MR,
    PackingTraits<int8_t, int32_t, inst_set_t::avx512>::NR,
    PackingTraits<int8_t, int32_t, inst_set_t::avx512>::NR_MIN,
    PackingTraits<int8_t, int32_t, inst_set_t::avx512>::ROW_INTERLEAVE,
    PackingTraits<int8_t, int32_t, inst_set_t::avx512>::MCB,
    PackingTraits<int8_t, int32_t, inst_set_t::avx512>::KCB,
    PackingTraits<int8_t, int32_t, inst_set_t::avx512>::NCB
};

// This function returns the correct blocking factors structure for given packing type.
inline const BlockingFactors* getBlockingFactors(Type packType) {
  if(packType == Type::packed8avx2) {
    return &Packed8Avx2BlockingFactors;
  } else if(packType == Type::packed8avx512) {
    return &Packed8Avx512BlockingFactors;
  } else {
    //ABORT("Only avx2 and avx512 instruction sets are supported for int8. {}", packType);
  }
}

void fbgemmPacked8PackInfo(const std::vector<int>& shape,
                           const Type packType,
                           const bool transpose,
                           int& nrow,
                           int& ncol,
                           uint64_t& packsize) {
    // Should be 2D - weight matrix
    //ABORT_IF(shape.size() != 2,
            //"Weight Matrix should be 2D");
    nrow = transpose ? shape[1] : shape[0];
    ncol = transpose ? shape[0] : shape[1];

    const BlockingFactors* params =getBlockingFactors(packType);

    packsize = fbgemm::PackMatrix<fbgemm::PackBMatrix<int8_t>, int8_t>::packedBufferSize(
        transpose ? shape[1] : shape[0],
        transpose ? shape[0] : shape[1], params);
    // add extra space for storing some other variables specific to B matrix
    // quantization sacles: 1 per column and float
    // quantization offset: 1 per column and int32
    // column offsets: 1 per column and int32
    packsize += ncol * (sizeof(float) + sizeof(int32_t) + sizeof(int32_t));
}

// This function computes the offset values for each column which are used for compensating the remainders of quantized values
// More detailed math is avilable in the FBGEMM's blog - https://engineering.fb.com/ml-applications/fbgemm/
inline void col_offsets_with_zero_pt_s8acc32(
    bool transpose,
    int K,
    int N,
    const int8_t* Bint8,
    const int32_t* B_zero_point,
    int32_t* col_offsets,
    int ncols_per_quant_group) {
  for (int n = 0; n < N; ++n) {
    int32_t sum = 0;
    for (int k = 0; k < K; ++k) {
      sum += transpose ? Bint8[k + n * K] : Bint8[k * N + n];
    }
    col_offsets[n] = sum - B_zero_point[n / ncols_per_quant_group] * K;
  }
}

void fbgemmPacked8Pack(
        PackBMatrix<int8_t>*& packedBN,
        float*& bqScale,
        int32_t*& bqZeropoint,
        int32_t*& col_offsets,
		const float* inData,
		const Type packType,
		const bool transpose,
		const int nrow,
		const int ncol,
		const uint64_t packsize) {
	int k = nrow;
	int n = ncol;
	int len = k * n;

	// 1. collect stats for each column
    bqScale = new float[n];
    bqZeropoint = new int32_t[n];

	const float* data = inData;
	float val = 0;

	if (transpose) {
		for (int jj = 0; jj < n; jj++) {
			float min = std::numeric_limits<float>::max(), max = std::numeric_limits<float>::min();
			double mean = 0, sqrsum = 0;
			for (int ii = 0; ii < k; ii++) {
				val = data[jj * k + ii];
				mean += val;
				sqrsum += val * val;
			}
			mean /= k;
			sqrsum /= k;
			sqrsum -= mean * mean;
			sqrsum = sqrt(sqrsum);

			min = (float)(mean - 7.0f*sqrsum);
			max = (float)(mean + 7.0f*sqrsum);
			bqScale[jj] = (max - min) / 255;
			bqZeropoint[jj] = (int32_t)(127 - max / bqScale[jj]);
		}
	} else {
		for (int jj = 0; jj < n; jj++) {
			float min = std::numeric_limits<float>::max(), max = std::numeric_limits<float>::min();
			double mean = 0, sqrsum = 0;
			for (int ii = 0; ii < k; ii++) {
				val = data[jj + ii * n];
				mean += val;
				sqrsum += val * val;
			}
			mean /= k;
			sqrsum /= k;
			sqrsum -= mean * mean;
			sqrsum = sqrt(sqrsum);

			min = (float)(mean - 7.0f*sqrsum);
			max = (float)(mean + 7.0f*sqrsum);
			bqScale[jj] = (max - min) / 255;
			bqZeropoint[jj] = (int32_t)(127 - max / bqScale[jj]);
		}
	}
	
  // 2. quantize
  int8_t* quantized = (int8_t*)malloc(len);
//#ifdef _MSC_VER
  //quantized = (int8_t*)_aligned_malloc(len, 256);
//#else
  //int result = posix_memalign((void**)&quantized, 256, len); result;
  //assert(result == 0);
//#endif
  for (int jj = 0; jj < n; jj++) {
    TensorQuantizationParams bQuantParam;
    bQuantParam.scale = bqScale[jj];
    bQuantParam.zero_point = bqZeropoint[jj];
    bQuantParam.precision = 8;

    if (transpose)
      fbgemm::Quantize<int8_t>(data + jj * k, quantized + jj * k, k, bQuantParam);
    else {
      for (int ii = 0; ii < k; ii++) {
        quantized[ii*n + jj] = fbgemm::Quantize<int8_t>(data[ii*n + jj], bQuantParam);
      }
    }
  }

  // 3. compute column offsets
  col_offsets = new int32_t[n];
  col_offsets_with_zero_pt_s8acc32(transpose, k, n, quantized, bqZeropoint, col_offsets, 1);



  // 4. packing
  const fbgemm::BlockingFactors* params = getBlockingFactors(packType);
  
   packedBN = new PackBMatrix<int8_t>(
      transpose ? matrix_op_t::Transpose : matrix_op_t::NoTranspose,
      nrow, ncol, quantized, transpose ? nrow : ncol, nullptr, 1, params);


//#ifdef _MSC_VER
  //_aligned_free(quantized);
//#else
  //free(quantized);
//#endif
free(quantized);
}


// GEMM operation on the packed B matrix in 8 bit integers
// C: output matrix
// A: A matrix
// B: B matrix (packed)
// m: the number of rows in A and C
// n: the number of columns in B and C
// k: the number of columns in A and the number of rows in B
// transA: whether A matrix is transposed or not
// transB: whether B matrix is transposed or not
void fbgemmPacked8Gemm(XTensor& C,
                       const XTensor A,
                       PackBMatrix<int8_t>* const packedBN,
                       float* const bqScale,
                       int32_t* const bqZeropoint,
                       int32_t* const col_offsets,
                       const size_t m,
                       const size_t n,
                       const size_t k,
                       const int transA,
                       const int transB) {

  // pack type
  //marian::Type packType = B->type();
  Type packType = Type::packed8avx2;

  const fbgemm::BlockingFactors* params = getBlockingFactors(packType);

  if((packType == Type::packed8avx2 && fbgemmHasAvx512Support())
     || (packType == Type::packed8avx2 && !fbgemmHasAvx2Support())
     || (packType == Type::packed8avx512 && !fbgemmHasAvx512Support())) {
    //ABORT("FBGEMM doesn't allow to use {} packing order on {} CPUs",
          //packType == Type::packed8avx2 ? "AVX2" : "AVX512",
          //fbgemmHasAvx512Support() ? "AVX512" : "AVX2");
    printf("FBGEMM doesn't allow to use %s packing order on %s CPUS\n",packType==packed8avx2?"AVX2":"AVX512",fbgemmHasAvx512Support()?"AVX512":"AVX2");
    return;
  }


  // compute range to quantize A (activations) - (min/max quantization)
  float min_est = std::numeric_limits<float>::max(), max_est = std::numeric_limits<float>::min();

  //int elem = A->shape().elements();
  int elem = A.unitNum;
  float* adata = (float*)A.data;

  // AVX based find min/max
  FindMinMax(adata, &min_est, &max_est, elem);

  float ascale = (max_est - min_est) / 255;
  int32_t azeropoint = (int32_t)(255 - max_est / ascale);

  std::vector<int32_t> row_offset_buf(PackAWithQuantRowOffset<uint8_t>::rowOffsetBufferSize());
  PackAWithQuantRowOffset<uint8_t> packAN(
      transA ? matrix_op_t::Transpose : matrix_op_t::NoTranspose,
      (int32_t)(transA ? k : m),
      (int32_t)(transA ? m : k),
      (float*)A.data,
      (int32_t)(transA ? m : k),
      nullptr, /*buffer for packed matrix*/
      ascale,
      azeropoint,
      1, /*groups*/
      row_offset_buf.data(),
      params);

  DoNothing<float, float> doNothingObj{};
  ReQuantizeForFloat<false, QuantizationGranularity::OUT_CHANNEL> outputProcObj(
      doNothingObj,
      ascale,
      bqScale,
      azeropoint,
      bqZeropoint,
      packAN.getRowOffsetBuffer(),
      col_offsets,
      nullptr,
      (std::uint32_t) n);

  // gemm computation
  fbgemmPacked(packAN, *packedBN, (float*)C.data, (int32_t*)C.data, (int32_t) n, outputProcObj, 0, 1, params);
}
