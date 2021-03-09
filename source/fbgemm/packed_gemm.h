#include "../niutensor/tensor/XTensor.h"
#include "../niutensor/tensor/core/CHeader.h"
#include "../niutensor/tensor/function/FHeader.h"
#include <fbgemm/QuantUtils.h>
#include <fbgemm/Fbgemm.h>
#include <vector>

using namespace nts;
using namespace fbgemm;

enum Type {packed8avx2, packed8avx512};

void fbgemmPacked8PackInfo(const std::vector<int>& shape,
                           const Type packType,
                           const bool transpose,
                           int& nrow,
                           int& ncol,
                           uint64_t& packsize); 
void fbgemmPacked8Pack(
		//XTensor out,
		int8_t*& packedbuf,
		const float* inData,
		const Type packType,
		const bool transpose,
		const int nrow,
		const int ncol,
		const uint64_t packsize);

void fbgemmPacked8Gemm(XTensor& C,
                       const XTensor A,
                       const XTensor B,
                       const size_t m,
                       const size_t n,
                       const size_t k,
                       const int transA,
                       const int transB);
