#include <immintrin.h>
#include <climits>

static int const kc=16;
static int const mr=7;
static int const nr=7;
static int const regLen=256;
/*
 * @brief compute the vector dot product. Notes that only supports float now.
 */
template<class T>
T avx2VecDotPro(T* const A, T* const B);

/*
 * @param A packed in contiguous format. Resides in L2 cache. Sizes: mr*kc.
 * @param B packed in contiguous format. Resides in L1 cache. Sizes: kc*nr.
 * @param C Notes that C is not contiguous and not necessary in any cache. Size: mr*nr.
 * @param regLen register length.
 * Notes that mr*nr elements should take up half of the registers (e.g. 8 ymm registers in avx2, i.e. 8*256bits).  
 * Notes that bits of kc elements must be multiples of the register length (e.g. 256 bits for avx).
 */
template<class T>
void kernel(T* const A, T* const B, T* const C, int const mr, int const kc, int const nr, int const regLen,
        int const m, int const n, int const CRowStart, int const CColStart);

/*
 * @brief Pack A into many blocks, and adjust the storage format so that it can be accessed continuously during calculation.
 * @param A The matrix to be Packed, the size is m*k.
 * @param m Number of rows of A.
 * @param k Number of columns of A.
 * @param mc Number of rows of a block.
 * @param kc Number of columns of block.
 *
 * @return packedBuf pointer to the packed matrix.
 *
 * Notes that although we pack A into small blocks of mc*kc, we can still access the small blocks of mr*kc continuously during calculation.
 * Because these two small blocks are both stored in row-major order.
 *
 * For example, an A matrix is as follows:
 *     a00 a01 a02 a03 a04 a05 a06
 *     a10 a11 a12 a13 a14 a15 a16 
 *     a20 a21 a22 a23 a24 a25 a26
 *     a30 a31 a32 a33 a34 a35 a36
 *     a40 a41 a42 a43 a44 a45 a46
 * m is 5, k is 7. Let mc is 4 and kc is 4. Then the packed matrix should be stored in memory like this:
 *
 * a00 a01 a02 a03 a10 a11 a12 a13 a20 a21 a22 a23 a30 a31 a32 a33 | a40 a41 a42 a43 0 0 0 0 0 0 0 0 0 0 0 0 | \
 * a04 a05 a06 0 a14 a15 a16 0 a24 a25 a26 0 a34 a35 a36 0 | a44 a45 a46 0 0 0 0 0 0 0 0 0 0 0 0 0 | 
 * Notes 1. | indicates the separation between blocks.
 * Notes 2. \ means continue on the next line.
 * Notes 3. Fill the free memory with 0.
 */
template<class T>
T* packA(T* A, int const m, int const k, int const mc, int const kc);

/*
 * @brief Pack B into many blocks, and adjust the storage format so that it can be accessed continuously during calculation.
 *
 * @param B The matrix to be Packed, the size is k*n.
 * @param k Number of rows of B.
 * @param n Number of columns of B.
 * @param kc Number of rows of a block.
 * @param nr Number of columns of block.
 *
 * @return packedBuf pointer to the packed matrix.
 *
 * For example, an B matrix is as follows:
 *    b00 b01 b02 b03 b04 b05 b06
 *    b10 b11 b12 b13 b14 b15 b16
 *    b20 b21 b22 b23 b24 b25 b26
 *    b30 b31 b32 b33 b34 b35 b36
 *    b40 b41 b42 b43 b44 b45 b46
 *    b50 b51 b52 b53 b54 b55 b56
 *    b60 b61 b62 b63 b64 b65 b66
 * k is 7, n is 7. Let kc is 4 and nr is 4. Then the packed matrix should be stored in memory like this:
 *
 * b00 b10 b20 b30 b01 b11 b21 b31 b02 b12 b22 b32 b03 b13 b23 b33 | b04 b14 b24 b34 b05 b15 b25 b35 b06 b16 b26 b36 0 0 0 0 | \
 * b40 b50 b60 0 b41 b51 b61 0 b42 b52 b62 0 b43 b53 b63 0 | b44 b54 b64 0 b45 b55 b65 0 b46 b56 b66 0 0 0 0 0 |
 */
template<class T>
T* packB(T* const B, int const k, int const n, int const kc, int const nr);

struct XBlockingFactors 
{
    int KC;
    int MR;
    int NR;
};

static int const LEVEL1_DCACHE_LINESIZE_BITS=64*8;
static int const LEVEL2_CACHE_LINESIZE_BITS=64*8;

template<class T>
void GEMM(T* const A, T* const B, T* const C, int const m, int const k, int const n, struct XBlockingFactors const factors, int const regLen);
