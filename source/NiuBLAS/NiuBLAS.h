#include <immintrin.h>
#include <climits>
#include "../niutensor/tensor/XTensor.h"

static int const kc=256;
static int const mr=7;
static int const nr=7;
static int const regLen=256;
static int const LEVEL1_DCACHE_LINESIZE_BITS=64*8;
static int const LEVEL2_CACHE_LINESIZE_BITS=64*8;

struct XBlockingFactors 
{
    int KC;
    int MR;
    int NR;
};

/*
 * @brief compute the vector dot product. Notes that only supports float now.
 */
template<class T>
T avx2VecDotPro(T* const A, T* const B)
{
   __m256 a = _mm256_loadu_ps(A); 
   __m256 b = _mm256_loadu_ps(B); 
   //__m256 dot = _mm256_mul_ps(a,b);
   //__m256 tmp = _mm256_shuffle_ps(a,b,_MM_SHUFFLE(1,2,3,4));
   __m256 dot = _mm256_dp_ps(a,b,0xff);
   float tmp[10];
   _mm256_storeu_ps(tmp,dot);
   a = _mm256_loadu_ps(tmp);
   tmp[0]=tmp[4];
   b = _mm256_loadu_ps(tmp);
   dot = _mm256_add_ps(a,b);
   _mm256_storeu_ps(tmp,dot);
   return tmp[0];
   //return dot;
}
/*
 * @param A packed in contiguous format. Resides in L2 cache. Sizes: mr*kc.
 * @param B packed in contiguous format. Resides in L1 cache. Sizes: kc*nr.
 * @param C Notes that C is not contiguous and not necessary in any cache. Size: mr*nr.
 * @param regLen register length.
 * Notes that mr*nr elements should take up half of the registers (e.g. 8 ymm registers in avx2, i.e. 8*256bits).  
 * Notes that bits of kc elements must be multiples of the register length (e.g. 256 bits for avx).
 */
template<class T>
void kernel(T* const A, T* const B, nts::XTensor* const C, int const mr, int const kc, int const nr, int const regLen,
        int const m, int const n, int const CRowStart, int const CColStart)
{ 
    // should check if the parameter fits the requirement.
    // step represents the number of sub vector dot products that need to be performed.
    int step = kc*sizeof(T)*CHAR_BIT/regLen;
    int num = regLen/(sizeof(T)*CHAR_BIT);
    //__m256 CAux;
    for(int i=0;i<mr;i++)
    {
        for(int j=0;j<nr;j++)
        {
            if(CRowStart+i>=m || CColStart+j>=n)
                continue;
            //compute c_mn
            
            T tmp=0;
            for(int s=0;s<step;s++)
            {
                tmp += avx2VecDotPro(A+i*kc+s*num,B+j*kc+s*num);
            }
            //printf("%.0f,",tmp);
            //C[(i)*n+CColStart+j] += tmp;
            C->Add2D(tmp,CRowStart+i,CColStart+j);
            //C[(CRowStart+i)*n+CColStart+j] += CRowStart+i+CColStart+j;
            //C[(CRowStart+i)*n+CColStart+j] = CRowStart+i+1;
            //C[(i)*n+CColStart+j] = CColStart+j+1;
            //printf("|%d",i*n+CColStart+j);
        }
        //printf("\n");
    }
    //printf("================\n");
}

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
T* packA(T* const A, int const m, int const k, int const mc, int const kc)
{
    //int mm=m%mc?m+mc:m;
    //int kk=k%kc?k+kc:k;
    //T* ATmp = new T[mm*kk];
    // Indicates which element in ATmp is currently being assigned.
    int cnt=0;
    int kStep = (k+kc-1)/kc;
    int mStep = (m+mc-1)/mc;
    T* packedBuf = new T[kStep*mStep*mc*kc];
    //packedBuf = new T[kStep*mStep*mc*kc];
    
    for(int ks=0;ks<kStep;ks++)
    {
        for(int ms=0;ms<mStep;ms++)
        {
            for(int r=0;r<mc;r++)
            {
                for(int c=0;c<kc;c++)
                {
                    int row,col;
                    row = ms*mc+r;
                    col = ks*kc+c;
                    //ATmp[cnt++]=row<m&&col<k?*((T*)A+row*k+col):0;
                    packedBuf[cnt++]=row<m&&col<k?A[row*k+col]:0;
                }
            }
        }
    }
    return packedBuf;
    //printPackedBuf(A,kStep*mStep,mc,kc);
}

/*
 * @brief Pack B into many blocks, and adjust the storage format so that it can be accessed continuously during calculation.
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
T* packB(T* const B, int const k, int const n, int const kc, int const nr)
{
    //int kk=k%kc?k+1:k;
    //int nn=n%nr?n+1:n;
    //T* BTmp = new T[kk*nn];
    int cnt=0;
    int kStep = (k+kc-1)/kc;
    int nStep = (n+nr-1)/nr;
    T* packedBuf = new T[kStep*nStep*kc*nr];
    //packedBuf = new T[kStep*nStep*kc*nr];

    for(int ks=0;ks<kStep;ks++)
    {
        for(int ns=0;ns<nStep;ns++)
        {
            for(int c=0;c<nr;c++)
            {
                for(int r=0;r<kc;r++)
                {
                    int row,col;
                    row = ks*kc+r;
                    col = ns*nr+c;
                    //BTmp[cnt++] = row<k&&col<n?*((T*)B+row*n+col):0;
                    packedBuf[cnt++] = row<k&&col<n?B[row*n+col]:0;
                }
            }
        }
    }
    return packedBuf;
    //printPackedBuf(B,kStep*nStep,kc,nr);
}
template<class T>
void NiuGEMM(T* const A, T* const B, nts::XTensor* const C, int const m, int const k, int const n, struct XBlockingFactors const factors, int const regLen)
{
    
    int mGroups = (m+factors.MR-1)/factors.MR;
    int kGroups = (k+factors.KC-1)/factors.KC;
    int nGroups = (n+factors.NR-1)/factors.NR;

    //indicates how many Cache line A and B should occupy
    int nACacheLine = m*k*sizeof(T)*CHAR_BIT/LEVEL2_CACHE_LINESIZE_BITS;
    int nBCacheLine = k*factors.NR*sizeof(T)*CHAR_BIT/LEVEL1_DCACHE_LINESIZE_BITS;
    //indicates how many T elements in a Cache line.
    int ACacheLineCap=LEVEL2_CACHE_LINESIZE_BITS/(sizeof(T)*CHAR_BIT);
    int BCacheLineCap=LEVEL1_DCACHE_LINESIZE_BITS/(sizeof(T)*CHAR_BIT);
    for(int kg=0;kg<kGroups;kg++)
    {
        for(int mg=0;mg<mGroups;mg++)
        {
            // load A to L2 Cache
            //for(int n=0;n<nACacheLine;n++)
            //{
                //_mm_prefetch(A+n*ACacheLineCap,_MM_HINT_T2);
            //}
            for(int ng=0;ng<nGroups;ng++)
            {
                // load B_j to L1 Cache
                //for(int n=0;n<nBCacheLine;n++)
                //{
                    //_mm_prefetch(B+n*BCacheLineCap,_MM_HINT_T1);
                //}
                int AStart = (kg*mGroups+mg)*factors.MR*factors.KC;
                int BStart = (kg*nGroups+ng)*factors.KC*factors.NR;
                int CStart = mg*factors.MR*n;
                int CRowStart = mg*factors.MR;
                int CColStart = ng*factors.NR;
                kernel(A+AStart, B+BStart, C, factors.MR, factors.KC, factors.NR, regLen, m, n, CRowStart, CColStart);
            }
        }
    }
}
