//#include "../niutensor/tensor/XTensor.h"
//#include "../niutensor/tensor/core/CHeader.h"
//#include "../niutensor/tensor/function/FHeader.h"
#include <immintrin.h>
#include <cstdio>
using namespace std;
//using namespace nts;

#define GEMM_SIMD_ALIGN_SIZE 32
#define DGEMM_MC 256
#define DGEMM_NC 256
#define DGEMM_KC 128
//#define DGEMM_MC 8
//#define DGEMM_NC 8
//#define DGEMM_KC 8
//#define DGEMM_MR 8
//#define DGEMM_NR 6
#define DGEMM_MR 8
#define DGEMM_NR 8

#define Min( i, j ) ( (i)<(j) ? (i): (j) )

struct aux_s {
    float *b_next;
    //float  *b_next_s;
    char   *flag;
    int    pc;
    int    m;
    int    n;
};
typedef struct aux_s aux_t;

//micro-panel a is stored in column major, lda=DGEMM_MR.
//#define a(i,j) a[ (j)*DGEMM_MR + (i) ]
////micro-panel b is stored in row major, ldb=DGEMM_NR.
//#define b(i,j) b[ (i)*DGEMM_NR + (j) ]
////result      c is stored in column major.
//#define c(i,j) c[ (i)*ldc + (j) ]

void bl_dgemm_int_8x8(
                        int    k,
                        float *a,
                        float *b,
                        float *c,
                        unsigned long long ldc,
                        aux_t* data
                      )
{
    int l, j, i;

    for ( l = 0; l < k; ++l )
    {                 
        __m256 rc;
        __m256 rb = _mm256_loadu_ps(b+l*DGEMM_NR);
        for(i=0;i<DGEMM_MR;i++)
        {
            __m256 ra = _mm256_broadcast_ss(a+l*DGEMM_MR+i);
            __m256 aux;
            aux = _mm256_loadu_ps(c+i*ldc);
            rc = _mm256_fmadd_ps(ra,rb,aux);
            _mm256_storeu_ps(c+i*ldc,rc);
        }
    }
}

//void bl_dgemm_ukr( int    k,
                   //float *a,
                   //float *b,
                   //float *c,
                   //unsigned long long ldc,
                   //aux_t* data )
//{
    //int l, j, i;

    //for ( l = 0; l < k; ++l )
    //{                 
        //for ( j = 0; j < DGEMM_NR; ++j )
        //{ 
            //for ( i = 0; i < DGEMM_MR; ++i )
            //{ 
                //c( i, j ) += a( i, l ) * b( l, j );
            //}
        //}
    //}

//}

//#define BL_MICRO_KERNEL bl_dgemm_ukr
#define BL_MICRO_KERNEL bl_dgemm_int_8x8

static void (*bl_micro_kernel) (
        int    k,
        float *a,
        float *b,
        float *c,
        unsigned long long ldc,
        aux_t  *aux
        ) = {
        BL_MICRO_KERNEL
};
float *bl_malloc_aligned(
        int    m,
        int    n,
        int    size
        )
{
    float *ptr;
    int    err;

    err = posix_memalign( (void**)&ptr, (size_t)GEMM_SIMD_ALIGN_SIZE, size * m * n );

    if ( err ) {
        printf( "bl_malloc_aligned(): posix_memalign() failures" );
        exit( 1 );    
    }

    return ptr;
}

/*
 * @brief pack the block in col major format.
 *
 * @param m number of rows of the block
 * @param k number of columns of the block
 * @param XA The address of the first element of the matrix in the row corresponding to the first column of the block
 * e.g. 
 * a00 a01 a02 a03    the block is a22 a23
 * a10 a11 a12 a13                 a32 a33
 * a20 a21 a22 a23
 * a30 a31 a32 a33
 * then XA should be address of a20
 * @param ldXA leading dimension
 * @param offseta the col index of the first element of the block in the matrix
 * e.g. the matrix and block are shown above. then the offseta should be 2
 * @param packA store the packed block.
 * e.g. the matrix and block are shown above. then the packA should be
 * a22 a32 a23 a33
 */
inline void packA_mcxkc_d(
        int    m,
        int    k,
        float *XA,
        int    ldXA,
        int    offseta,
        float *packA
        )
{
    int    i, p;
    // @a_pntr store the address of each element in the first column of the block.
    float *a_pntr[ DGEMM_MR ];

    for ( i = 0; i < m; i ++ ) {
        //a_pntr[ i ] = XA + ( offseta + i );
        a_pntr[ i ] = XA + i*ldXA + offseta;
    }

    for ( i = m; i < DGEMM_MR; i ++ ) {
        a_pntr[ i ] = XA + ( offseta + 0 );
    }

    for ( p = 0; p < k; p ++ ) {
        for ( i = 0; i < DGEMM_MR; i ++ ) {
            if(i>=m)
            {
                *packA=0;
                packA++;
                continue;
            }
            *packA = *a_pntr[ i ];
            packA ++;
            //a_pntr[ i ] = a_pntr[ i ] + ldXA;
            a_pntr[i]++;
        }
    }
}


/*
 * @brief pack the block in row major format.
 *
 * @param n number of rows of the block
 * @param k number of columns of the block
 * @param XB The address of the first element of the matrix in the row corresponding to the first row of the block
 * e.g. 
 * b00 b01 b02 b03    the block is b22 b33
 * b10 b11 b12 b13                 b32 b33
 * b20 b21 b22 b23
 * b30 b31 b32 b33
 * then XB should be address of b20
 * @param ldXB leading dimension
 * @param offsetb the column index of the first element of the block in the matrix
 * e.g. the matrix and block are shown above. then the offsetb should be 2
 * @param packB store the packed block.
 * e.g. the matrix and block are shown above. then the packB should be
 * b22 b33 b32 b33
 */

inline void packB_kcxnc_d(
        int    n,
        int    k,
        float *XB,
        int    ldXB, // ldXB is the original n
        int    offsetb,
        float *packB
        )
{
    int    i, p; 
    // @b_pntr store the address of each element in the first row of the block.
    //float *b_pntr[ DGEMM_NR ];
    float *b_pntr[ DGEMM_KC ];

    for ( i = 0; i < k; i ++ ) {
        //b_pntr[ i ] = XB + ldXB * ( offsetb + j );
        b_pntr[ i ] = XB + ldXB * i + offsetb;
    }

    for ( i = k; i < DGEMM_KC; i ++ ) {
        b_pntr[ i ] = XB + ldXB * 0 +  offsetb;
    }

    for ( p = 0; p < k; p ++ ) {
        for ( i = 0; i < DGEMM_NR; i ++ ) {
            if(i>=n)
            {
                *packB ++ = 0;
                continue;
            }
            *packB ++ = *b_pntr[ p ] ++;
        }
    }
}

/*
 * @param m MC
 * @param n NC
 * @param k KC
 * @param packA packed block A. size is MCxKC
 * @param packB packed block B. size is KCxNC
 * @param C the start address of corresponding block of the result matrix. notes that the matrix is stored in column major.
 */
void bl_macro_kernel(
        int    m,
        int    n,
        int    k,
        float *packA,
        float *packB,
        float *C,
        int    ldc
        )
{
    int bl_ic_nt;
    int    i, ii, j;
    aux_t  aux;
    char *str;

    aux.b_next = packB;

    for ( j = 0; j < n; j += DGEMM_NR ) {                        // 2-th loop around micro-kernel
        aux.n  = Min( n - j, DGEMM_NR );
        for ( i = 0; i < m; i += DGEMM_MR ) {                    // 1-th loop around micro-kernel
            aux.m = Min( m - i, DGEMM_MR );
            // what does this piece of code mean ?
            if ( i + DGEMM_MR >= m ) {
                aux.b_next += DGEMM_NR * k;
            }

            ( *bl_micro_kernel ) (
                    k,
                    &packA[ i * k ],
                    &packB[ j * k ],
                    &C[ i * ldc + j ],
                    (unsigned long long) ldc,
                    &aux
                    );
        }                                                        // 1-th loop around micro-kernel
    }                                                            // 2-th loop around micro-kernel
}

// C must be aligned
void bl_dgemm(
        int    m,
        int    n,
        int    k,
        float *XA,
        int    lda,
        float *XB,
        int    ldb,
        float *C,        // must be aligned
        int    ldc        // ldc must also be aligned
        //float *XA_TRANS
        )
{
    int    i, j, p, bl_ic_nt;
    int    ic, ib, jc, jb, pc, pb;
    int    ir, jr;
    float *packA, *packB;
    char   *str;

    // Early return if possible
    if ( m == 0 || n == 0 || k == 0 ) {
        printf( "bl_dgemm(): early return\n" );
        return;
    }

    // sequential is the default situation
    bl_ic_nt = 1;
    // check the environment variable
    //str = getenv( "BLISLAB_IC_NT" );
    //if ( str != NULL ) {
        //bl_ic_nt = (int)strtol( str, NULL, 10 );
    //}

    // Allocate packing buffers
    packA  = bl_malloc_aligned( DGEMM_KC, ( DGEMM_MC + 1 ) * bl_ic_nt, sizeof(float) );
    packB  = bl_malloc_aligned( DGEMM_KC, ( DGEMM_NC + 1 )           , sizeof(float) );

    for ( jc = 0; jc < n; jc += DGEMM_NC ) {                                       // 5-th loop around micro-kernel
        jb = Min( n - jc, DGEMM_NC );
        for ( pc = 0; pc < k; pc += DGEMM_KC ) {                                   // 4-th loop around micro-kernel
        // so for each pc, we will update packA and packB.
            // @pb row number of a block
            pb = Min( k - pc, DGEMM_KC );

            for ( j = 0; j < jb; j += DGEMM_NR ) {
                // so I want to know the order in which B is packed.
                packB_kcxnc_d(
                        Min( jb - j, DGEMM_NR ),
                        pb,
                        &XB[ pc*ldb ],
                        ldb, // should be ldXB instead
                        jc + j,
                        &packB[ j * pb ]
                        );
            }


            // so for A, the pack order is: K->M
            for ( ic = 0; ic < m; ic += DGEMM_MC ) {                               // 3-rd loop around micro-kernel
            // so for each ic, we execute a kernel computation.

                ib = Min( m - ic, DGEMM_MC );

                for ( i = 0; i < ib; i += DGEMM_MR ) {
                    packA_mcxkc_d(
                            Min( ib - i, DGEMM_MR ),
                            pb,
                            &XA[ (ic+i)*lda ],
                            lda,
                            pc,
                            &packA[ 0 * DGEMM_MC * pb + i * pb ]
                            );
                }

                //ib = min( m - ic, DGEMM_MC );
                //jb = min( n - jc, DGEMM_NC );
                //pb = min( k - pc, DGEMM_KC );
                //packA  = bl_malloc_aligned( DGEMM_KC, ( DGEMM_MC + 1 ) * bl_ic_nt, sizeof(double) );
                //packB  = bl_malloc_aligned( DGEMM_KC, ( DGEMM_NC + 1 )           , sizeof(double) );
                //for ( jc = 0; jc < n; jc += DGEMM_NC ) {                                       // 5-th loop around micro-kernel
                //for ( ic = 0; ic < m; ic += DGEMM_MC ) {                               // 3-rd loop around micro-kernel
                bl_macro_kernel(
                        ib,
                        jb,
                        pb,
                        packA  + 0 * DGEMM_MC * pb,
                        //packA_ref  + 0 * DGEMM_MC * pb,
                        packB,
                        &C[ ic * ldc + jc ], 
                        ldc
                        );
            }                                                                     // End 3.rd loop around micro-kernel
        }                                                                         // End 4.th loop around micro-kernel
    }                                                                             // End 5.th loop around micro-kernel

    free( packA );
    free( packB );
}
