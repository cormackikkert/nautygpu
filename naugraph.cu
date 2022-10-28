/*****************************************************************************
 *                                                                            *
 *  Graph-specific auxiliary source file for version 2.7 of nauty.            *
 *                                                                            *
 *   Copyright (1984-2019) Brendan McKay.  All rights reserved.               *
 *   Subject to waivers and disclaimers in nauty.h.                           *
 *                                                                            *
 *   CHANGE HISTORY                                                           *
 *       16-Nov-00 : initial creation out of nautil.c                         *
 *       22-Apr-01 : added aproto line for Magma                              *
 *                   EXTDEFS is no longer required                            *
 *                   removed dynamic allocation from refine1()                *
 *       21-Nov-01 : use NAUTYREQUIRED in naugraph_check()                    *
 *       23-Nov-06 : add targetcell(); make bestcell() local                  *
 *       10-Dec-06 : remove BIGNAUTY                                          *
 *       10-Nov-09 : remove shortish and permutation types                    *
 *       23-May-10 : add densenauty()                                         *
 *       15-Jan-12 : add TLS_ATTR attributes                                  *
 *       23-Jan-13 : add some parens to make icc happy                        *
 *       15-Oct-19 : fix default size of dnwork[] to match densenauty()       *
 *                                                                            *
 *****************************************************************************/

#define ONE_WORD_SETS
#include "nauty.h"
#include "helper_cuda.h"
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/transform_scan.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <cuda_profiler_api.h>

#define PAR_TRIVIAL
#define PAR_NON_TRIVIAL

/* macros for hash-codes: */
#define MASH(l,i) ((((l) ^ 065435) + (i)) & 077777)
#define MASHATOMIC(l,i) ((((l) ^ 065435) + (i)) & 077777)//\
    atomicXor((int*)&l, (int)065435); atomicAdd((int*)&l, (int)i); atomicAnd((int*)&i,(int)077777);
/* : expression whose long value depends only on long l and int/long i.
   Anything goes, preferably non-commutative. */

#define CLEANUP(l) ((int)((l) % 077777))
/* : expression whose value depends on long l and is less than 077777
   when converted to int then short.  Anything goes. */

#if  MAXM==1
#define M 1
#else
#define M m
#endif

#define ADDELEMENT1GPU(setadd,pos)  (*(setadd) |= BITTGPU[pos])
#define ISELEMENT1GPU(setadd,pos)   ((*(setadd) & BITTGPU[pos]) != 0)

#define ISELEMENT0GPU(setadd,pos) (((setadd)[SETWD(pos)] & BITTGPU[SETBT(pos)]) != 0)
#define ADDELEMENT0GPU(setadd,pos)  ((setadd)[SETWD(pos)] |= BITTGPU[SETBT(pos)])

#if WORDSIZE==64
#define ADDELEMENT1GPUATOMIC(setadd,pos)  atomicOr((unsigned long long*) setadd, (unsigned long long)BITTGPU[pos])
#define ADDELEMENT0GPUATOMIC(setadd,pos)  atomicOr((unsigned long long*) (setadd+SETWD(pos)), (unsigned long long)BITTGPU[SETBT(pos)])
#endif


#define BITTGPU bitgpu

#if  (MAXM==1) && defined(ONE_WORD_SETS)
#define ADDELEMENTGPU ADDELEMENT1GPU
#define ISELEMENTGPU ISELEMENT1GPU
#define ADDELEMENTGPUATOMIC ADDELEMENT1GPUATOMIC
#else
#define ADDELEMENTGPU ADDELEMENT0GPU
#define ISELEMENTGPU ISELEMENT0GPU
#define ADDELEMENTGPUATOMIC ADDELEMENT0GPUATOMIC
#endif



#if  WORDSIZE==64
#ifdef SETWORD_LONGLONG
static __device__ __constant__
setword bitgpu[] =  {01000000000000000000000LL,0400000000000000000000LL,
    0200000000000000000000LL,0100000000000000000000LL,
    040000000000000000000LL,020000000000000000000LL,
    010000000000000000000LL,04000000000000000000LL,
    02000000000000000000LL,01000000000000000000LL,
    0400000000000000000LL,0200000000000000000LL,
    0100000000000000000LL,040000000000000000LL,
    020000000000000000LL,010000000000000000LL,
    04000000000000000LL,02000000000000000LL,
    01000000000000000LL,0400000000000000LL,0200000000000000LL,
    0100000000000000LL,040000000000000LL,020000000000000LL,
    010000000000000LL,04000000000000LL,02000000000000LL,
    01000000000000LL,0400000000000LL,0200000000000LL,
    0100000000000LL,040000000000LL,020000000000LL,010000000000LL,
    04000000000LL,02000000000LL,01000000000LL,0400000000LL,
    0200000000LL,0100000000LL,040000000LL,020000000LL,
    010000000LL,04000000LL,02000000LL,01000000LL,0400000LL,
    0200000LL,0100000LL,040000LL,020000LL,010000LL,04000LL,
    02000LL,01000LL,0400LL,0200LL,0100LL,040LL,020LL,010LL,
    04LL,02LL,01LL};
#else
static __device__ __constant__ 
setword bitgpu[] = {01000000000000000000000,0400000000000000000000,
    0200000000000000000000,0100000000000000000000,
    040000000000000000000,020000000000000000000,
    010000000000000000000,04000000000000000000,
    02000000000000000000,01000000000000000000,
    0400000000000000000,0200000000000000000,
    0100000000000000000,040000000000000000,020000000000000000,
    010000000000000000,04000000000000000,02000000000000000,
    01000000000000000,0400000000000000,0200000000000000,
    0100000000000000,040000000000000,020000000000000,
    010000000000000,04000000000000,02000000000000,
    01000000000000,0400000000000,0200000000000,0100000000000,
    040000000000,020000000000,010000000000,04000000000,
    02000000000,01000000000,0400000000,0200000000,0100000000,
    040000000,020000000,010000000,04000000,02000000,01000000,
    0400000,0200000,0100000,040000,020000,010000,04000,
    02000,01000,0400,0200,0100,040,020,010,04,02,01};
#endif
#endif

#if  WORDSIZE==32
static __device__ __constant__
setword bitgpu[] = {020000000000,010000000000,04000000000,02000000000,
    01000000000,0400000000,0200000000,0100000000,040000000,
    020000000,010000000,04000000,02000000,01000000,0400000,
    0200000,0100000,040000,020000,010000,04000,02000,01000,
    0400,0200,0100,040,020,010,04,02,01};
#endif

#if WORDSIZE==16
static __device__ __constant__
setword bitgpu[] = {0100000,040000,020000,010000,04000,02000,01000,0400,0200,
    0100,040,020,010,04,02,01};
#endif

/* aproto: header new_nauty_protos.h */

dispatchvec dispatch_graph =
{isautom,testcanlab,updatecan,refine,refine1,cheapautom,targetcell,
    naugraph_freedyn,naugraph_check,NULL,NULL};

#if !MAXN
DYNALLSTATGPU(set,workset,workset_sz);
DYNALLSTAT(int,workperm,workperm_sz);
DYNALLSTAT(int,bucket,bucket_sz);
DYNALLSTAT(set,dnwork,dnwork_sz);
DYNALLSTAT(set,workset2,workset2_sz);
DYNALLSTATGPU(int,gpu_count,gpu_count_sz);
DYNALLSTATGPU(int,partition_number,partition_number_sz);
DYNALLSTATGPU(int,inv_lab,inv_lab_sz);
#else
static __managed__ TLS_ATTR set workset[MAXM];   /* used for scratch work */
static TLS_ATTR int workperm[MAXN];
static TLS_ATTR int bucket[MAXN+2];
static TLS_ATTR set dnwork[2*500*MAXM];
static __managed__ int gpu_count[MAXN]; // scratch array
static __managed__ int partition_number[MAXN]; // scratch array
static __managed__ int inv_lab[MAXN]; // scratch array
#endif

// kernel magic values
const int blocksize = 32;

//__managed__ set gptr;
__managed__ int newcells = 0;
__device__ __managed__ long longcode;
volatile __device__ __managed__ int hint;

/*****************************************************************************
 *                                                                            *
 *  isautom(g,perm,digraph,m,n) = TRUE iff perm is an automorphism of g       *
 *  (i.e., g^perm = g).  Symmetry is assumed unless digraph = TRUE.           *
 *                                                                            *
 *****************************************************************************/

    boolean
isautom(graph *g, int *perm, boolean digraph, int m, int n)
{
    set *pg;
    int pos;
    set *pgp;
    int posp,i;

    for (pg = g, i = 0; i < n; pg += M, ++i)
    {
        pgp = GRAPHROW(g,perm[i],M);
        pos = (digraph ? -1 : i);

        while ((pos = nextelement(pg,M,pos)) >= 0)
        {
            posp = perm[pos];
            if (!ISELEMENT(pgp,posp)) return FALSE;
        }
    }
    return TRUE;
}

/*****************************************************************************
 *                                                                            *
 *  testcanlab(g,canong,lab,samerows,m,n) compares g^lab to canong,           *
 *  using an ordering which is immaterial since it's only used here.  The     *
 *  value returned is -1,0,1 if g^lab <,=,> canong.  *samerows is set to      *
 *  the number of rows (0..n) of canong which are the same as those of g^lab. *
 *                                                                            *
 *  GLOBALS ACCESSED: workset<rw>,permset(),workperm<rw>                      *
 *                                                                            *
 *****************************************************************************/

    int
testcanlab(graph *g, graph *canong, int *lab, int *samerows, int m, int n)
{
    int i,j;
    set *ph;

#if !MAXN
    DYNALLOC1(int,workperm,workperm_sz,n,"testcanlab");
    DYNALLOC1(set,workset2,workset2_sz,m,"testcanlab");
#endif

    for (i = 0; i < n; ++i) workperm[lab[i]] = i;

    for (i = 0, ph = canong; i < n; ++i, ph += M)
    {
        permset(GRAPHROW(g,lab[i],M),workset2,M,workperm);
        for (j = 0; j < M; ++j)
            if (workset[j] < ph[j])
            {
                *samerows = i;
                return -1;
            }
            else if (workset[j] > ph[j])
            {
                *samerows = i;
                return 1;
            }
    }

    *samerows = n;
    return 0;
}

/*****************************************************************************
 *                                                                            *
 *  updatecan(g,canong,lab,samerows,m,n) sets canong = g^lab, assuming        *
 *  the first samerows of canong are ok already.                              *
 *                                                                            *
 *  GLOBALS ACCESSED: permset(),workperm<rw>                                  *
 *                                                                            *
 *****************************************************************************/

    void
updatecan(graph *g, graph *canong, int *lab, int samerows, int m, int n)
{
    int i;
    set *ph;

#if !MAXN
    DYNALLOC1GPU(int,workperm,workperm_sz,n,"updatecan");
#endif

    for (i = 0; i < n; ++i) workperm[lab[i]] = i;

    for (i = samerows, ph = GRAPHROW(canong,samerows,M);
            i < n; ++i, ph += M)
        permset(GRAPHROW(g,lab[i],M),ph,M,workperm);
}


/*****************************************************************************
 *                                                                            *
 *  GPU Kernels for refine                                                    *
 *                                                                            *
 *****************************************************************************/
__global__ void count_trivial_cell(set *gptr, int *lab, int* cnt, int n) {
    /* For vertex i counts how many neighbours are in the target cell (workset) */
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (0 <= i && i < n) {
        cnt[i] = ISELEMENTGPU(gptr, lab[i]);
    }
}

__global__ void count_non_trivial_cell_threaded_coalesce(graph *g, int* cnt, int n, int m, set *workset) { 
    /* For vertex i counts how many neighbours are in the target cell (workset) */
    __shared__ int cnt_shared;
    if (threadIdx.x == 0) {
        cnt_shared = 0;
    }
    __syncthreads();

    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = j / m;
    int c1 = j % m;

    int total = 0;
    if (0 <= i && i < n && 0 <= c1 && c1 < m) {
        unsigned long x;
        set set1 = *(workset + c1);
        set set2 = *(GRAPHROW(g, i, m) + c1);
        
        if ((x = (set1 & set2)) != 0) {
            atomicAdd(&cnt_shared,  __popcll(x));
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            atomicAdd(&cnt[i], cnt_shared);
        }
    }
}

struct intersects_trivial_cell
{
    __device__ intersects_trivial_cell() {
    }
__device__
  bool operator()(const int x)
  {
    return gpu_count[x] > 0;//ISELEMENT1GPU(gptr, lab[x]);
  }
};

__global__ void update_active_cells(int *ptn, int level,
        int *cnt, set *active, int n) {
    /* Looks at cnt to see where a cell has been split, and notes the
       splits in active and ptn */
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < 0 || i >= n) return;
    
    // Figure out some information about the current position
    bool is_end_of_global_cell = (i == n-1) || (ptn[i] <= level);
    bool is_end_of_cell = (cnt[i] != cnt[i+1]) || is_end_of_global_cell;
    bool is_start_of_global_cell = (i == 0) || (ptn[i-1] <= level);
    bool is_start_of_cell = (cnt[i-1] != cnt[i]) || is_start_of_global_cell;

    if (is_start_of_global_cell && is_end_of_global_cell) return;
   
    if (is_start_of_cell) {
        if (!is_start_of_global_cell) {
            // Mark and add the cell to active
            // There could be race conditions if different threads handle this
            // (found out the hard way)
            ADDELEMENTGPUATOMIC(active,i);
            ptn[i-1] = level;
            if (is_end_of_cell) {
                hint = i;
            }
        }
        // TODO: This isn't actually atomic, doesn't seem to throw errors
        // but likely effects pruning speed, need to investigate. Nonetheless
        // isn't necessary for the refinement procedure.
        longcode = MASHATOMIC(longcode, i + cnt[i]);
    }
}

// Unary function for thrust
struct is_end_of_partition
{
    int level;
    is_end_of_partition(int _level) {
        level = _level;
    }
  __host__ __device__
  int operator()(const int x)
  {
    return x <= level;
  }
};

/*****************************************************************************
 *                                                                            *
 *  refine(g,lab,ptn,level,numcells,count,active,code,m,n) performs a         *
 *  refinement operation on the partition at the specified level of the       *
 *  partition nest (lab,ptn).  *numcells is assumed to contain the number of  *
 *  cells on input, and is updated.  The initial set of active cells (alpha   *
 *  in the paper) is specified in the set active.  Precisely, x is in active  *
 *  iff the cell starting at index x in lab is active.                        *
 *  The resulting partition is equitable if active is correct (see the paper  *
 *  and the Guide).                                                           *
 *  *code is set to a value which depends on the fine detail of the           *
 *  algorithm, but which is independent of the labelling of the graph.        *
 *  count is used for work space.                                             *
 *                                                                            *
 *  GLOBALS ACCESSED:  workset<w>,bit<r>,nextelement(),bucket<w>,workperm<w>  *
 *                                                                            *
 *****************************************************************************/

    void
refine(graph *g, int *lab, int *ptn, int level, int *numcells,
        int *count, set *active, int *code, int m, int n)
{
int i,c1,c2,labc1;
setword x;
set *set1,*set2;
int split1,split2,cell1,cell2;
int cnt,bmin,bmax;
int maxcell,maxpos,hint;
set *gptr;
       

static int gridsize = (n + blocksize - 1) / blocksize;

#if !MAXN
DYNALLOC1(int,workperm,workperm_sz,n,"refine");
DYNALLOC1GPU(set,workset,workset_sz,m,"refine");
DYNALLOC1(int,bucket,bucket_sz,n+2,"refine");
DYNALLOC1GPU(int,gpu_count,gpu_count_sz,n,"refine");
DYNALLOC1GPU(int,inv_lab,inv_lab_sz,n,"refine");
DYNALLOC1GPU(int,partition_number,partition_number_sz,n,"refine");
#endif

longcode = *numcells;
split1 = -1;
hint = 0;

// Calculate partition_number[i], indicates the colour [0 0 0 1 1 1 2 2 2 ... ]
thrust::copy(thrust::cuda::par, ptn, ptn+n, partition_number);

thrust::transform_exclusive_scan(
        thrust::cuda::par,
        partition_number,
        partition_number+n,
        partition_number,
        is_end_of_partition(level),
        0,
        thrust::plus<int>()
);

while (*numcells < n && ((split1 = hint, ISELEMENT(active,split1))
            || (split1 = nextelement(active,M,split1)) >= 0
            || (split1 = nextelement(active,M,-1)) >= 0))
{
    DELELEMENT(active,split1);
    for (split2 = split1; ptn[split2] > level; ++split2) {}
    longcode = MASH(longcode,split1+split2);

    if (split1 == split2) {      
        /* trivial splitting cell */
        gptr = GRAPHROW(g,lab[split1],M);
        count_trivial_cell<<<gridsize, blocksize>>>(gptr, lab, gpu_count, n);
        cudaDeviceSynchronize();
        
        auto it = thrust::make_zip_iterator(thrust::make_tuple(partition_number, gpu_count));
        thrust::stable_sort_by_key(
            thrust::cuda::par,
            it,
            it+n,
            lab
        ); 
        
        update_active_cells<<<gridsize/3,3*blocksize>>>(ptn, level, gpu_count, active, n);
        cudaDeviceSynchronize();

    } else {
        /* nontrivial splitting cell */
        EMPTYSET(workset,m);
        for (i = split1; i <= split2; ++i)
            ADDELEMENT(workset,lab[i]);
        longcode = MASH(longcode,split2-split1+1);
        
        thrust::fill(thrust::cuda::par, inv_lab, inv_lab+n, 0);
        
        count_non_trivial_cell_threaded_coalesce<<<gridsize*m/4, 4*blocksize>>>(g, inv_lab, n, m, workset);
        cudaDeviceSynchronize();
        
        // Reshuffle inv_lab, as the above kernel just places memory in coalesced format to be fast
        thrust::gather(thrust::cuda::par, lab, lab+n, inv_lab, gpu_count);
        
        thrust::stable_sort_by_key(
            thrust::cuda::par,
            thrust::make_zip_iterator(thrust::make_tuple(partition_number, gpu_count)),
            thrust::make_zip_iterator(thrust::make_tuple(partition_number, gpu_count))+n,
            lab
        ); 
        update_active_cells<<<gridsize/3,3*blocksize>>>(ptn, level, gpu_count, active, n);
        cudaDeviceSynchronize();

    }



    // Recalculate partition_number[i], indicates the colour [0 0 0 1 1 1 2 2 2 ... ]
    thrust::copy(thrust::cuda::par, ptn, ptn+n, partition_number);
    
    thrust::transform_exclusive_scan(
            thrust::cuda::par,
            partition_number,
            partition_number+n,
            partition_number,
            is_end_of_partition(level),
            0,
            thrust::plus<int>()
    );

    *numcells = partition_number[n-1]+1;
}
longcode = MASH(longcode,*numcells);
*code = CLEANUP(longcode);
}

/*****************************************************************************
 *                                                                            *
 *  refine1(g,lab,ptn,level,numcells,count,active,code,m,n) is the same as    *
 *  refine(g,lab,ptn,level,numcells,count,active,code,m,n), except that       *
 *  m==1 is assumed for greater efficiency.  The results are identical in all *
 *  respects.  See refine (above) for the specs.                              *
 *                                                                            *
 *  NOTE: Unimplemented for parallelisation (doesn't make sense too,          *
 *        parallelisation is only good for big graphs)                        *
 *****************************************************************************/

    void
refine1(graph *g, int *lab, int *ptn, int level, int *numcells,
        int *count, set *active, int *code, int m, int n)
{
    int i,c1,c2,labc1;
    setword x;
    int split1,split2,cell1,cell2;
    int cnt,bmin,bmax;
    set *gptr,workset0;
    int maxcell,maxpos;

#if !MAXN 
    DYNALLOC1(int,workperm,workperm_sz,n,"refine1"); 
    DYNALLOC1(int,bucket,bucket_sz,n+2,"refine1"); 
#endif

    longcode = *numcells;
    split1 = -1;

    hint = 0;
    while (*numcells < n && ((split1 = hint, ISELEMENT1(active,split1))
                || (split1 = nextelement(active,1,split1)) >= 0
                || (split1 = nextelement(active,1,-1)) >= 0))
    {
        DELELEMENT1(active,split1);
        for (split2 = split1; ptn[split2] > level; ++split2) {}
        longcode = MASH(longcode,split1+split2);
        if (split1 == split2)       /* trivial splitting cell */
        {
            gptr = GRAPHROW(g,lab[split1],1);
            for (cell1 = 0; cell1 < n; cell1 = cell2 + 1) 
            {
                for (cell2 = cell1; ptn[cell2] > level; ++cell2) {}
                if (cell1 == cell2) continue;
                c1 = cell1;
                c2 = cell2;
                while (c1 <= c2)
                {
                    labc1 = lab[c1];
                    if (ISELEMENT1(gptr,labc1))
                        ++c1;
                    else
                    {
                        lab[c1] = lab[c2];
                        lab[c2] = labc1;
                        --c2;
                    }
                }
                if (c2 >= cell1 && c1 <= cell2)
                {
                    ptn[c2] = level;
                    longcode = MASH(longcode,c2);
                    ++*numcells;
                    if (ISELEMENT1(active,cell1) || c2-cell1 >= cell2-c1)
                    {
                        ADDELEMENT1(active,c1);
                        if (c1 == cell2) hint = c1;
                    }
                    else
                    {
                        ADDELEMENT1(active,cell1);
                        if (c2 == cell1) hint = cell1;
                    }
                }
            }
        }

        else        /* nontrivial splitting cell */
        {
            workset0 = 0;
            for (i = split1; i <= split2; ++i)
                ADDELEMENT1(&workset0,lab[i]);
            longcode = MASH(longcode,split2-split1+1);

            for (cell1 = 0; cell1 < n; cell1 = cell2 + 1)
            {
                for (cell2 = cell1; ptn[cell2] > level; ++cell2) {}
                if (cell1 == cell2) continue;
                i = cell1;
                if ((x = workset0 & g[lab[i]]) != 0)
                    cnt = POPCOUNT(x);
                else
                    cnt = 0;
                count[i] = bmin = bmax = cnt;
                bucket[cnt] = 1;
                while (++i <= cell2)
                {
                    if ((x = workset0 & g[lab[i]]) != 0)
                        cnt = POPCOUNT(x);
                    else
                        cnt = 0;
                    while (bmin > cnt) bucket[--bmin] = 0;
                    while (bmax < cnt) bucket[++bmax] = 0;
                    ++bucket[cnt];
                    count[i] = cnt;
                }
                if (bmin == bmax)
                {
                    longcode = MASH(longcode,bmin+cell1);
                    continue;
                }
                c1 = cell1;
                maxcell = -1;
                for (i = bmin; i <= bmax; ++i)
                    if (bucket[i])
                    {
                        c2 = c1 + bucket[i];
                        bucket[i] = c1;
                        longcode = MASH(longcode,i+c1);
                        if (c2-c1 > maxcell)
                        {
                            maxcell = c2-c1;
                            maxpos = c1;
                        }
                        if (c1 != cell1)
                        {
                            ADDELEMENT1(active,c1);
                            if (c2-c1 == 1) hint = c1;
                            ++*numcells;
                        }
                        if (c2 <= cell2) ptn[c2-1] = level;
                        c1 = c2;
                    }
                for (i = cell1; i <= cell2; ++i)
                    workperm[bucket[count[i]]++] = lab[i];
                for (i = cell1; i <= cell2; ++i) lab[i] = workperm[i];
                if (!ISELEMENT1(active,cell1))
                {
                    ADDELEMENT1(active,cell1);
                    DELELEMENT1(active,maxpos);
                }
            }
        }
    }

    longcode = MASH(longcode,*numcells);
    *code = CLEANUP(longcode);
}

/*****************************************************************************
 *                                                                            *
 *  cheapautom(ptn,level,digraph,n) returns TRUE if the partition at the      *
 *  specified level in the partition nest (lab,ptn) {lab is not needed here}  *
 *  satisfies a simple sufficient condition for its cells to be the orbits of *
 *  some subgroup of the automorphism group.  Otherwise it returns FALSE.     *
 *  It always returns FALSE if digraph!=FALSE.                                *
 *                                                                            *
 *  nauty assumes that this function will always return TRUE for any          *
 *  partition finer than one for which it returns TRUE.                       *
 *                                                                            *
 *****************************************************************************/

    boolean
cheapautom(int *ptn, int level, boolean digraph, int n)
{
    int i,k,nnt;

    if (digraph) return FALSE;

    k = n;
    nnt = 0;
    for (i = 0; i < n; ++i)
    {
        --k;
        if (ptn[i] > level)
        {
            ++nnt;
            while (ptn[++i] > level) {}
        }
    }

    return (k <= nnt + 1 || k <= 4);
}

/*****************************************************************************
 *                                                                            *
 *  bestcell(g,lab,ptn,level,tc_level,m,n) returns the index in lab of the    *
 *  start of the "best non-singleton cell" for fixing.  If there is no        *
 *  non-singleton cell it returns n.                                          *
 *  This implementation finds the first cell which is non-trivially joined    *
 *  to the greatest number of other cells.                                    *
 *                                                                            *
 *  GLOBALS ACCESSED: bit<r>,workperm<rw>,workset<rw>,bucket<rw>              *
 *                                                                            *
 *****************************************************************************/

    static int
bestcell(graph *g, int *lab, int *ptn, int level, int tc_level, int m, int n)
{
    int i;
    set *gp;
    setword setword1,setword2;
    int v1,v2,nnt;

#if !MAXN 
    DYNALLOC1(int,workperm,workperm_sz,n,"bestcell"); 
    DYNALLOC1GPU(set,workset,workset_sz,m,"bestcell"); 
    DYNALLOC1(int,bucket,bucket_sz,n+2,"bestcell"); 
#endif

    /* find non-singleton cells: put starts in workperm[0..nnt-1] */

    i = nnt = 0;

    while (i < n)
    {
        if (ptn[i] > level)
        {
            workperm[nnt++] = i;
            while (ptn[i] > level) ++i;
        }
        ++i;
    }

    if (nnt == 0) return n;

    /* set bucket[i] to # non-trivial neighbours of n.s. cell i */

    for (i = nnt; --i >= 0;) bucket[i] = 0;

    for (v2 = 1; v2 < nnt; ++v2)
    {
        EMPTYSET(workset,m);
        i = workperm[v2] - 1;
        do
        {
            ++i;
            ADDELEMENT(workset,lab[i]);
        }
        while (ptn[i] > level);
        for (v1 = 0; v1 < v2; ++v1)
        {
            gp = GRAPHROW(g,lab[workperm[v1]],m);
#if  MAXM==1
            setword1 = *workset & *gp;
            setword2 = *workset & ~*gp;
#else
            setword1 = setword2 = 0;
            for (i = m; --i >= 0;)
            {
                setword1 |= workset[i] & gp[i];
                setword2 |= workset[i] & ~gp[i];
            }
#endif
            if (setword1 != 0 && setword2 != 0)
            {
                ++bucket[v1];
                ++bucket[v2];
            }
        }
    }

    /* find first greatest bucket value */

    v1 = 0;
    v2 = bucket[0];
    for (i = 1; i < nnt; ++i)
        if (bucket[i] > v2)
        {
            v1 = i;
            v2 = bucket[i];
        }

    return (int)workperm[v1];
}

/*****************************************************************************
 *                                                                            *
 *  targetcell(g,lab,ptn,level,tc_level,digraph,hint,m,n) returns the index   *
 *  in lab of the next cell to split.                                         *
 *  hint is a suggestion for the answer, which is obeyed if it is valid.      *
 *  Otherwise we use bestcell() up to tc_level and the first non-trivial      *
 *  cell after that.                                                          *
 *                                                                            *
 *****************************************************************************/

    int
targetcell(graph *g, int *lab, int *ptn, int level, int tc_level,
        boolean digraph, int hint, int m, int n)
{
    int i;

    if (hint >= 0 && ptn[hint] > level &&
            (hint == 0 || ptn[hint-1] <= level))
        return hint;
    else if (level <= tc_level)
        return bestcell(g,lab,ptn,level,tc_level,m,n);
    else
    {
        for (i = 0; i < n && ptn[i] <= level; ++i) {}
        return (i == n ? 0 : i);
    }
}

/*****************************************************************************
 *                                                                            *
 *  densenauty(g,lab,ptn,orbits,&options,&stats,m,n,h)                        *
 *  is a slightly simplified interface to nauty().  It allocates enough       *
 *  workspace for 500 automorphisms and checks that the densegraph dispatch   *
 *  vector is in use.                                                         *
 *                                                                            *
 *****************************************************************************/

    void
densenauty(graph *g, int *lab, int *ptn, int *orbits,
        optionblk *options, statsblk *stats, int m, int n, graph *h)
{
    if (options->dispatch != &dispatch_graph)
    {
        fprintf(ERRFILE,"Error: densenauty() needs standard options block\n");
        exit(1);
    }

#if !MAXN
    /* Don't increase 2*500*m in the next line unless you also increase
       the default declaration of dnwork[] earlier. */
    DYNALLOC1(set,dnwork,dnwork_sz,2*500*m,"densenauty malloc");
#endif

    nauty(g,lab,ptn,NULL,orbits,options,stats,dnwork,2*500*m,m,n,h);
}

/*****************************************************************************
 *                                                                            *
 *  naugraph_check() checks that this file is compiled compatibly with the    *
 *  given parameters.   If not, call exit(1).                                 *
 *                                                                            *
 *****************************************************************************/

    void
naugraph_check(int wordsize, int m, int n, int version)
{
    if (wordsize != WORDSIZE)
    {
        fprintf(ERRFILE,"Error: WORDSIZE mismatch in naugraph.c\n");
        exit(1);
    }

#if MAXN
    if (m > MAXM)
    {
        fprintf(ERRFILE,"Error: MAXM inadequate in naugraph.c\n");
        exit(1);
    }

    if (n > MAXN)
    {
        fprintf(ERRFILE,"Error: MAXN inadequate in naugraph.c\n");
        exit(1);
    }
#endif

    if (version < NAUTYREQUIRED)
    {
        fprintf(ERRFILE,"Error: naugraph.c version mismatch\n");
        exit(1);
    }
}

/*****************************************************************************
 *                                                                            *
 *  naugraph_freedyn() - free the dynamic memory in this module               *
 *                                                                            *
 *****************************************************************************/

    void
naugraph_freedyn(void)
{
#if !MAXN
    DYNFREEGPU(workset,workset_sz);
    DYNFREE(workperm,workperm_sz);
    DYNFREE(bucket,bucket_sz);
    DYNFREE(dnwork,dnwork_sz);
#endif
}
