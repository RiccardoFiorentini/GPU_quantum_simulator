#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <vector>
#include <sys/time.h>
#include <cuda_runtime.h>

#define PI  (2*asin(1))
#define GATE_MAX_LEN 63
#define UNITARY_DIM 4

#define GATE_QUBIT "qubit"
#define GATE_CX "cx"
#define GATE_X "x"
#define GATE_SX "sx"
#define GATE_Z "z"
#define GATE_S "s"
#define GATE_SDG "sdg"
#define GATE_T "t"
#define GATE_TDG "tdg"
#define GATE_RZ "rz"
#define GATE_H "h"

#define IS_NOT_CX_OP 127
#define IS_CX_OP 1
#define NUMTHREAD 1024
#define NUMBLOCKS 1
#define MAX_COSTANT 2048

#define CHECK(call)                                                                       \
{                                                                                     \
    const cudaError_t err = call;                                                     \
    if (err != cudaSuccess)                                                           \
    {                                                                                 \
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE);                                                           \
    }                                                                                 \
}
 
#define CHECK_KERNELCALL()                                                                \
{                                                                                     \
    const cudaError_t err = cudaGetLastError();                                       \
    if (err != cudaSuccess)                                                           \
    {                                                                                 \
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE);                                                           \
    }                                                                                 \
}

typedef struct{
    float val[4];
}unitary;

__constant__ unitary d_Ur[MAX_COSTANT];
__constant__ unitary d_Ui[MAX_COSTANT];
__constant__ char d_Targ[MAX_COSTANT];
__constant__ char d_Arg[MAX_COSTANT];

void putb(long long int, int);
void parse_circuit(char*, int*, int*, float**, float**, char**, char**);



//function to get the time of day in seconds
double get_time(){
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

__global__ void init_state_vector(float *vr, float *vi, int num_q){
    int th_id = blockIdx.x*blockDim.x + threadIdx.x;
    if(th_id < (1LLU<<(num_q))){
        vr[th_id] = (th_id==0);
        vi[th_id] = 0.0;
    }
}

__device__ void gate_costant(float *vr, float *vi, int num_q, int op,  int target){
    float tmp0_r, tmp0_i, tmp1_r, tmp1_i;
    int th_id = blockIdx.x*blockDim.x + threadIdx.x;
    long long int pos0, pos1;
    unitary Ur = d_Ur[op];
    unitary Ui = d_Ui[op];
    
    for(int i = th_id; i<(1LLU<<(num_q-1)); i+=NUMTHREAD){
        
        pos0 = ((i>>target)<<(target+1))|(i&(((1LLU)<<target)-1));
        pos1 = pos0|((1LLU)<<target);

        tmp0_r = vr[pos0]*Ur.val[0] - vi[pos0]*Ui.val[0] + vr[pos1]*Ur.val[1] - vi[pos1]*Ui.val[1];
        tmp0_i = vr[pos0]*Ui.val[0] + vi[pos0]*Ur.val[0] + vr[pos1]*Ui.val[1] + vi[pos1]*Ur.val[1];

        tmp1_r = vr[pos0]*Ur.val[2] - vi[pos0]*Ui.val[2] + vr[pos1]*Ur.val[3] - vi[pos1]*Ui.val[3];
        tmp1_i = vr[pos0]*Ui.val[2] + vi[pos0]*Ur.val[2] + vr[pos1]*Ui.val[3] + vi[pos1]*Ur.val[3];

        vr[pos0] = tmp0_r;
        vr[pos1] = tmp1_r;
        vi[pos0] = tmp0_i;
        vi[pos1] = tmp1_i;
    }
}

__device__ void gate_texture(float *vr, float *vi, int num_q, int op,  int target, cudaTextureObject_t texUi, cudaTextureObject_t texUr){
    float tmp0_r, tmp0_i, tmp1_r, tmp1_i;
    int th_id = blockIdx.x*blockDim.x + threadIdx.x;
    long long int pos0, pos1;
    unitary Ur;
    unitary Ui;

    Ui.val[0] = tex1Dfetch<float>(texUi, 4 * op);
    Ui.val[1] = tex1Dfetch<float>(texUi, 4 * op + 1);
    Ui.val[2] = tex1Dfetch<float>(texUi, 4 * op + 2);
    Ui.val[3] = tex1Dfetch<float>(texUi, 4 * op + 3);

    Ur.val[0] = tex1Dfetch<float>(texUr, 4 * op);
    Ur.val[1] = tex1Dfetch<float>(texUr, 4 * op + 1);
    Ur.val[2] = tex1Dfetch<float>(texUr, 4 * op + 2);
    Ur.val[3] = tex1Dfetch<float>(texUr, 4 * op + 3);
    
    for(int i = th_id; i<(1LLU<<(num_q-1)); i+=NUMTHREAD){
        
        pos0 = ((i>>target)<<(target+1))|(i&(((1LLU)<<target)-1));
        pos1 = pos0|((1LLU)<<target);

        tmp0_r = vr[pos0]*Ur.val[0] - vi[pos0]*Ui.val[0] + vr[pos1]*Ur.val[1] - vi[pos1]*Ui.val[1];
        tmp0_i = vr[pos0]*Ui.val[0] + vi[pos0]*Ur.val[0] + vr[pos1]*Ui.val[1] + vi[pos1]*Ur.val[1];

        tmp1_r = vr[pos0]*Ur.val[2] - vi[pos0]*Ui.val[2] + vr[pos1]*Ur.val[3] - vi[pos1]*Ui.val[3];
        tmp1_i = vr[pos0]*Ui.val[2] + vi[pos0]*Ur.val[2] + vr[pos1]*Ui.val[3] + vi[pos1]*Ur.val[3];

        vr[pos0] = tmp0_r;
        vr[pos1] = tmp1_r;
        vi[pos0] = tmp0_i;
        vi[pos1] = tmp1_i;
    }
}

__device__ void cnot(float *vr, float *vi, int num_q, int control, int target){
    float tmp0_r, tmp0_i, tmp1_r, tmp1_i;
    int th_id = blockIdx.x*blockDim.x + threadIdx.x;
    long long int pos0, pos1;
    int min_idx, max_idx;

    for(int i = th_id; i<(1LLU<<(num_q-2)); i+=NUMTHREAD){
        min_idx = control < target ? control : target;
        max_idx = control > target ? control : target;

        pos0 = ((i>>(max_idx-1))<<(max_idx+1)) | (((i&(((1LLU)<<(max_idx-1))-1))>>min_idx)<<(min_idx+1)) | (i&(((1LLU)<<min_idx)-1)) | (((1LLU)<<control));
        pos1 = pos0|((1LLU)<<target);

        tmp0_r = vr[pos1];
        tmp0_i = vi[pos1];

        tmp1_r = vr[pos0];
        tmp1_i = vi[pos0];

        vr[pos0] = tmp0_r;
        vr[pos1] = tmp1_r;
        vi[pos0] = tmp0_i;
        vi[pos1] = tmp1_i;
    }
}

__global__ void kernel_costant(int numOp, int num_q, float *vr, float *vi){    
    for(int i = 0; i < numOp; i++){
        if(d_Arg[i] == IS_NOT_CX_OP){
            gate_costant(vr, vi, num_q, i, (int)d_Targ[i]);
        }else{
            cnot(vr, vi, num_q, (int)d_Targ[i], (int)d_Arg[i]);
        }
        __syncthreads();
    }
}

__global__ void kernel_texture(int numOp, int num_q, float *vr, float *vi, cudaTextureObject_t texArg, cudaTextureObject_t texTarg, cudaTextureObject_t texUi, cudaTextureObject_t texUr){
    for(int i = 0; i < numOp; i++){
        if(d_Arg[i] == IS_NOT_CX_OP){
            gate_texture(vr, vi, num_q, i, (int)tex1D<char>(texTarg, i), texUi, texUr);
        }else{
            cnot(vr, vi, num_q, (int)tex1D<char>(texTarg, i), (int)tex1D<char>(texArg, i));
        }
        __syncthreads();
    }
}

void mm2x2(unitary *m1_r, unitary *m2_r, unitary *m1_i, unitary *m2_i){
    unitary tmp_r;
    unitary tmp_i;
    
    tmp_r.val[0] = m1_r->val[0] * m2_r->val[0] - m1_i->val[0] * m2_i->val[0] + m1_r->val[1] * m2_r->val[2] - m1_i->val[1] * m2_i->val[2];
    tmp_i.val[0] = m1_r->val[0] * m2_i->val[0] + m1_i->val[0] * m2_r->val[0] + m1_r->val[1] * m2_i->val[2] + m1_i->val[1] * m2_r->val[2];

    tmp_r.val[1] = m1_r->val[0] * m2_r->val[1] - m1_i->val[0] * m2_i->val[1] + m1_r->val[1] * m2_r->val[3] - m1_i->val[1] * m2_i->val[3];
    tmp_i.val[1] = m1_r->val[0] * m2_i->val[1] + m1_i->val[0] * m2_r->val[1] + m1_r->val[1] * m2_i->val[3] + m1_i->val[1] * m2_r->val[3];

    tmp_r.val[2] = m1_r->val[2] * m2_r->val[0] - m1_i->val[2] * m2_i->val[0] + m1_r->val[3] * m2_r->val[2] - m1_i->val[3] * m2_i->val[2];
    tmp_i.val[2] = m1_r->val[2] * m2_i->val[0] + m1_i->val[2] * m2_r->val[0] + m1_r->val[3] * m2_i->val[2] + m1_i->val[3] * m2_r->val[2];

    tmp_r.val[3] = m1_r->val[2] * m2_r->val[1] - m1_i->val[2] * m2_i->val[1] + m1_r->val[3] * m2_r->val[3] - m1_i->val[3] * m2_i->val[3];
    tmp_i.val[3] = m1_r->val[2] * m2_i->val[1] + m1_i->val[2] * m2_r->val[1] + m1_r->val[3] * m2_i->val[3] + m1_i->val[3] * m2_r->val[3];

    memcpy(m2_r, &tmp_r, sizeof(unitary));
    memcpy(m2_i, &tmp_i, sizeof(unitary));
}

void initM2(unitary *m_r, unitary *m_i){
    m_r->val[0]=1;
    m_r->val[1]=0;
    m_r->val[2]=0;
    m_r->val[3]=1;

    m_i->val[0]=0;
    m_i->val[1]=0;
    m_i->val[2]=0;
    m_i->val[3]=0;
}

bool isIdentity(unitary *m_r, unitary *m_i){
    return fabs(m_r->val[0]-1)<1e-3 && fabs(m_r->val[1])<1e-3 && fabs(m_r->val[2])<1e-3 && fabs(m_r->val[3]-1)<1e-3 && 
            fabs(m_i->val[0])<1e-3 && fabs(m_i->val[1])<1e-3 && fabs(m_i->val[2])<1e-3 && fabs(m_i->val[3])<1e-3;
}

int quicksort_partition(int *hist , int *perm, int begin, int end){
    int pivot_val,pos,tmp;

    pivot_val = hist[end];
    pos = begin;

    for(int i=begin; i<end; i++){
        if(hist[i] > pivot_val){
            tmp = hist[i];
            hist[i] = hist[pos];
            hist[pos] = tmp;

            tmp = perm[i];
            perm[i] = perm[pos];
            perm[pos] = tmp;

            pos++;
        }
    }

    tmp = hist[end];
    hist[end] = hist[pos];
    hist[pos] = tmp;

    tmp = perm[end];
    perm[end] = perm[pos];
    perm[pos] = tmp;

    return pos;
}

void quicksort(int *hist , int *perm, int begin, int end){
    if(begin >= end) return;
    
    int pivot;
    pivot = quicksort_partition(hist,perm,begin,end);
    quicksort(hist,perm,begin,pivot-1);
    quicksort(hist,perm,pivot+1,end);
}

int main(int argc, char *argv[]){
    int num_q, num_g, num_m;
    double *cumul;
    long long meas;
    float *gate_r, *gate_i, *d_state_vec_r, *d_state_vec_i;
    char *target, *cnot_arg;
    unitary Ur, Ui;
    float tmpFloat = 1;
    double t_start, t_end, t_exe;
    unitary *acc_r;
    unitary *acc_i;
    float *sv_r, *sv_i;
    char* d_Tex_Arg;
    char* d_Tex_Targ;
    unitary* d_Tex_Ui;
    unitary* d_Tex_Ur;

    //Salva operazioni in ordine, da vedere come trasformare in array
    float *VecGate_r, *VecGate_i;
    unitary *d_VecGate_r, *d_VecGate_i;
    char *VecTarg, *VecArg, *d_VecTarg, *d_VecArg;
    int numOp;
        
    if(argc < 2){
        printf("QUANTUM CIRCUIT SIMULATOR\n");
        printf("Usage: %s <circuit_file_name>\n",argv[0]);
        exit(1);
    }

    t_start = get_time();

    parse_circuit(argv[1], &num_q, &num_g, &gate_r, &gate_i, &target, &cnot_arg);
    
    acc_r = (unitary*) malloc(sizeof(unitary)*num_q);
    acc_i = (unitary*) malloc(sizeof(unitary)*num_q);
    VecGate_r = (float*) malloc(sizeof(float)*4*num_g);
    VecGate_i = (float*) malloc(sizeof(float)*4*num_g);
    VecTarg = (char*)malloc(sizeof(char)*num_g);
    VecArg  = (char*)malloc(sizeof(char)*num_g);
    numOp = 0;

    for(int i=0; i<num_q; i++){
        initM2(&acc_r[i], &acc_i[i]);
    }

    sv_r = (float*) malloc(sizeof(float)*((1LLU)<<num_q));
    sv_i = (float*) malloc(sizeof(float)*((1LLU)<<num_q));

    CHECK(cudaMalloc(&d_state_vec_r, ((1LLU)<<num_q)*sizeof(float)));
    CHECK(cudaMalloc(&d_state_vec_i, ((1LLU)<<num_q)*sizeof(float)));
    cudaDeviceSynchronize();

    int numBlocks;
    numBlocks = ceil((1LLU<<(num_q))/(double)NUMTHREAD);
    init_state_vector<<<numBlocks, NUMTHREAD>>>(
        d_state_vec_r,
        d_state_vec_i,
        num_q
    );

    cudaDeviceSynchronize();
    CHECK_KERNELCALL();

    for(int i=0; i<num_g; i++){
        if(cnot_arg[i]==IS_NOT_CX_OP){
            memcpy(&Ur.val, &(gate_r[i*4]), sizeof(float)*4); //necessario? se passassimo direttamente gate_r[i*4]?
            memcpy(&Ui.val, &(gate_i[i*4]), sizeof(float)*4);
            mm2x2(&Ur, &acc_r[target[i]], &Ui, &acc_i[target[i]]);

            /* numBlocks = ceil((1LLU<<(num_q-1))/(double)NUMTHREAD);
            memcpy(&Ur.val, &(gate_r[i*4]), sizeof(float)*4);
            memcpy(&Ui.val, &(gate_i[i*4]), sizeof(float)*4);
            kernel_gate_2<<<numBlocks, NUMTHREAD>>>(
               d_state_vec_r,
               d_state_vec_i,
               num_q,
               Ur,
               Ui,
               (int)target[i]
            ); */
        }else{
            numBlocks = ceil((1LLU<<(num_q-1))/(double)NUMTHREAD);
            if(!isIdentity(&acc_r[target[i]], &acc_i[target[i]])){
                //inserire operazione
                /*unitary *VecGate_r, *VecGate_i;
                float *VecTarg, *VecArg;
                int numOp;*/

                VecGate_i[numOp*4] = acc_i[target[i]].val[0];
                VecGate_i[numOp*4 + 1] = acc_i[target[i]].val[1];
                VecGate_i[numOp*4 + 2] = acc_i[target[i]].val[2];
                VecGate_i[numOp*4 + 3] = acc_i[target[i]].val[3];

                VecGate_r[numOp*4] = acc_r[target[i]].val[0];
                VecGate_r[numOp*4 + 1] = acc_r[target[i]].val[1];
                VecGate_r[numOp*4 + 2] = acc_r[target[i]].val[2];
                VecGate_r[numOp*4 + 3] = acc_r[target[i]].val[3];

                VecTarg[numOp] = (int)target[i];
                VecArg[numOp] = IS_NOT_CX_OP;
                numOp++;

                /*kernel_gate_2<<<numBlocks, NUMTHREAD>>>(
                    d_state_vec_r,
                    d_state_vec_i,
                    num_q,
                    acc_r[target[i]],
                    acc_i[target[i]],
                    (int)target[i]
                );*/
                initM2(&acc_r[target[i]], &acc_i[target[i]]);
            }
            
            if(!isIdentity(&acc_r[cnot_arg[i]], &acc_i[cnot_arg[i]])){
                VecGate_i[numOp*4] = acc_r[cnot_arg[i]].val[0];
                VecGate_i[numOp*4 + 1] = acc_r[cnot_arg[i]].val[1];
                VecGate_i[numOp*4 + 2] = acc_r[cnot_arg[i]].val[2];
                VecGate_i[numOp*4 + 3] = acc_r[cnot_arg[i]].val[3];

                VecGate_r[numOp*4] = acc_i[cnot_arg[i]].val[0];
                VecGate_r[numOp*4 + 1] = acc_i[cnot_arg[i]].val[1];
                VecGate_r[numOp*4 + 2] = acc_i[cnot_arg[i]].val[2];
                VecGate_r[numOp*4 + 3] = acc_i[cnot_arg[i]].val[3];

                VecTarg[numOp] = (int)cnot_arg[i];
                VecArg[numOp] = IS_NOT_CX_OP;
                numOp++;
                /*kernel_gate_2<<<numBlocks, NUMTHREAD>>>(
                    d_state_vec_r,
                    d_state_vec_i,
                    num_q,
                    acc_r[cnot_arg[i]],
                    acc_i[cnot_arg[i]],
                    (int)cnot_arg[i]
                );*/
                initM2(&acc_r[cnot_arg[i]], &acc_i[cnot_arg[i]]);
            }

            numBlocks = ceil((1LLU<<(num_q-2))/(double)NUMTHREAD);

            VecGate_i[numOp*4] = 0;
            VecGate_i[numOp*4 + 1] = 0;
            VecGate_i[numOp*4 + 2] = 0;
            VecGate_i[numOp*4 + 3] = 0;

            VecGate_r[numOp*4] = 0;
            VecGate_r[numOp*4 + 1] = 0;
            VecGate_r[numOp*4 + 2] = 0;
            VecGate_r[numOp*4 + 3] = 0;

            VecTarg[numOp] = (int)target[i];
            VecArg[numOp] = (int)cnot_arg[i];
            numOp++;

            /*kernel_cnot<<<numBlocks, NUMTHREAD>>>(
                d_state_vec_r,
                d_state_vec_i,
                num_q,
                (int)target[i],
                (int)cnot_arg[i]
            );*/
        }
        //cudaDeviceSynchronize();
        CHECK_KERNELCALL();
    }

    /*Permute the qubits*/
    int *histogram = (int*)malloc(sizeof(int)*num_q);
    int *permutation = (int*)malloc(sizeof(int)*num_q);
    for(int i=0; i<num_q; i++){
        histogram[i] = 0;
        permutation[i] = i;
    }
    for(int i=0; i<numOp; i++){
        if(VecArg[numOp] != (int)IS_NOT_CX_OP){
            histogram[VecArg[numOp]]++;
        }
        histogram[VecTarg[numOp]]++;
    }
    quicksort(histogram,permutation,0,num_q-1);
    free(histogram);
    for(int i=0; i<numOp; i++){
        if(VecArg[numOp] != (int)IS_NOT_CX_OP){
            VecArg[numOp] = permutation[VecArg[numOp]];
        }
        VecTarg[numOp] = permutation[VecTarg[numOp]];
    }
    /*Qubit permutation end*/

    numBlocks = ceil((1LLU<<(num_q-1))/(double)NUMTHREAD);
    for(int i=0; i<num_q; i++){
        if(!isIdentity(&acc_r[i], &acc_i[i])){
            VecGate_i[numOp*4] = acc_i[i].val[0];
            VecGate_i[numOp*4 + 1] = acc_i[i].val[1];
            VecGate_i[numOp*4 + 2] = acc_i[i].val[2];
            VecGate_i[numOp*4 + 3] = acc_i[i].val[3];

            VecGate_r[numOp*4] = acc_r[i].val[0];
            VecGate_r[numOp*4 + 1] = acc_r[i].val[1];
            VecGate_r[numOp*4 + 2] = acc_r[i].val[2];
            VecGate_r[numOp*4 + 3] = acc_r[i].val[3];

            VecTarg[numOp] = i;
            VecArg[numOp] = IS_NOT_CX_OP;
            numOp++;
            /*kernel_gate_2<<<numBlocks, NUMTHREAD>>>(
                d_state_vec_r,
                d_state_vec_i,
                num_q,
                acc_r[i],
                acc_i[i],
                i
            );*/
        }
    }

    if(numOp>MAX_COSTANT){
        //usa texture
        cudaMalloc((void**)&d_Tex_Arg, numOp * sizeof(char));
        cudaMalloc((void**)&d_Tex_Targ, numOp * sizeof(char));
        cudaMalloc((void**)&d_Tex_Ui, numOp * sizeof(unitary));
        cudaMalloc((void**)&d_Tex_Ur, numOp * sizeof(unitary));

        cudaMemcpy(d_Tex_Arg, VecArg, numOp * sizeof(char), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Tex_Targ, VecTarg, numOp * sizeof(char), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Tex_Ui, VecGate_i, numOp * sizeof(unitary), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Tex_Ur, VecGate_r, numOp * sizeof(unitary), cudaMemcpyHostToDevice);

        // Creazione del Texture Object per d_Arg
        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeLinear;
        resDesc.res.linear.devPtr = d_Tex_Arg;
        resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
        resDesc.res.linear.desc.x = 32; // 32 bit per float
        resDesc.res.linear.sizeInBytes = numOp * sizeof(char);

        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModePoint;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        cudaTextureObject_t texArg = 0;
        cudaCreateTextureObject(&texArg, &resDesc, &texDesc, NULL);

        // Creazione del Texture Object per d_Targ
        resDesc.res.linear.devPtr = d_Tex_Targ;
        cudaTextureObject_t texTarg = 0;
        cudaCreateTextureObject(&texTarg, &resDesc, &texDesc, NULL);

        // Creazione del Texture Object per d_Ui
        resDesc.res.linear.devPtr = d_Tex_Ui;
        resDesc.res.linear.desc.x = 32 * 4; // 128 bit per unitary
        cudaTextureObject_t texUi = 0;
        cudaCreateTextureObject(&texUi, &resDesc, &texDesc, NULL);

        // Creazione del Texture Object per d_Ur
        resDesc.res.linear.devPtr = d_Tex_Ur;
        cudaTextureObject_t texUr = 0;
        cudaCreateTextureObject(&texUr, &resDesc, &texDesc, NULL);

        kernel_texture<<<NUMBLOCKS, NUMTHREAD>>>(
            numOp,
            num_q,
            d_state_vec_r,
            d_state_vec_i,
            texArg,
            texTarg,
            texUi,
            texUr
        );

        cudaDeviceSynchronize();

        // Distruzione dei Texture Objects
        cudaDestroyTextureObject(texArg);
        cudaDestroyTextureObject(texTarg);
        cudaDestroyTextureObject(texUi);
        cudaDestroyTextureObject(texUr);

        // Deallocazione della memoria
        cudaFree(d_Tex_Arg);
        cudaFree(d_Tex_Targ);
        cudaFree(d_Tex_Ui);
        cudaFree(d_Tex_Ur);

    }else{
        //passa a GPU in constant mem l'elenco di operazioni -> da vedere come fare per la costant
        cudaMemcpyToSymbol(d_Arg, VecArg, numOp*sizeof(char));
        cudaMemcpyToSymbol(d_Targ, VecTarg, numOp*sizeof(char));
        cudaMemcpyToSymbol(d_Ui, VecGate_i, numOp*sizeof(unitary));
        cudaMemcpyToSymbol(d_Ur, VecGate_r, numOp*sizeof(unitary));

        //lancio kernel
        kernel_costant<<<NUMBLOCKS, NUMTHREAD>>>(
            numOp,
            num_q,
            d_state_vec_r,
            d_state_vec_i
        );

        cudaDeviceSynchronize();
    }
    
    t_end = get_time();
    t_exe = t_end - t_start;

    //free di VecGate,VecTarg...
    free(VecGate_i);
    free(VecGate_r);
    free(VecTarg);
    free(VecArg);       
    free(acc_i);
    free(acc_r);

    CHECK(cudaMemcpy(sv_r, d_state_vec_r, ((1LLU)<<num_q)*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(sv_i, d_state_vec_i, ((1LLU)<<num_q)*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_state_vec_i));
    CHECK(cudaFree(d_state_vec_r));
    
    free(gate_r);
    free(gate_i);
    free(target);
    free(cnot_arg);
    free(permutation);

    /* long long unsigned max_idx;
    float max_p = -1;
    float prob;

    for(long long unsigned i = 0; i<((1LLU)<<num_q); i++){
        prob = sv_r[i]*sv_r[i] + sv_i[i]*sv_i[i];
        if(prob>0) printf("%llu : %f + %f i\n",i,sv_r[i],sv_i[i]);
        if(prob > max_p){
            max_p = prob;
            max_idx = i;
        }
    }
    free(sv_r);
    free(sv_i);
    printf("MOST LIKELY MEASUREMENT: %llu (%f)\n",max_idx,max_p);
    */
    
    //printf("Execution time: %lf\n", t_exe);
    printf("%lf\n", t_exe);

    return 0;
}

void parse_circuit(char *filename, int *num_q, int *num_g, float **gate_r, float **gate_i, char **target, char **cnot_arg){
    FILE *f;
    char c;
    int qubit_num = 0;
    int curr_qubit,curr_qubit2;
    char gate_name[GATE_MAX_LEN+1];
    int str_l;
    float arg;

    f = fopen(filename,"r");
    if(!f){
        printf("ERROR: cannot open circuit file\n");
        exit(1);
    }
    
    fscanf(f,"%d",num_q);
    fscanf(f,"%d",num_g);

    (*gate_r) = (float*) malloc(sizeof(float)*(*num_g)*4);
    (*gate_i) = (float*) malloc(sizeof(float)*(*num_g)*4);
    (*target) = (char*) malloc(sizeof(char)*(*num_g));
    (*cnot_arg) = (char*) malloc(sizeof(char)*(*num_g));

    if(!(*gate_r) || !(*gate_i) || !(*cnot_arg)){
        printf("ERROR: cannot allocate circuit\n");
        free(*gate_r);
        free(*gate_i);
        free(*target);
        free(*cnot_arg);
        exit(1);
    }

    int i;
    fscanf(f,"%c",&c);
    for(i=0; i<(*num_g) && !feof(f); i++){
        while((isblank(c) || c=='\n' || c=='\r' || c==',' || c==';' || !isgraph(c)) && !feof(f)){
            fscanf(f,"%c",&c);
        }

        gate_name[0] = c;
        gate_name[1] = '\0';
        str_l = 1;
        fscanf(f,"%c",&c);
        while(isgraph(c) && c!='[' && str_l<GATE_MAX_LEN){
            gate_name[str_l] = c;
            gate_name[str_l+1] = '\0';
            str_l++;
            fscanf(f,"%c",&c);
        }

        (*cnot_arg)[i] = IS_NOT_CX_OP;
        if(!strcmp(gate_name,GATE_CX)){
            (*cnot_arg)[i] = IS_CX_OP;
        }else if(!strcmp(gate_name,GATE_X)){
            (*gate_r)[i*4]   =  0.0;
            (*gate_r)[i*4+1] =  1.0;
            (*gate_r)[i*4+2] =  1.0;
            (*gate_r)[i*4+3] =  0.0;

            (*gate_i)[i*4]   =  0.0;
            (*gate_i)[i*4+1] =  0.0;
            (*gate_i)[i*4+2] =  0.0;
            (*gate_i)[i*4+3] =  0.0;
        }else if(!strcmp(gate_name,GATE_SX)){
            (*gate_r)[i*4]   =  0.5;
            (*gate_r)[i*4+1] =  0.5;
            (*gate_r)[i*4+2] =  0.5;
            (*gate_r)[i*4+3] =  0.5;

            (*gate_i)[i*4]   =  0.5;
            (*gate_i)[i*4+1] = -0.5;
            (*gate_i)[i*4+2] = -0.5;
            (*gate_i)[i*4+3] =  0.5;
        }else if(!strcmp(gate_name,GATE_Z)){
            (*gate_r)[i*4]   =  1.0;
            (*gate_r)[i*4+1] =  0.0;
            (*gate_r)[i*4+2] =  0.0;
            (*gate_r)[i*4+3] = -1.0;

            (*gate_i)[i*4]   =  0.0;
            (*gate_i)[i*4+1] =  0.0;
            (*gate_i)[i*4+2] =  0.0;
            (*gate_i)[i*4+3] =  0.0;
        }else if(!strcmp(gate_name,GATE_S)){
            (*gate_r)[i*4]   =  1.0;
            (*gate_r)[i*4+1] =  0.0;
            (*gate_r)[i*4+2] =  0.0;
            (*gate_r)[i*4+3] =  0.0;

            (*gate_i)[i*4]   =  0.0;
            (*gate_i)[i*4+1] =  0.0;
            (*gate_i)[i*4+2] =  0.0;
            (*gate_i)[i*4+3] =  1.0;
        }else if(!strcmp(gate_name,GATE_SDG)){
            (*gate_r)[i*4]   =  1.0;
            (*gate_r)[i*4+1] =  0.0;
            (*gate_r)[i*4+2] =  0.0;
            (*gate_r)[i*4+3] =  0.0;

            (*gate_i)[i*4]   =  0.0;
            (*gate_i)[i*4+1] =  0.0;
            (*gate_i)[i*4+2] =  0.0;
            (*gate_i)[i*4+3] = -1.0;
        }else if(!strcmp(gate_name,GATE_T)){
            (*gate_r)[i*4]   =  1.0;
            (*gate_r)[i*4+1] =  0.0;
            (*gate_r)[i*4+2] =  0.0;
            (*gate_r)[i*4+3] =  cos(PI/4);

            (*gate_i)[i*4]   =  0.0;
            (*gate_i)[i*4+1] =  0.0;
            (*gate_i)[i*4+2] =  0.0;
            (*gate_i)[i*4+3] =  sin(PI/4);
        }else if(!strcmp(gate_name,GATE_TDG)){
            (*gate_r)[i*4]   =  1.0;
            (*gate_r)[i*4+1] =  0.0;
            (*gate_r)[i*4+2] =  0.0;
            (*gate_r)[i*4+3] =  cos(PI/4);

            (*gate_i)[i*4]   =  0.0;
            (*gate_i)[i*4+1] =  0.0;
            (*gate_i)[i*4+2] =  0.0;
            (*gate_i)[i*4+3] =  -sin(PI/4);
        }else if(gate_name[0] == GATE_RZ[0] && gate_name[1] == GATE_RZ[1]){
            sscanf(gate_name+3,"%f",&arg);
            (*gate_r)[i*4]   =  1.0;
            (*gate_r)[i*4+1] =  0.0;
            (*gate_r)[i*4+2] =  0.0;
            (*gate_r)[i*4+3] =  cos(arg);

            (*gate_i)[i*4]   =  0.0;
            (*gate_i)[i*4+1] =  0.0;
            (*gate_i)[i*4+2] =  0.0;
            (*gate_i)[i*4+3] =  sin(arg);
        }else if(!strcmp(gate_name,GATE_H)){
            (*gate_r)[i*4]   =  1.0/sqrt(2);
            (*gate_r)[i*4+1] =  1.0/sqrt(2);
            (*gate_r)[i*4+2] =  1.0/sqrt(2);
            (*gate_r)[i*4+3] = -1.0/sqrt(2);

            (*gate_i)[i*4]   =  0.0;
            (*gate_i)[i*4+1] =  0.0;
            (*gate_i)[i*4+2] =  0.0;
            (*gate_i)[i*4+3] =  0.0;
        }else{
            printf("Unknown token: %s\n",gate_name);
            printf("Input format: \n\n");
            printf("<num_qubit> <num_gates>\n");
            printf("<quantum_circuit> \\\\single quantum register\n\n");
            printf("Supported operations: cx, x, sx, z, s, sdg, t, tdg, rz, h\n");
            fclose(f);
            free(*gate_r);
            free(*gate_i);
            free(*target);
            free(*cnot_arg);

            exit(1);
        }
        
        while((c!='$' && c!='[') && !feof(f))
            fscanf(f,"%c",&c);
        fscanf(f,"%d",(*target)+i);

        if((*cnot_arg)[i] == IS_CX_OP){
            fscanf(f,"%c",&c);
            while((c!='$' && c!='[') && !feof(f))
                fscanf(f,"%c",&c);
            fscanf(f,"%d",(*cnot_arg)+i);
        }

        fscanf(f,"%c",&c);
        while((isblank(c) || c==10 || c==',' || c==';' || c==']' || !isgraph(c)) && !feof(f))
            fscanf(f,"%c",&c);
    }

    fclose(f);
    
    return;
}

void putb(long long int n, int len){
    long long int mask = 1LLU << (len-1);
    int m_len = len;
    while(m_len){
        printf("%d",(n&mask)>>(m_len-1));
        mask >>= 1;
        m_len--;
    }
}
