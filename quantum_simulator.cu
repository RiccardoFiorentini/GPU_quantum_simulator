#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#define PI  (asin(1))
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


void putb(long long int, int);
void parse_circuit(char*, int*, int*, float**, float**, char**, char**);

typedef struct{
    float val[4];
}unitary;

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

__global__ void kernel_gate(float *vr, float *vi, int num_q, unitary Ur, unitary Ui, int target){
    float tmp0_r, tmp0_i, tmp1_r, tmp1_i;
    int th_id = blockIdx.x*blockDim.x + threadIdx.x;
    long long int pos0, pos1;
    
    if(th_id < (1LLU<<(num_q-1))){
        
        pos0 = ((th_id>>target)<<(target+1))|(th_id&(((1LLU)<<target)-1));
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

__global__ void kernel_cnot(float *vr, float *vi, int num_q, int control, int target){
    float tmp0_r, tmp0_i, tmp1_r, tmp1_i;
    int th_id = blockIdx.x*blockDim.x + threadIdx.x;
    long long int pos0, pos1;

    int min_idx, max_idx;
    if(th_id < (1LLU<<(num_q-2))){
        min_idx = control < target ? control : target;
        max_idx = control > target ? control : target;

        pos0 = ((th_id>>(max_idx-1))<<(max_idx+1)) | (((th_id&(((1LLU)<<(max_idx-1))-1))>>min_idx)<<(min_idx+1)) | (th_id&(((1LLU)<<min_idx)-1)) | (((1LLU)<<control));
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

int main(int argc, char *argv[]){

    int num_q, num_g, num_m;
    double *cumul;
    long long meas;
    float *gate_r, *gate_i, *d_state_vec_r, *d_state_vec_i;
    char *target, *cnot_arg;
    unitary Ur, Ui;
    float tmpFloat = 1;
    double t_start, t_end, t_exe;

    if(argc < 2){
        printf("QUANTUM CIRCUIT SIMULATOR\n");
        printf("Usage: %s <circuit_file_name>\n",argv[0]);
        exit(1);
    }

    t_start = get_time();

    parse_circuit(argv[1], &num_q, &num_g, &gate_r, &gate_i, &target, &cnot_arg);

    float sv_r[((1LLU)<<num_q)], sv_i[((1LLU)<<num_q)];

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
            numBlocks = ceil((1LLU<<(num_q-1))/(double)NUMTHREAD);
            memcpy(&Ur.val, &(gate_r[i*4]), sizeof(float)*4);
            memcpy(&Ui.val, &(gate_i[i*4]), sizeof(float)*4);
            kernel_gate<<<numBlocks, NUMTHREAD>>>(
                d_state_vec_r,
                d_state_vec_i,
                num_q,
                Ur,
                Ui,
                (int)target[i]
            );
        }else{
            numBlocks = ceil((1LLU<<(num_q-2))/(double)NUMTHREAD);
            kernel_cnot<<<numBlocks, NUMTHREAD>>>(
                d_state_vec_r,
                d_state_vec_i,
                num_q,
                (int)target[i],
                (int)cnot_arg[i]
            );
        }
        //cudaDeviceSynchronize();
        CHECK_KERNELCALL();
    }
    cudaDeviceSynchronize();
    t_end = get_time();
    t_exe = t_end - t_start;
    CHECK(cudaMemcpy(sv_r, d_state_vec_r, ((1LLU)<<num_q)*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(sv_i, d_state_vec_i, ((1LLU)<<num_q)*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_state_vec_i));
    CHECK(cudaFree(d_state_vec_r));
    
    free(gate_r);
    free(gate_i);
    free(target);
    free(cnot_arg); 

    long long unsigned max_idx;
    float max_p = -1;
    float prob;

    /*for(long long unsigned i = 0; i<((1LLU)<<num_q); i++){
        prob = sv_r[i]*sv_r[i] + sv_i[i]*sv_i[i];
        if(prob>0) printf("%llu : %f + %f i\n",i,sv_r[i],sv_i[i]);
        if(prob > max_p){
            max_p = prob;
            max_idx = i;
        }
    }*/

    //printf("MOST LIKELY MEASUREMENT: %llu (%f)\n",max_idx,max_p);
    printf("Execution time: %lf\n", t_exe);

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
