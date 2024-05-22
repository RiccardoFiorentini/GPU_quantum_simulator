#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <complex.h>
#include <time.h>

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

void execute_single_qubit_gate(complex*, int, complex[UNITARY_DIM], int);
void execute_cnot(complex*, int, int, int);
complex *compute_state_vector(char*, int*);
double *compute_state_cumulative_distribution(complex*, int);
long long int measurement(double*, int);
int putb(long long int, int);

int main(int argc, char *argv[]){

    int num_q, num_m;
    complex *v;
    double *cumul;
    long long meas;

    if(argc < 3){
        printf("QUANTUM CIRCUIT SIMULATOR\n");
        printf("Usage: %s <circuit_file_name> <number_of_measurement>\n",argv[0]);
        exit(1);
    }

    //RAND PRE-HEAT
    srand(time(NULL));
    for(int i=0; i<10; i++) rand();

    //NUMBER OF MEASUREMENTS
    num_m = atoi(argv[2]);

    //READ THE CIRCUIT FILE AND COMPUTE THE FINAL STATE VECTOR
    v = compute_state_vector(argv[1],&num_q);

    if(v==NULL){
        printf("ERROR while parsing quantum circuit\n");
        exit(1);
    }

    //OBTAIN THE CUMULATIVE DISTRUBUTION OF THE FINAL MEASUREMENT
    cumul = compute_state_cumulative_distribution(v,num_q);
    if(cumul==NULL){
        free(v);
        exit(1);
    }

    //REPEAT THE MEASUREMENTS AND OUTPUT THE RESULTS
    /*for(int i=0; i<num_m; i++){
        meas = measurement(cumul,num_q);
        printf("MEASUREMENT: ");
        putb(meas,num_q);
        printf(" (%ld)\n",meas);
    }*/

    free(v);
    free(cumul);

    return 0;
}

void execute_single_qubit_gate(complex *v, int num_q, complex U[4], int target){
    complex tmp0, tmp1;
    long long int mask = (1LLU) << target;
    for(long long int i=0; i<(1LLU)<< num_q ; i++){
        if(i < (i^mask)){
            tmp0 = v[i];
            tmp1 = v[(i^mask)];
            v[i] =          tmp0*U[0] + tmp1*U[2];
            v[(i^mask)] =   tmp0*U[1] + tmp1*U[3];
        }
    }
}

void execute_cnot(complex *v, int num_q, int control, int target){
    complex tmp0, tmp1;
    long long int c_mask = (1LLU) << control;
    long long int t_mask = (1LLU) << target;
    for(long long int i=0; i<(1LLU)<< num_q ; i++){
        if((i < (i^t_mask)) && (i&c_mask)){
            tmp0 = v[i];
            tmp1 = v[(i^t_mask)];
            v[i] =          tmp1;
            v[(i^t_mask)] = tmp0;
        }
    }
}

//function to get the time of day in seconds
double get_time(){
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

complex *compute_state_vector(char *filename, int *num_q){
    FILE *f;
    char c;
    int qubit_num = 0;
    int curr_qubit,curr_qubit2;
    char gate_name[GATE_MAX_LEN+1];
    int is_cnot, str_l;
    complex U[UNITARY_DIM];
    double arg, t_start, t_end, t_exe;

    complex *v = NULL;

    f = fopen(filename,"r");
    if(!f){
        printf("ERROR: cannot open circuit file\n");
        exit(1);
    }

    do fscanf(f,"%c",&c);
    while(c!=';' && !feof(f));
    do fscanf(f,"%c",&c);
    while((isblank(c) || c==10 || c==',' || c==';' || !isgraph(c)) && !feof(f));

    do fscanf(f,"%c",&c);
    while(c!=';' && !feof(f));
    do fscanf(f,"%c",&c);
    while((isblank(c) || c==10 || c==',' || c==';' || !isgraph(c)) && !feof(f));

    t_start = get_time();

    while(!feof(f)){
        
        while((isblank(c) || c==10 || c==',' || c==';' || !isgraph(c)) && !feof(f)){
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

        is_cnot = 0;
        if(!strcmp(gate_name,GATE_QUBIT)){
            while((c!='$' && c!='[') && !feof(f))
                fscanf(f,"%c",&c);
            
            fscanf(f,"%d",&qubit_num);

            v = (complex*) malloc(sizeof(complex)*((1LLU)<< qubit_num));
            if(!v){
                printf("Malloc error\n");
                fclose(f);
                return NULL;
            }

            for(long long int i=1; i<(1LLU)<< qubit_num ; i++)
                v[i] = 0;
            v[0] = 1.0;

            while((c!='\n' && c!=10) && !feof(f))
                fscanf(f,"%c",&c);
            continue;
        }else if(!strcmp(gate_name,GATE_CX)){
            is_cnot = 1;
        }else if(!strcmp(gate_name,GATE_X)){
            U[0] = 0.0; U[1] = 1.0;
            U[2] = 1.0; U[3] = 0.0;
        }else if(!strcmp(gate_name,GATE_SX)){
            U[0] = (1.0 + I)/2.0; U[1] = (1.0 - I)/2.0;
            U[2] = (1.0 - I)/2.0; U[3] = (1.0 + I)/2.0;
        }else if(!strcmp(gate_name,GATE_Z)){
            U[0] = 1.0; U[1] = 0.0;
            U[2] = 0.0; U[3] =-1.0;
        }else if(!strcmp(gate_name,GATE_S)){
            U[0] = 1.0; U[1] = 0.0;
            U[2] = 0.0; U[3] = cexp(I*PI/2.0);
        }else if(!strcmp(gate_name,GATE_SDG)){
            U[0] = 1.0; U[1] = 0.0;
            U[2] = 0.0; U[3] = cexp(-I*PI/2.0);
        }else if(!strcmp(gate_name,GATE_T)){
            U[0] = 1.0; U[1] = 0.0;
            U[2] = 0.0; U[3] = cexp(I*PI/4.0);
        }else if(!strcmp(gate_name,GATE_TDG)){
            U[0] = 1.0; U[1] = 0.0;
            U[2] = 0.0; U[3] = cexp(-I*PI/4.0);
        }else if(gate_name[0] == GATE_RZ[0] && gate_name[1] == GATE_RZ[1]){
            sscanf(gate_name+3,"%lf",&arg);
            U[0] = 1.0; U[1] = 0.0;
            U[2] = 0.0; U[3] = cexp(I*arg);
        }else if(!strcmp(gate_name,GATE_H)){
            U[0] = 1.0/sqrt(2.0); U[1] = 1.0/sqrt(2.0);
            U[2] = 1.0/sqrt(2.0); U[3] =-1.0/sqrt(2.0);
        }else{
            printf("Unknown token: %s\n",gate_name);
            printf("Input format: \n\n");
            printf("OPENQASM 3.0;\n");
            printf("include \"stdgates.inc\";\n");
            printf("qubit[<num_qubit>] q; or qubit q[<num_qubit>]; \\\\single quantum register \n");
            printf("<quantum_circuit>\n\n");
            printf("Supported operations: cx, x, sx, z, s, sdg, t, tdg, rz, h\n");
            fclose(f);
            if(v) free(v);
            return NULL;
        }
        
        while((c!='$' && c!='[') && !feof(f))
            fscanf(f,"%c",&c);
        fscanf(f,"%d",&curr_qubit);

        if(is_cnot){
            fscanf(f,"%c",&c);
            while((c!='$' && c!='[') && !feof(f))
                fscanf(f,"%c",&c);
            fscanf(f,"%d",&curr_qubit2);

            execute_cnot(v,qubit_num,curr_qubit,curr_qubit2);
        }else{
            execute_single_qubit_gate(v,qubit_num,U,curr_qubit);
        }

        fscanf(f,"%c",&c);
        while((isblank(c) || c==10 || c==',' || c==';' || c==']' || !isgraph(c)) && !feof(f))
            fscanf(f,"%c",&c);
    }
    t_end = get_time();
    fclose(f);
    
    t_exe = t_end - t_start;
    printf("Execution time: %lf \n", t_exe);

    (*num_q) = qubit_num;

    return v;

}

double *compute_state_cumulative_distribution(complex *v, int num_q){
    double *res = (double*) malloc(sizeof(complex)*((1LLU)<< num_q));
    if(!res){
        printf("Malloc error\n");
        return NULL;
    }
    double acc = 0.0;
    for(long long int i=0; i<(1LLU)<< num_q ; i++){
        acc += cabs(v[i])*cabs(v[i]);
        res[i] = acc;
    }
    return res;
}

long long int measurement(double *cumul_dist, int num_q){
    double randn = 0.0;
    double coeff = 1.0/RAND_MAX;
    for(int i=0; i<10; i++){
        randn += rand()*coeff;
        coeff *= 1.0/RAND_MAX;
    }

    long long int idx = 0;
    while(((cumul_dist[idx] == 0.0) || (cumul_dist[idx] < randn)) && idx < (1LLU<<num_q)-1){
        idx++;
    }
    return idx;
}

int putb(long long int n, int len){
    long long int mask = 1LLU << (len-1);
    int m_len = len;
    while(m_len){
        printf("%d",(n&mask)>>(m_len-1));
        mask >>= 1;
        m_len--;
    }
}
